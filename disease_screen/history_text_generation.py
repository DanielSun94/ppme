import os
import re
import json
from datetime import datetime
from disease_screen_logger import logger
from util import get_data, get_local_llm_model_tokenizer
from read_data import read_diagnosis_symptom_embedding, get_symptom_index_map
from disease_screen_config import (symptom_info_dict, llm_cache_folder, data_file_code_true_path,
                                   symptom_file_path_template, fixed_question_answer_folder)

os.environ['NCCL_P2P_DISABLE'] = "1"
prompt_template_dict = {
    'eng': ("Please act as a patient and answer the following four questions based on the given electronic medical "
            "record (attached). Respond in natural sentences and keep your answers concise; "
            "do not make them overly lengthy. Please respond as if you were a real patient."
            "Do not provide any information other than the answers to the questions.\n"
            "Question 1: Please tell me your gender and age. If this information is not recorded in the "
            "electronic medical record, please respond with 'I don't know.'\n"
            "Question 2: Please tell me your past medical history (previous illnesses). "
            "If this information is not recorded in the electronic medical record, please "
            "respond with 'I don't know.' Do not provide information that is not present in the electronic medical "
            "record. Please do not reply symptoms or illness that causes your current admission.\n"
            "Question 3: Do you know any illnesses that your family members have had? If this information is not "
            "recorded in the electronic medical record, please respond with 'I don't know.'\n"
            'Question 4: Where are you feeling unwell (history of present illness)?\n'
            "Please respond in the following format:\n"
            "#Start#\n"
            "#1#: (answer)\n#2#: (answer)\n#3#: (answer)\n#4#: (answer)\n#End#\n"
            "The electronic medical record data is as follows:\n"),
    'chn': (
        '请你扮演一名病人，根据给定的电子病历（附后）回答如下四个问题。请用自然的句子简要回复，不要回复的过于冗长，请回复的像一个真实的患者一样。'
        '除了问题的回答，不要回答其它任何信息。\n'
        '问题1：请告诉我你的性别，年龄信息。如果电子病例中未记录相关信息，请回答我不知道。\n'
        '问题2：请告诉我你的既往史（以前得过什么病）？如果电子病历中未记录相关信息，请回答我不知道，'
        '请不要回答电子病历中不存在的信息。你不可以在这个问题的回答中提到你本次入院的主诉、症状和诊断疾病\n'
        '问题3：请问你家里人之前得过什么病？如果电子病例中未记录相关信息，请回答我不知道。\n'
        '问题4：请问你最近哪儿不舒服(现病史)？如果电子病例中未记录相关信息，请回答我不知道。\n\n'
        '请按照如下格式回复:\n'
        '#回答开始#\n\n'
        '#1#: (answer)\n'
        '#2#: (answer)\n'
        '#3#: (answer)\n'
        '#4#: (answer)\n'
        '#回答结束#'
        '电子病历数据如下所示:\n\n'
    )
}


def generate_fixed_question_answer(data_list, id_mapping, call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize,
                                   max_input_len, save_size, concurrent_size, visit_type_filter_key,
                                   data_source_filter_key):
    os.makedirs(fixed_question_answer_folder, exist_ok=True)

    parsed_key_set = set()
    parsed_file_list = os.listdir(fixed_question_answer_folder)
    parsed_file_count = 0
    for file in parsed_file_list:
        file_path = os.path.join(fixed_question_answer_folder, file)
        if (visit_type_filter_key not in file_path or data_source_filter_key not in file_path
                or '.json' not in file_path):
            continue
        data_json = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        for item in data_json:
            parsed_key_set.add(item[0])
            parsed_file_count += 1
    logger.info(f'Parsed file count: {parsed_file_count}')

    valid_data_list = []
    for item in data_list:
        if item[0] not in parsed_key_set:
            valid_data_list.append(item)
    logger.info('remaining data len: {}'.format(len(valid_data_list)))

    cursor, cache_list = 0, []
    while cursor < len(valid_data_list):
        prompt_list, key_list = [], []
        if cursor + concurrent_size < len(valid_data_list) - 1:
            end_idx = cursor + concurrent_size
        else:
            end_idx = len(valid_data_list) - 1
        if cursor == end_idx:
            break

        for item in valid_data_list[cursor:end_idx]:
            key, context, language = item
            prompt = prompt_template_dict[language] + context

            # 当上下文过长是截断
            token_list = tokenizer_tokenize(prompt)
            if len(token_list) > max_input_len:
                logger.info(f'context length too long (length {len(token_list)}), truncate to {max_input_len} tokens')
                token_list = token_list[:max_input_len]
            prompt_truncated = tokenizer_reverse_tokenize(token_list)
            prompt_list.append([prompt_truncated, language])
            key_list.append(key)

        cursor = end_idx

        success_flag = False
        failure_time = 0
        result_list = []
        response_list = None
        while not success_flag:
            try:
                prompt_content_list = [item[0] for item in prompt_list]
                language_list = [item[1] for item in prompt_list]
                response_list = call_llm(prompt_content_list)
                for response, key, language in zip(response_list, key_list, language_list):
                    result = parse_response(response, language)
                    result_list.append([key, result, id_mapping[key]])
                success_flag = True
            except Exception as _:
                failure_time += 1
                logger.info(f'failed, failure time: {failure_time}')
                result_list = []
                if failure_time > 3:
                    success_flag = True
                    for response, prompt, key in zip(response_list, prompt_list, key_list):
                        logger.info(f'key: {key}')
                        logger.info(f'prompt: {prompt}')
                        logger.info(response)

        if len(result_list) == 0:
            continue

        cache_list += result_list

        if len(cache_list) >= save_size:
            file_name = (datetime.now().strftime('%m%d%Y%H%M%S')
                         + f'_{visit_type_filter_key}_{data_source_filter_key}.json')
            file_path = os.path.join(fixed_question_answer_folder, file_name)
            json.dump(cache_list, open(file_path, 'w', encoding='utf-8-sig'))
            cache_list = []

    file_name = (datetime.now().strftime('%m%d%Y%H%M%S')
                 + f'_{visit_type_filter_key}_{data_source_filter_key}.json')
    file_path = os.path.join(fixed_question_answer_folder, file_name)
    json.dump(cache_list, open(file_path, 'w', encoding='utf-8-sig'))


def parse_response(response, language):
    result = {}
    if language == 'eng':
        start_idx = response.find('#Start#')
        end_idx = response.find('#End#')
    else:
        assert language == 'chn'
        start_idx = response.find('#回答开始#')
        end_idx = response.find('#回答结束#')
    response = response[start_idx + 6: end_idx]
    matches = re.findall(r"#(\d+)#: (.+?)(?=#\d+|$)", response, re.DOTALL)
    assert len(matches) == 4
    for match in matches:
        number, content = match
        if number == '1':
            result['基本信息'] = content
        elif number == '2':
            result['既往史'] = content
        elif number == '3':
            result['家族史'] = content
        else:
            assert number == '4'
            result['现病史'] = content
    return result


# 用于工程化部署所需的既往史，家族史的encoding
def main():
    top_n = 1
    digit = 4
    weight = False
    read_from_cache = True
    strategy = "ALL"
    diagnosis_lower = 20
    max_data_size = -1
    embedding_size = 1024
    save_size = 4096

    visit_type_filter_key = 'outpatient' # 'outpatient, hospitalization',
    data_source_filter_key = 'srrsh' # ('srrsh', 'mimic_iv')

    model_id = 'qwen/Qwen2___5-72B-Instruct-GPTQ-Int4'
    tensor_split_num = 2
    enforce_eager = False
    gpu_utilization = 0.9
    max_output_token = 1024
    max_model_len = 3072
    max_input_len = max_model_len - max_output_token - 10
    llm_cache = llm_cache_folder
    vllm_type = 'gptq'
    statr_index, end_index = 0, 3000000
    data_split_strategy = 'custom'
    concurrent_size = 256
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

    logger.info(f'visit_type_filter_key: {visit_type_filter_key}, data_source_filter_key: {data_source_filter_key}')
    # logger.warn('The strategy will filter a lot of data')
    symptom_path_list = [
        [symptom_file_path_template.format('chn'), 'chn'],
        [symptom_file_path_template.format('eng'), 'eng']

    ]
    filter_key = data_source_filter_key + '-' + visit_type_filter_key
    index_symptom_dict, symptom_index_dict = get_symptom_index_map(symptom_path_list)
    (train_dataset, valid_dataset, test_dataset, diagnosis_index_map, index_diagnosis_map) = (
        read_diagnosis_symptom_embedding(
            data_file_code_true_path, symptom_info_dict, symptom_index_dict, top_n, digit, weight, embedding_type=0,
            diagnosis_lower=diagnosis_lower, strategy=strategy, read_from_cache=read_from_cache, use_symptom=1,
            embedding_size=embedding_size, filter_key=filter_key, data_split_strategy=data_split_strategy))

    valid_key_set, id_mapping = set(), dict()
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        for item in dataset:
            key = item[0]
            unified_key = '-'.join(key.split('-')[1:])
            valid_key_set.add(unified_key)
            id_mapping[unified_key] = key
    logger.info(f'len valid_key_set: {len(valid_key_set)}')
    data_list = get_data(data_file_code_true_path, valid_key_set, visit_type_filter_key, data_source_filter_key,
                         max_data_size)

    if statr_index != -1 and end_index != -1:
        data_list = data_list[statr_index:end_index]
    logger.info(f'len data size: {len(data_list)}')

    call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize = (
        get_local_llm_model_tokenizer(model_id, max_output_token, llm_cache, max_model_len,
                                      tensor_split_num, enforce_eager, gpu_utilization, vllm_type)
    )
    logger.info('local llm load success')

    generate_fixed_question_answer(data_list, id_mapping, call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize,
                                   max_input_len, save_size, concurrent_size, visit_type_filter_key,
                                   data_source_filter_key)
    logger.info('success')


if __name__ == '__main__':
    main()
