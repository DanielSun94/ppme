import csv
import os.path
import pickle
import json
from datetime import datetime
from transformers import AutoTokenizer
from srrsh_logger import logger
from vllm import LLM, SamplingParams
from srrsh_config import fused_outpatient_admission_discharge_history_note, discharge_data_path, icu_patient_idx_file


def parse_response(response):
    response = response.upper()
    if 'YES' in response and 'NO' in response:
        raise ValueError('')
    elif 'YES' in response:
        return 1
    elif 'NO' in response:
        return 0
    else:
        raise ValueError('')


def get_model(max_token, model_split, model_id, enforce_eager, gpu_utilization, cache_folder, vllm_type):
    if len(cache_folder) > 0:
        model_path = str(os.path.join(cache_folder, model_id))
    else:
        model_path = model_id
    max_model_len, tp_size = max_token, model_split

    logger.info('local model, model id: {}'.format(model_path))
    llm_model = LLM(
        model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_utilization,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        quantization=vllm_type
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side='left',
        trust_remote_code=True,
    )
    return llm_model, tokenizer


def get_local_llm_model_tokenizer(model_id, max_output_token, llm_cache, max_token, tensor_split,
                                  enforce_eager, gpu_utilization, vllm_type):
    if model_id == 'qwen/Qwen2___5-32B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'qwen/Qwen2___5-72B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    else:
        raise ValueError('')

    logger.info('model id: {}'.format(model_id))
    llm_model, llm_tokenizer = get_model(cache_folder=llm_cache, max_token=max_token, enforce_eager=enforce_eager,
                                         model_split=tensor_split, model_id=model_id, gpu_utilization=gpu_utilization,
                                         vllm_type=vllm_type)
    def call_llm(prompt):
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            prompt = llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = llm_model.generate(prompt, sampling_strategy)
            result = outputs[0].outputs[0].text
        else:
            prompt_list = []
            for prompt_ in prompt:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_}
                ]
                prompt_ = llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_list.append(prompt_)
            outputs = llm_model.generate(prompt_list, sampling_strategy)
            result = []
            for output in outputs:
                text = output.outputs[0].text
                result.append(text)
        return result

    def tokenizer_tokenize(input_info):
        result = llm_tokenizer(input_info)['input_ids']
        return result

    def tokenizer_reverse_tokenize(input_info):
        result = llm_tokenizer.decode(input_info)
        return result
    return call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize


def select_discharge_data(data_path, save_path, read_from_cache=True):
    if os.path.exists(save_path) and read_from_cache:
        new_data_list = pickle.load(open(save_path, 'rb'))
        return new_data_list
    data_dict = dict()
    with open(data_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if line[0] == 'outpatient':
                continue
            key = 'srrsh-hospitalization-' + line[1]
            discharge_note = line[53]
            data_dict[key] = discharge_note

    data_list = [[key, data_dict[key]]for key in data_dict.keys()]

    data_list = sorted(data_list, key=lambda x:x[0])
    pickle.dump(data_list, open(save_path, 'wb'))
    return data_list


prompt_template = (
    '请你判断给定的电子病历数据中的病人是否是重症病人。以下三个条件满足任意一个即可认定是重症病人。若均不满足则不是重症病人。\n'
    '1.电子病历中明确说明其在本次住院期间入住了重症监护室\n'
    '2.在本次住院期间接受了手术（外科手术或内科介入手术均算，但各类影像检查（包括ECG,EEG）和实验室检查不算）\n'
    '3.在本次住院期间发生抢救事件。'
    '你的回答中需要回复Yes（是重症）或No(不是重症)。无需回复其它内容'
    '\n'
    '电子病历数据如下所示:\n\n')

def find_severe_patient(data_list, call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize,
                        max_input_len, save_size, concurrent_size, mapping_idx_set):
    if os.path.exists(icu_patient_idx_file):
        data_dict = json.load(open(icu_patient_idx_file, 'r', encoding='utf-8-sig'))
    else:
        data_dict = {}
    logger.info(f'Parsed file count: {len(data_dict)}')

    valid_data_list, hit_count = [], 0
    for item in data_list:
        if item[0] in mapping_idx_set:
            new_key = mapping_idx_set[item[0]]
            if new_key not in data_dict:
                valid_data_list.append(item)
            else:
                hit_count += 1
    logger.info('remaining data len: {}, hit_count: {}'.format(len(valid_data_list), hit_count))

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
            key, context = item
            prompt = prompt_template + context

            # 当上下文过长是截断
            token_list = tokenizer_tokenize(prompt)
            if len(token_list) >= max_input_len:
                logger.info(f'context length too long (length {len(token_list)}), truncate to {max_input_len} tokens')
                token_list = token_list[:max_input_len]
            prompt_truncated = tokenizer_reverse_tokenize(token_list)
            prompt_list.append(prompt_truncated)
            key_list.append(key)

        cursor = end_idx

        success_flag = False
        failure_time = 0
        result_list = []
        response_list = None
        while not success_flag:
            try:
                response_list = call_llm(prompt_list)
                for response, key in zip(response_list, key_list):
                    result = parse_response(response)
                    result_list.append([key, result])
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
        if len(cache_list) > save_size:
            for key, data in cache_list:
                if key in mapping_idx_set:
                    new_key = mapping_idx_set[key]
                    data_dict[new_key] = data
                else:
                    logger.info(f'key: {key} info is not in structurized dataset')
            logger.info(f'current data dict size: {len(data_dict)}')
            json.dump(data_dict, open(icu_patient_idx_file, 'w', encoding='utf-8-sig'))
            cache_list = []
    json.dump(data_dict, open(icu_patient_idx_file, 'w', encoding='utf-8-sig'))


def read_idx_mapping(folders):
    data_dict = dict()
    for folder in folders:
        sub_folder_list = os.listdir(folder)
        for sub_folder in sub_folder_list:
            sub_folder_path = os.path.join(folder, sub_folder)
            file_list = os.listdir(sub_folder_path)
            for file_name in file_list:
                if 'detail.csv' not in file_name:
                    continue
                file_key = file_name[:-11]
                data_dict[file_key] = str(sub_folder) + '-' + file_key
    return data_dict


def main():
    model_id = 'qwen/Qwen2___5-32B-Instruct-GPTQ-Int4'
    tensor_split_num = 1
    enforce_eager = False
    gpu_utilization = 0.9
    max_output_token = 512
    max_model_len = 4096
    max_input_len = max_model_len - max_output_token - 10
    llm_cache = '/mnt/disk_1/llm_cache'
    vllm_type = 'gptq'
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

    save_size = 4096
    concurrent_size = 32
    root = '/home/sunzhoujian/remote_development/ecdai/resource/disease_screen/structured_symptom/'
    folders = [
        os.path.join(root, 'Qwen2-72B-Instruct-GPTQ-Int4/srrsh/hospitalization'),
        os.path.join(root, 'Qwen2___5-72B-Instruct-GPTQ-Int4/mimic_iii/hospitalization'),
        os.path.join(root, 'Qwen2___5-72B-Instruct-GPTQ-Int4/mimic_iv/hospitalization'),
        os.path.join(root, 'Qwen2___5-72B-Instruct-GPTQ-Int4/srrsh/outpatient')
    ]
    mapping_idx_set = read_idx_mapping(folders)
    logger.info(f'len mapping idx set: {len(mapping_idx_set)}')

    data_list = select_discharge_data(fused_outpatient_admission_discharge_history_note, discharge_data_path)
    logger.info('data list loaded')

    call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize = (
        get_local_llm_model_tokenizer(model_id, max_output_token, llm_cache, max_model_len,
                                      tensor_split_num, enforce_eager, gpu_utilization, vllm_type)
    )
    find_severe_patient(data_list, call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize,
                        max_input_len, save_size, concurrent_size, mapping_idx_set)
    logger.info('')


if __name__ == '__main__':
    main()
