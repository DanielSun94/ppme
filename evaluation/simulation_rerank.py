# 本脚本的目的是为llm和react的评估问题提供第二条解决方案
# 目前的直接诊断模式的评估受moderator影响。因此还是需要按照原来的排序方式重新做一遍数据
# 因此，拟开展的做法是抽取对话的所有上下文，然后在最后适配一次，这样既不用重新跑数据，也可以很方便的进行分类插拔适配（digit可以调整）
import os
import re
import json
import pickle
import openai
import numpy as np
from util import chn_index_diagnosis_name_mapping, eng_index_diagnosis_name_mapping
from evaluation_config import (evaluation_result_cache_folder, diagnosis_cache_template, chn_icd_mapping_path,
                               eng_icd_mapping_file, evaluation_rerank_cache_folder)
from evaluation_logger import logger


rerank_prompt_template = {
    'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据和病人的对话历史，推测他有什么疾病。你们的对话历史是：\n{}\n\n。"
           "你需要基于这一堆话，根据给出的疾病清单，列出造成病人本次入院的最可能的5种疾病，注意，越高风险的疾病应当排在越前面。"
           "你不能输出重复的疾病。\n你的输出只需要包含疾病编号即可，格式应当遵循：\n"
           "#数字#\n#数字#\n#数字#\n#数字#\n#数字#\n这样的格式。\n"
           "疾病清单是：\n{}",
    'eng': "Assume you are a doctor conducting a medical consultation with a patient. "
           "You need to infer the patient's potential diseases based on the conversation history. "
           "The conversation history is:\n{}\n\n. Based on this dialogue, using the provided disease list, "
           "list the 5 most likely diseases causing the patient's current admission, with higher-risk diseases "
           "ranked higher. You must not list duplicate diseases.\nYour output should only include the disease numbers, "
           "formatted as follows:\n#Number#\n#Number#\n#Number#\n#Number#\n#Number#\n.\nThe disease list is:\n{}"
}


def read_data(data_source, type_filter):
    file_list = os.listdir(evaluation_result_cache_folder)
    file_path_list = []
    for file in file_list:
        if data_source in file and type_filter in file:
            file_path = os.path.join(evaluation_result_cache_folder, file)
            file_path_list.append([file_path, file])

    data_list = []
    for file_path, file in file_path_list:
        data_list.append([file_path, file, json.load(open(file_path, 'r', encoding='utf-8-sig'))])
    return data_list


def reorganize_dialogue(data_list, language):
    dialogue_list = []
    for file_path, file, data in data_list:
        original_dialogue = data['dialogue']
        diagnosis = data['data']['oracle_diagnosis'].lower().strip()
        diagnosis = diagnosis.split("$$$$$")[0]
        if len(dialogue_list) == 0:
            logger.info('error')
        dialogue_str = ''
        for i in range(len(original_dialogue)-2):
            if i % 2 == 0:
                assert original_dialogue[i]['role'] == 'doctor'
                if language == 'eng':
                    dialogue_str += f'Turn: #{i//2+1}, Doctor Said:\n'
                else:
                    dialogue_str += f'第{i // 2 + 1}轮，医生说:\n'
            else:
                if language == 'eng':
                    dialogue_str += f'Turn: #{i // 2 + 1}, Patient Said:\n'
                else:
                    dialogue_str += f'第{i // 2 + 1}轮，病人说:\n'
            response = original_dialogue[i]['show_response']
            dialogue_str += response + '\n'
        dialogue_list.append([file_path, file, dialogue_str, diagnosis])
    return dialogue_list

def convert_dict_to_str(icd_mapping_dict):
    rank_str = ''
    for idx in icd_mapping_dict:
        rank_str += f'#{idx}#: {icd_mapping_dict[idx]}\n'
    return rank_str


endpoint_info = {
    "local_qwen_2__5_72b_int4":
        {
            "api_key": "token-abc123",
            "url": "http://localhost:8000/v1",
            "model_name": "Qwen/Qwen2.5-72B-Instruct-GPTQ-INT4"
        }
}

def non_streaming_call_llm(model_name, content):
    client = openai.OpenAI(
        api_key=endpoint_info[model_name]['api_key'],
        base_url=endpoint_info[model_name]['url'],
    )

    completion = client.chat.completions.create(
        model=endpoint_info[model_name]['model_name'],
        max_tokens=256,
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': content}],
        # temperature=0.0,
    )
    message = completion.choices[0].message.content
    return message


def parse_result_disease(response):
    response_list = response.split('\n')
    contents = []
    for s in response_list:
        match = re.match(r"#(\d+)#", s)
        if match:
            result = match.group(1)
            contents.append(int(result))
    assert len(contents) == 5
    return contents


def obtain_target_list(top_n, digit, weight, strategy, lower_threshold, filter_key_list):
    target_list = []
    for filter_key in filter_key_list:
        diagnosis_cache = diagnosis_cache_template.format(digit, top_n, weight, strategy, lower_threshold, filter_key)
        structurized_dict, diagnosis_index_map, index_diagnosis_map = pickle.load(open(diagnosis_cache, 'rb'))
        if 'srrsh' in filter_key:
            language = 'chn'
            icd_mapping_dict = chn_index_diagnosis_name_mapping(chn_icd_mapping_path, index_diagnosis_map)
        else:
            language = 'eng'
            icd_mapping_dict = eng_index_diagnosis_name_mapping(eng_icd_mapping_file, index_diagnosis_map)

        ranking_str = convert_dict_to_str(icd_mapping_dict)
        for llm_type in 'llm', 'react':
            data_list = read_data(filter_key, llm_type)
            dialogue_list = reorganize_dialogue(data_list, language)

            for file_path, file_name, dialogue, diagnosis in dialogue_list:
                target_file_name = f'{top_n}_{digit}_{weight}_{strategy}_{lower_threshold}_' + file_name
                key = f'{top_n}_{digit}_{weight}_{strategy}_{lower_threshold}_{filter_key}_{llm_type}'
                prompt = rerank_prompt_template[language].format(dialogue, ranking_str)
                save_path = os.path.join(evaluation_rerank_cache_folder, target_file_name)
                diagnosis = diagnosis.lower()[:digit]
                if diagnosis in diagnosis_index_map:
                    diagnosis_idx = diagnosis_index_map[diagnosis]
                else:
                    diagnosis_idx = -1
                    logger.info('diagnosis mapping failure error')
                target_list.append([file_path, file_name, prompt, save_path, key, diagnosis, diagnosis_idx])
    return target_list


def parse_result(target_list):
    logger.info('start parsing')
    for item in target_list:
        file_path, file_name, prompt, save_path, key, diagnosis, diagnosis_idx = item
        if os.path.exists(save_path):
            logger.info(f'{file_name} already exists')
            continue
        rank = 100
        success_flag, failed_time, result = False, 0, []
        while not success_flag:
            try:
                response = non_streaming_call_llm("local_qwen_2__5_72b_int4", prompt)
                result = parse_result_disease(response)
                success_flag = True
                logger.info(f'{file_name} success')
            except:
                failed_time += 1
                if failed_time > 3:
                    logger.info(f'{file_name} failed')
                    break
        for idx, diagnosis_idx_candidate in enumerate(result):
            if diagnosis_idx == diagnosis_idx_candidate:
                rank = idx + 1
                break
        json.dump([file_path, file_name, prompt, save_path, key, diagnosis, diagnosis_idx, rank],
                  open(save_path, 'w', encoding='utf-8-sig'))
        logger.info(f'{file_name}, final rank: {rank}')


def eval_result():
    file_list = os.listdir(evaluation_rerank_cache_folder)
    data_dict = {}
    for file_name in file_list:
        file_path = os.path.join(evaluation_rerank_cache_folder, file_name)
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        key, rank = data[4], data[7]
        if key not in data_dict:
            data_dict[key] = []
        data_dict[key].append(rank)

    logger.info('')
    for key in data_dict:
        logger.info(f'key: {key}')
        rank_list = np.array(data_dict[key])
        logger.info(f'top 1: {np.sum(rank_list<  2)/len(rank_list)}')
        logger.info(f'top 2: {np.sum(rank_list < 3) / len(rank_list)}')
        logger.info(f'top 3: {np.sum(rank_list < 4) / len(rank_list)}')
        logger.info(f'top 4: {np.sum(rank_list < 5) / len(rank_list)}')
        logger.info(f'top 5: {np.sum(rank_list < 6) / len(rank_list)}')
        logger.info('')
    print('')


def main():
    # filter_key_list = 'srrsh-hospitalization',
    # filter_key_list = 'srrsh-outpatient',
    # filter_key_list = 'mimic_iv',
    filter_key_list = 'mimic_iii',
    top_n, digit, weight, strategy, lower_threshold = 1, 4, False, 'DISCARD_OL', 20
    target_list = obtain_target_list(top_n, digit, weight, strategy, lower_threshold, filter_key_list)
    # parse_result(target_list)
    eval_result()





if __name__ == '__main__':
    main()
