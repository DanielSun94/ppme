from disease_screen_logger import logger
import csv
import os
import random
from datetime import datetime
import threading
import time
import numpy as np
from emr_structurize_util import (call_remote_llm, call_remote_tokenizer_tokenize,
                                  call_remote_tokenizer_reverse_tokenize, construct_question_list)
import argparse
from itertools import islice
from util import read_context, get_local_llm_model_tokenizer
from disease_screen_config import (symptom_file_path_template, llm_cache_folder, final_data_folder,
                                   disease_screen_folder)

os.environ['NCCL_P2P_DISABLE'] = "1"


def read_id_order_seq(id_list, start_index, end_index, random_seed):
    logger.info('full id_list length: {}'.format(len(id_list)))
    random.Random(random_seed).shuffle(id_list)
    if start_index >= 0 and end_index == -1:
        id_list = id_list[start_index:]
    elif 0 <= start_index < end_index and end_index >= 0:
        id_list = id_list[start_index:end_index]
    elif start_index == -1 and end_index > 0:
        id_list = id_list[:end_index]
    else:
        raise ValueError('illegal start index or end index')
    return id_list


def get_next_sample_info(data_dict, transformed_id_dict, id_seq_list, batch_size, start_index):
    id_list = []
    for idx, key in enumerate(id_seq_list):
        if key in data_dict and key not in transformed_id_dict:
            id_list.append([key, start_index + idx])
        if len(id_list) >= batch_size:
            break
    return id_list


def result_preprocess(result_str, target_length):
    result_str = result_str.strip()
    start_index = result_str.find('#1#')
    if start_index == -1:
        # 之所以有这个设置，是因为01 AI的大模型特别喜欢空一格，原因不明。也只有这个模型有这个现象
        start_index = result_str.find('#1 #')
    if start_index == -1:
        logger.info('format illegal')
        raise ValueError('Invalid result')

    result_list = result_str[start_index:].split('\n')
    if len(result_list) < target_length:
        # logger.info('Illegal result: {}'.format(result_str))
        logger.info('count illegal, too short')
        raise ValueError('Invalid result')

    if len(result_list) >= target_length:
        result_list = result_list[:target_length]

    for i, result in enumerate(result_list):
        flag_1 = '#{}#'.format(i + 1) in result or '#{} #'.format(i + 1) in result
        flag_2 = ('NO' in result and 'NA' not in result and 'YES' not in result) or \
                 ('NO' not in result and 'NA' in result and 'YES' not in result) or \
                 ('NO' not in result and 'NA' not in result and 'YES' in result)
        if not (flag_1 and flag_2):
            if not flag_1:
                logger.info('count illegal')
            elif not flag_2:
                logger.info('format illegal')
            raise ValueError('Invalid result')
    return result_list


def parse_level_1_symptom(content_list, init_prompt, general_symptom_dict, symptom_info, question_one_list,
                          call_llm, dialogue_list, batch_size):
    # 先生成所有需要解析的数据，然后一个个过
    prompt_info_list = []
    for item in content_list:
        unified_id, content = item
        logger.info('start parsing: {}'.format(unified_id))
        for sub_level_1_prompt, sub_level_1_symptom_list in question_one_list:
            prompt = init_prompt + content + '\n' + sub_level_1_prompt
            # 第4个指的是这个prompt有没有解析成功，第五个是失败的次数
            prompt_info_list.append([unified_id, prompt, sub_level_1_prompt, sub_level_1_symptom_list, False, 0])

    logger.info('level 1 symptom batch prompt length: {}'.format(len(prompt_info_list)))
    all_parsed = False
    while not all_parsed:
        batch_prompt, batch_unified_id, batch_symptom_list, batch_idx = [], [], [], []
        for idx, item in enumerate(prompt_info_list):
            if not item[4] and item[5] < 4:
                batch_prompt.append(item[1])
                batch_unified_id.append(item[0])
                batch_symptom_list.append(item[3])
                batch_idx.append(idx)
            if len(batch_prompt) == batch_size:
                break
        assert len(batch_symptom_list) > 0

        result_str_list = call_llm(batch_prompt)
        for result_str, unified_id, symptom_list, idx, prompt in (
                zip(result_str_list, batch_unified_id, batch_symptom_list, batch_idx, batch_prompt)):
            try:
                result_list = result_preprocess(result_str, len(symptom_list))
                for result, symptom in zip(result_list, symptom_list):
                    if 'NO' in result:
                        general_symptom_dict[unified_id][symptom] = 'NO'
                        if symptom_info[unified_id][symptom] is not None:
                            for factor_group in symptom_info[unified_id][symptom]:
                                for factor in symptom_info[unified_id][symptom][factor_group]:
                                    symptom_info[unified_id][symptom][factor_group][factor] = 'NO'
                    elif 'NA' in result:
                        general_symptom_dict[unified_id][symptom] = 'NA'
                        if symptom_info[unified_id][symptom] is not None:
                            for factor_group in symptom_info[unified_id][symptom]:
                                for factor in symptom_info[unified_id][symptom][factor_group]:
                                    symptom_info[unified_id][symptom][factor_group][factor] = 'NA'
                    else:
                        if 'YES' not in result:
                            logger.info('Answer illegal')
                            raise ValueError('')
                        general_symptom_dict[unified_id][symptom] = 'YES'
                prompt_info_list[idx][4] = True
                dialogue_list.append([unified_id, prompt, result_str])
            except ValueError as _:
                prompt_info_list[idx][5] += 1
                logger.info(f'unified_id error: {prompt_info_list[idx][0]}')
                if prompt_info_list[idx][5] > 3:
                    logger.info(f'prompt: {prompt_info_list[idx][1]}, response: {result_str}')
                    logger.info('exceed maximum retry time (3)')
                continue

        parse_count = 0
        for item in prompt_info_list:
            if item[4] or item[5] >= 4:
                parse_count += 1
        if parse_count == len(prompt_info_list):
            all_parsed = True

    return symptom_info, general_symptom_dict


def parse_level_2_symptom(secondary_prompt_info_list, symptom_info, dialogue_list, call_llm, concurrent_size):
    all_parsed = False
    while not all_parsed:
        batch_prompt, batch_unified_id, batch_symptom_list, batch_idx, batch_first_symptom = [], [], [], [], []
        for idx, item in enumerate(secondary_prompt_info_list):
            if not item[5] and item[6] < 4:
                batch_prompt.append(item[1])
                batch_unified_id.append(item[0])
                batch_symptom_list.append(item[3])
                batch_idx.append(idx)
                batch_first_symptom.append(item[4])
            if len(batch_prompt) == concurrent_size:
                break
        assert len(batch_symptom_list) > 0

        result_str_list = call_llm(batch_prompt)
        for result_str, unified_id, symptom_list, idx, prompt, first_symptom in (
                zip(result_str_list, batch_unified_id, batch_symptom_list, batch_idx, batch_prompt,
                    batch_first_symptom)):
            try:
                result_list = result_preprocess(result_str, len(symptom_list))
                for result, (factor_group, factor) in zip(result_list, symptom_list):
                    if 'NO' in result:
                        symptom_info[unified_id][first_symptom][factor_group][factor] = 'NO'
                    elif 'NA' in result:
                        symptom_info[unified_id][first_symptom][factor_group][factor] = 'NA'
                    else:
                        if 'YES' not in result:
                            logger.info('Answer illegal')
                            raise ValueError('')
                        symptom_info[unified_id][first_symptom][factor_group][factor] = 'YES'
                secondary_prompt_info_list[idx][5] = True
                dialogue_list.append([unified_id, prompt, result_str, first_symptom])
            except ValueError as _:
                secondary_prompt_info_list[idx][6] += 1
                logger.info(f'unified_id error: {secondary_prompt_info_list[idx][0]}')
                if secondary_prompt_info_list[idx][6] > 3:
                    logger.info(f'prompt: {secondary_prompt_info_list[idx][1]}, response: {result_str}')
                    logger.info('exceed maximum retry time (3), exit')

        parse_count = 0
        for item in secondary_prompt_info_list:
            if item[5] or item[6] > 3:
                parse_count += 1
        if parse_count == len(secondary_prompt_info_list):
            all_parsed = True


def initialize_symptom_info(symptom_dict, unified_id_list):
    symptom_info, general_symptom_dict = dict(), dict()
    for unified_id in unified_id_list:
        symptom_info[unified_id] = dict()
        general_symptom_dict[unified_id] = dict()
        for key in symptom_dict:
            symptom_info[unified_id][key] = dict()
            general_symptom_dict[unified_id][key] = "NA"
            for factor_group in symptom_dict[key]:
                symptom_info[unified_id][key][factor_group] = dict()
                for factor in symptom_dict[key][factor_group]:
                    symptom_info[unified_id][key][factor_group][factor] = "NA"
    return symptom_info, general_symptom_dict


def parse_symptom(data_dict, unified_id_list, symptom_dict, question_one_list, question_two_list_dict, call_llm,
                  tokenizer_tokenize, tokenizer_reverse_tokenize, language, max_context_len, concurrent_size):
    dialogue_list = []
    if language == 'eng':
        init_prompt = "Please assume you are a senior doctor, given the below admission record:\n\n"
    else:
        assert language == 'chn'
        init_prompt = "请假设您是一位高级医生，基于以下入院记录：\n\n"

    symptom_info, general_symptom_dict = initialize_symptom_info(symptom_dict, unified_id_list)
    content_list = []
    for unified_id in unified_id_list:
        content_list.append([unified_id, data_dict[unified_id]])

    # 当上下文过长是截断
    for i in range(len(content_list)):
        unified_id, content = content_list[i]
        token_list = tokenizer_tokenize(content)
        if len(token_list) > max_context_len:
            logger.info(f'{unified_id} context length too long (length {len(token_list)}), '
                        f'truncate to {max_context_len} tokens')
            token_list = token_list[:max_context_len]
        content = tokenizer_reverse_tokenize(token_list)
        content_list[i] = [unified_id, content]

    parse_level_1_symptom(content_list, init_prompt, general_symptom_dict, symptom_info, question_one_list,
                          call_llm, dialogue_list, concurrent_size)

    secondary_prompt_info_list = []
    for unified_id in symptom_info:
        content = None
        for item in content_list:
            if item[0] == unified_id:
                content = item[1]
        assert content is not None
        for first_symptom in symptom_info[unified_id]:
            second_level_prompt_list = question_two_list_dict[first_symptom]
            if second_level_prompt_list is None:
                continue
            if general_symptom_dict[unified_id][first_symptom] != 'YES':
                continue
            logger.info(f'start parsing: {unified_id}, level 1 symptom {first_symptom}')
            for (second_level_prompt, symptom_list) in second_level_prompt_list:
                prompt = init_prompt + content + '\n' + second_level_prompt
                secondary_prompt_info_list.append([unified_id, prompt, second_level_prompt, symptom_list, first_symptom,
                                                   False, 0])
    logger.info('level 1 symptom batch prompt length: {}'.format(len(secondary_prompt_info_list)))
    parse_level_2_symptom(secondary_prompt_info_list, symptom_info, dialogue_list, call_llm, concurrent_size)
    return dialogue_list, symptom_info, general_symptom_dict


def single_admission_record_parse(data_dict, unified_id_list, symptom_dict, level_1_list, level_2_list_dict, call_llm,
                                  tokenizer_tokenize, tokenizer_reverse_tokenize, language, unified_id_idx_list,
                                  symptom_folder, folder_size, transformed_id_dict, max_context_token, batch_size):
    start_time = datetime.now()
    dialogue_list, symptom_info, general_symptom_dict = (
        parse_symptom(data_dict, unified_id_list, symptom_dict, level_1_list, level_2_list_dict, call_llm,
                      tokenizer_tokenize, tokenizer_reverse_tokenize, language, max_context_token, batch_size)
    )

    for unified_id_idx, unified_id in zip(unified_id_idx_list, unified_id_list):
        formatted_num = "{:05d}".format(unified_id_idx // folder_size)
        folder_path = os.path.join(symptom_folder, formatted_num)
        os.makedirs(folder_path, exist_ok=True)

        symptom_path = os.path.join(folder_path, unified_id + '_symptom.csv')
        with open(symptom_path, 'w', encoding='utf-8-sig', newline='') as f:
            data_to_write = [['symptom', 'factor_group', 'factor', 'state']]
            for symptom in general_symptom_dict[unified_id]:
                data_to_write.append([symptom, 'N/A', 'N/A', general_symptom_dict[unified_id][symptom]])
            for symptom in symptom_info[unified_id]:
                for factor_group in symptom_info[unified_id][symptom]:
                    for factor in symptom_info[unified_id][symptom][factor_group]:
                        state = symptom_info[unified_id][symptom][factor_group][factor]
                        data_to_write.append([symptom, factor_group, factor, state])
            csv.writer(f).writerows(data_to_write)

        detail_path = os.path.join(folder_path, unified_id + '_detail.csv')
        with open(detail_path, 'w', encoding='utf-8-sig', newline='') as f:
            data_to_write = [['question', 'answer']]
            for item in dialogue_list:
                if item[0] == unified_id:
                    data_to_write.append(item)
            csv.writer(f).writerows(data_to_write)

        transformed_id_dict[unified_id] = formatted_num

    end_time = datetime.now()
    time_diff = end_time - start_time
    logger.info(f"Time difference: {time_diff}")


def parse_argument():
    parser = argparse.ArgumentParser()
    # model_id = 'openai/gpt_4o_mini'
    # model_id = 'meta/llama3-8b'
    # model_id = 'ZhipuAI/glm-4-9b-chat'
    # model_id = 'qwen/Qwen2-72B-Instruct'
    # model_id = 'qwen/Qwen2-72B-Instruct-GPTQ-Int4'
    # model_id = 'qwen/Qwen2-72B-Instruct-GPTQ-Int8'
    # model_id = 'qwen/Qwen2-7B-Instruct'
    # model_id = '01ai/Yi-1___5-34B-Chat'
    # model_id = '01ai/Yi-1___5-9B-Chat'
    # model_id = 'tclf90/Yi-1___5-34B-Chat-16K-GPTQ-Int4'
    # model_id = 'tclf90/Qwen2-72B-Instruct-GPTQ-Int3'
    model_id = 'qwen/Qwen2___5-72B-Instruct-GPTQ-Int4'
    # model_id = 'qwen/Qwen2___5-32B-Instruct-GPTQ-Int4'
    # model_id = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
    # model_id = 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4'
    parser.add_argument('--model_id', help='', default=model_id, type=str)
    parser.add_argument('--tensor_split_num', help='', default=2, type=int)
    parser.add_argument('--enforce_eager', help='', default=0, type=int)
    parser.add_argument('--visible_gpu', help='', default="0,1", type=str)
    parser.add_argument('--gpu_utilization', help='', default=0.98, type=float)
    # 注意，此处的max_data_size指的是数据行数，此处的数据可重复。而batch size指独立数据个数，因此在max_data_size接近batch size时
    # 可能会出现独立数据小于batch size规定而报错
    parser.add_argument('--start_index', help='', default=0, type=int)
    parser.add_argument('--end_index', help='', default=-1, type=int)
    parser.add_argument('--batch_size', help='', default=512, type=int)
    parser.add_argument('--folder_size', help='', default=2000, type=int)
    # 这里3072是给context。剩下的1024给了prompt和生成，正常文本长度中位数为700-800左右，给了四倍，会对特别冗长地进行一定地截断，
    # 但是能容纳绝大多数文本。
    parser.add_argument('--max_model_len', help='', default=3594, type=int)
    parser.add_argument('--max_output_token', help='', default=1536, type=int)
    parser.add_argument('--max_context_token', help='', default=2048, type=int)
    parser.add_argument('--max_data_size', help='', default=-1, type=int)
    parser.add_argument('--random_seed', help='', default=715, type=int)
    parser.add_argument('--language', help='', default='chn', type=str)
    parser.add_argument('--llm_type', help='', default='local', type=str)
    parser.add_argument('--vllm_type', help='', default=None, type=str)
    parser.add_argument('--llm_concurrent_batch_size', help='', default=128, type=int)
    parser.add_argument('--visit_type_filter_key', help='', default="outpatient", type=str)
    parser.add_argument('--data_source_filter_key', help='', default="srrsh", type=str)
    parser.add_argument('--save_folder', help='', default='', type=str)
    parser.add_argument('--llm_cache_folder', help='', default=llm_cache_folder, type=str)
    args = vars(parser.parse_args())
    for key in args:
        logger.info('{}: {}'.format(key, args[key]))

    # 校验数据是否正确
    # 如果大模型是本地的，则要确保batch size为1，模型在符合要求的模型清单中
    if args['llm_type'] == 'local':
        if args['language'] == 'chn':
            assert args['model_id'] in {'ZhipuAI/glm-4-9b-chat', 'qwen/Qwen2-72B-Instruct', 'qwen/Qwen2-7B-Instruct',
                                        '01ai/Yi-1___5-34B-Chat', '01ai/Yi-1___5-9B-Chat',
                                        'qwen/Qwen2-72B-Instruct-GPTQ-Int8', 'qwen/Qwen2-72B-Instruct-GPTQ-Int4',
                                        'tclf90/Yi-1___5-34B-Chat-16K-GPTQ-Int4', 'tclf90/Qwen2-72B-Instruct-GPTQ-Int3',
                                        'qwen/Qwen2___5-32B-Instruct-GPTQ-Int4',
                                        'qwen/Qwen2___5-72B-Instruct-GPTQ-Int4',
                                        'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4',
                                        'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4'}
        else:
            assert args['language'] == 'eng'
            assert args['model_id'] in {'LLM-Research/Meta-Llama-3___1-70B-Instruct-GPTQ-INT4',
                                        'qwen/Qwen2___5-32B-Instruct-GPTQ-Int4',
                                        'qwen/Qwen2___5-72B-Instruct-GPTQ-Int4',
                                        'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4',
                                        'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4'}
    else:
        assert args['llm_type'] == 'remote'
        assert args['language'] == 'eng'
        assert 'gpt' in args['model_id'] or 'llama' in args['model_id']
        assert args['llm_concurrent_batch_size'] == 1
    return args


def tokenize_stat(data, tokenizer):
    token_list = []
    for key in data:
        tokenize_list = tokenizer(data[key])
        token_list.append(len(tokenize_list))
        if len(tokenize_list) > 0 and (len(token_list) == len(data) or len(token_list) % 1000 == 0):
            print(f'length: {len(token_list)}')
            print(f'average length: {np.average(token_list)}')
            print(f'max length: {np.max(token_list)}')
            print(f'median length: {np.median(token_list)}')


def main():
    args = parse_argument()
    model_id = args['model_id']
    max_output_token = args['max_output_token']
    folder_size = args['folder_size']
    batch_size = args['batch_size']
    max_model_len = args['max_model_len']
    start_index = args['start_index']
    end_index = args['end_index']
    random_seed = args['random_seed']
    max_context_token = args['max_context_token']
    visit_type_filter_key = args['visit_type_filter_key']
    tensor_split_num = args['tensor_split_num']
    data_source_filter_key = args['data_source_filter_key']
    llm_cache = args['llm_cache_folder']
    gpu_utilization = args['gpu_utilization']
    visible_gpu = args['visible_gpu']
    llm_type = args['llm_type']
    save_folder = args['save_folder']
    language = args['language']
    enforce_eager = args['enforce_eager']
    concurrent_size = args['llm_concurrent_batch_size']
    vllm_type = args['vllm_type']
    max_data_size = args['max_data_size']
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpu
    file_name = 'use_code_map_True_digit_4_fraction_0.95.csv'
    file_path = os.path.join(final_data_folder, file_name)
    maximum_questions = 30

    assert enforce_eager == 0 or enforce_eager == 1
    enforce_eager = True if enforce_eager == 1 else False

    if save_folder == '':
        save_folder = disease_screen_folder

    symptom_dict = read_symptom(symptom_file_path_template.format(language))
    level_1_list, level_2_list_dict = (
        construct_question_list(symptom_dict, language, maximum_questions=maximum_questions))
    structured_symptom_folder_template = os.path.join(save_folder, 'structured_symptom/{}/{}/{}')
    symptom_folder = structured_symptom_folder_template.format(model_id.split('/')[1], data_source_filter_key,
                                                               visit_type_filter_key)
    logger.info('symptom save folder: {}'.format(symptom_folder))
    os.makedirs(symptom_folder, exist_ok=True)
    origin_data_dict = read_context(file_path, visit_type_filter_key, data_source_filter_key,
                                    max_data_size=max_data_size)

    id_seq_list = sorted(list(origin_data_dict.keys()))
    id_seq_list = read_id_order_seq(id_seq_list, start_index, end_index, random_seed)
    data_dict = slim_data(origin_data_dict, id_seq_list)
    logger.info(f'data read success, data parsing size: {len(data_dict)}')

    unified_parsed_id_list = []
    hit_parsed_id_list = []
    if os.path.exists(symptom_folder):
        transformed_id_dict = dict()
        folder_list = os.listdir(symptom_folder)
        for folder in folder_list:
            file_list = os.listdir(os.path.join(symptom_folder, folder))
            for file in file_list:
                file_name = file.replace('_symptom.csv', '').replace('_detail.csv', '')
                transformed_id_dict[file_name] = folder
        for key in transformed_id_dict:
            # assert key in origin_data_dict
            unified_parsed_id_list.append(key)
            if key in data_dict:
                hit_parsed_id_list.append(key)
            # logger.info('unified_id: {} parse success'.format(key))
    else:
        transformed_id_dict = dict()
    logger.info(f'parsed id list len: {len(unified_parsed_id_list)}')
    logger.info(f'remaining size: {len(data_dict) - len(hit_parsed_id_list)}')
    logger.info('start parsing data')

    if llm_type == 'local':
        call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize = (
            get_local_llm_model_tokenizer(model_id, max_output_token, llm_cache, max_model_len,
                                          tensor_split_num, enforce_eager, gpu_utilization, vllm_type)
        )
        logger.info('local llm load success')
    else:
        assert llm_type == 'remote'
        call_llm = lambda x: call_remote_llm(model_id, x)
        tokenizer_tokenize = lambda x: call_remote_tokenizer_tokenize(model_id, x)
        tokenizer_reverse_tokenize = lambda x: call_remote_tokenizer_reverse_tokenize(model_id, x)

    while True:
        sample_info = get_next_sample_info(data_dict, transformed_id_dict, id_seq_list, batch_size, start_index)
        if len(sample_info) == 0:
            break

        if llm_type == 'local':
            unified_id_list = [str(item[0]) for item in sample_info]
            unified_id_idx_list = [item[1] for item in sample_info]
            single_admission_record_parse(data_dict, unified_id_list, symptom_dict, level_1_list, level_2_list_dict,
                                          call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize, language,
                                          unified_id_idx_list, symptom_folder,
                                          folder_size, transformed_id_dict, max_context_token, concurrent_size)
        else:
            assert llm_type == 'remote'
            threads = []
            for i, (unified_id, unified_id_idx) in enumerate(sample_info):
                unified_id_list, unified_id_idx_list = [unified_id], [unified_id_idx]
                time.sleep(0.02)
                thread = threading.Thread(
                    target=single_admission_record_parse,
                    args=(data_dict, unified_id_list, symptom_dict, level_1_list, level_2_list_dict,
                          call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize, language, unified_id_idx_list,
                          symptom_folder, folder_size, transformed_id_dict, max_context_token, concurrent_size)
                )
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()


def read_symptom(symptom_path):
    symptom_dict = dict()
    with open(symptom_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            symptom, factor_group, factor = line
            if symptom not in symptom_dict:
                symptom_dict[symptom] = dict()
            if len(factor_group) > 0:
                assert len(factor_group) > 0
                if factor_group not in symptom_dict[symptom]:
                    symptom_dict[symptom][factor_group] = []
                symptom_dict[symptom][factor_group].append(factor)
    return symptom_dict


def slim_data(data_dict, id_seq_list):
    # 丢弃本进程中因为index设置不会用到的数据
    # 丢弃Admission Record中无需保留的内容
    new_data_dict = dict()
    for unified_id in id_seq_list:
        new_data_dict[unified_id] = data_dict[unified_id]
    return new_data_dict


if __name__ == '__main__':
    main()
