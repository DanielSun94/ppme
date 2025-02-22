# 交叉性能分析
import os
from evaluation_logger import logger
import json
from util import (read_eval_data, candidate_srrsh_analysis, candidate_mimic_analysis,
                  read_chinese_icd, read_english_icd, eval_performance)
from evaluation_config import evaluation_result_cache_folder, evaluation_final_diagnosis_eval
from doctor_serve.util import non_streaming_call_llm


def call_llm(prompt, target_llm_name):
    return non_streaming_call_llm(target_llm_name, prompt)


eng_icd_mapping_dict = read_english_icd()
chn_icd_mapping_dict = read_chinese_icd()


def read_candidate(dialogue):
    utterance = dialogue[-2]['full_response']
    start_index = utterance.find('<AFFILIATED-INFO>') + len('<AFFILIATED-INFO>')
    end_index = utterance.find('</AFFILIATED-INFO>')
    data = json.loads(utterance[start_index:end_index])
    candidate_list = data['candidate_disease_list']
    return candidate_list


def main():
    language = 'chn'
    rank_llm_name = 'deepseek_v3_remote'
    # openbiollm llama_3_3_70b ultra_medical_llm deepseek_r1_70b huatuogpt_o1_72b local_qwen_2__5_72b_int4_2
    # deepseek_r1_remote local_qwen_2__5_72b
    source_llm_name = 'local_qwen_2__5_72b'
    data_source = 'srrsh'
    doctor_type = 'llm'
    turn_num = '10'
    start_index = 0
    end_index = 2500
    max_size = 2500
    logger.info(f'language: {language}')
    key_list = [data_source, doctor_type, language, rank_llm_name, source_llm_name, turn_num]

    key = data_source + '_' + doctor_type
    target_folder = os.path.join(evaluation_final_diagnosis_eval, rank_llm_name, source_llm_name, key)
    os.makedirs(target_folder, exist_ok=True)

    for item in key_list:
        logger.info(f'key: {item}')
    logger.info(f'start_index: {start_index}')
    logger.info(f'end_index: {end_index}')

    data_dict = read_eval_data(evaluation_result_cache_folder, source_llm_name, turn_num,
                               data_source, doctor_type)

    ordered_list = sorted(data_dict.keys())
    ordered_list = ordered_list[start_index:end_index]
    parse_index = 0
    eval_performance(target_folder, max_size)
    for key in ordered_list:
        target_file_path = os.path.join(target_folder, turn_num + "_" + key+'.json')
        if os.path.exists(target_file_path):
            logger.info(f'path: {target_file_path} parsed')
            continue

        parse_index += 1
        dialogue, data = data_dict[key][1]['dialogue'], data_dict[key][1]['data']
        client_id = data['source'] + '-' + data['visit_type'] + '-' + data['patient_visit_id']
        result = read_candidate(dialogue)
        if 'srrsh' in client_id:
            icd_dict = chn_icd_mapping_dict
            oracle_diagnosis_str = data['table_diagnosis'].strip().lower()
            first_diagnosis_str = oracle_diagnosis_str.split('$$$$$')[0]
            oracle_diagnosis_icd = data['oracle_diagnosis'].strip().lower()
            first_diagnosis_icd = oracle_diagnosis_icd.split('$$$$$')[0]
            rank_dict = candidate_srrsh_analysis(result, first_diagnosis_str, first_diagnosis_icd, icd_dict, call_llm,
                                                 rank_llm_name)
        else:
            assert 'mimic' in client_id
            icd_dict = eng_icd_mapping_dict
            oracle_diagnosis = data['oracle_diagnosis'].strip().lower()
            first_diagnosis = oracle_diagnosis.split('$$$$$')[0]
            if first_diagnosis[:3] not in icd_dict or first_diagnosis[:4] not in icd_dict:
                logger.info('ERROR')
                rank_dict = {3: 100, 4: 100}
            else:
                diagnosis_name_3 = icd_dict[first_diagnosis[:3]]
                diagnosis_name_4 = icd_dict[first_diagnosis[:4]]
                rank_dict = candidate_mimic_analysis(result, diagnosis_name_3, diagnosis_name_4, call_llm,
                                                     rank_llm_name)
        data_to_save = {
            'rank_dict': rank_dict,
        }
        logger.info(f'client_id: {client_id}, success')
        json.dump(data_to_save, open(target_file_path, 'w', encoding='utf-8-sig'))

        if parse_index % 20 == 0:
            eval_performance(target_folder, max_size)
    logger.info('data dict length: {}'.format(len(data_dict)))
    eval_performance(target_folder, max_size)


if __name__ == '__main__':
    main()
