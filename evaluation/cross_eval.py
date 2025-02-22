# 交叉性能分析
import os
from evaluation_logger import logger
import json
import traceback
from util import (read_eval_data, read_conversation, candidate_srrsh_analysis, candidate_mimic_analysis,
                  read_chinese_icd, read_english_icd, parse_result_disease_think, diagnosis_prompt_dict,
                  eval_performance)
from evaluation_config import evaluation_result_cache_folder, evaluation_cross_eval_folder
from doctor_serve.util import non_streaming_call_llm


def call_llm(prompt, target_llm_name):
    return non_streaming_call_llm(target_llm_name, prompt)


def re_diagnosis(dialogue_str, language, client_id, target_llm_name):
    prompt = diagnosis_prompt_dict[language].format(dialogue_str)
    success_flag = False
    failed_time = 0
    result_list = []
    while not success_flag:
        try:
            response = call_llm(prompt, target_llm_name)
            logger.info(f're diagnosis response: {response}')
            result_list = parse_result_disease_think(response)
            success_flag = True
        except:
            failed_time += 1
            logger.info(u"Error Trance {}".format(traceback.format_exc()))
            if failed_time > 5:
                logger.info(f'client: {client_id}, high risk diseases parse failed')
                break
    return result_list


eng_icd_mapping_dict = read_english_icd()
chn_icd_mapping_dict = read_chinese_icd()


def main():
    language = 'eng'
    target_llm_name = 'deepseek_r1_70b'
    rank_llm_name = 'gpt_4o_openai'
    source_llm_name = 'local_qwen_2__5_72b_int4_2'
    data_source = 'mimic'
    doctor_type = 'llm'
    turn_num = "10"
    start_index = 0
    end_index = 100000
    logger.info(f'language: {language}')
    key_list = [turn_num, data_source, doctor_type, language, target_llm_name, source_llm_name]

    key = data_source + '_' + doctor_type
    target_folder = os.path.join(evaluation_cross_eval_folder, source_llm_name, target_llm_name, key)
    os.makedirs(target_folder, exist_ok=True)

    for item in key_list:
        logger.info(f'key: {item}')
    logger.info(f'start_index: {start_index}')
    logger.info(f'end_index: {end_index}')

    data_dict = read_eval_data(evaluation_result_cache_folder, source_llm_name,
                               turn_num, data_source, doctor_type)
    ordered_list = sorted(data_dict.keys())
    ordered_list = ordered_list[start_index:end_index]
    parse_index = 0
    eval_performance(target_folder)
    for key in ordered_list:
        target_file_path = os.path.join(target_folder, key+'.json')
        if os.path.exists(target_file_path):
            logger.info(f'key: {key} parsed')
            continue

        parse_index += 1
        dialogue, data = data_dict[key][1]['dialogue'], data_dict[key][1]['data']
        client_id = data['source'] + '-' + data['visit_type'] + '-' + data['patient_visit_id']
        conversation_str = read_conversation(dialogue, language)
        result = re_diagnosis(conversation_str, language, client_id, target_llm_name)
        logger.info('re_diagnosis success')
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
            'cross_eval_diagnosis': result,
            'original_data': data_dict[key],
            'rank_llm_name': rank_llm_name,
            'source_llm': source_llm_name,
            'target_llm': target_llm_name,
        }
        logger.info(f'client_id: {client_id}, success')
        json.dump(data_to_save, open(target_file_path, 'w', encoding='utf-8-sig'))

        if parse_index % 20 == 0:
            eval_performance(target_folder)
    logger.info('data dict length: {}'.format(len(data_dict)))


if __name__ == '__main__':
    main()
