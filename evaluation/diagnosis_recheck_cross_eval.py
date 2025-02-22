# 交叉性能分析
import os
from evaluation_logger import logger
import json
from util import (candidate_srrsh_analysis, candidate_mimic_analysis, read_chinese_icd, read_english_icd,
                  eval_performance)
from evaluation_config import (evaluation_final_diagnosis_eval, evaluation_cross_eval_folder)
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


def read_data(folder, source_llm, cross_llm, key, turn_num):
    target_folder = os.path.join(folder, source_llm, cross_llm, key)
    data_dict = dict()
    file_list = os.listdir(target_folder)
    for file in file_list:
        if file[:2] != turn_num:
            continue
        file_path = os.path.join(target_folder, file)
        json_file = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        data_source = json_file['original_data'][1]['data']['source']
        visit_type = json_file['original_data'][1]['data']['visit_type']
        patient_visit_id = json_file['original_data'][1]['data']['patient_visit_id']
        unified_id = data_source + '-' + visit_type + '-' + patient_visit_id
        data_dict[unified_id] = json_file
    return data_dict


# 用于huatuogpt o1的输出的recheck
def main():
    language = 'chn'
    rank_llm_name = 'deepseek_v3_remote'
    source_llm_name = 'local_qwen_2__5_72b_int4_2'
    cross_llm_name = 'huatuogpt_o1_72b'
    data_source = 'srrsh'
    doctor_type = 'llm'
    start_index = 0
    turn_num = "10"
    end_index = 100000
    logger.info(f'language: {language}')
    key_list = [data_source, doctor_type, language, rank_llm_name, source_llm_name, cross_llm_name]

    key = data_source + '_' + doctor_type
    target_folder = os.path.join(evaluation_final_diagnosis_eval, rank_llm_name,
                                 'cross_eval_' + cross_llm_name + '_source_' + source_llm_name, key)
    os.makedirs(target_folder, exist_ok=True)

    for item in key_list:
        logger.info(f'key: {item}')
    logger.info(f'start_index: {start_index}')
    logger.info(f'end_index: {end_index}')

    # for normal eval
    key = data_source + '_' + doctor_type
    data_dict = read_data(evaluation_cross_eval_folder, source_llm_name, cross_llm_name, key, str(turn_num))
    # for cross eval

    ordered_list = sorted(data_dict.keys())
    ordered_list = ordered_list[start_index:end_index]
    parse_index = 0
    eval_performance(target_folder)
    for key in ordered_list:
        target_file_path = os.path.join(target_folder, turn_num + "_" + key+'.json')
        if os.path.exists(target_file_path):
            logger.info(f'key: {key} parsed')
            continue

        parse_index += 1
        candidate, data = data_dict[key]['cross_eval_diagnosis'], data_dict[key]['original_data'][1]['data']
        client_id = data['source'] + '-' + data['visit_type'] + '-' + data['patient_visit_id']
        if 'srrsh' in client_id:
            icd_dict = chn_icd_mapping_dict
            oracle_diagnosis_str = data['table_diagnosis'].strip().lower()
            first_diagnosis_str = oracle_diagnosis_str.split('$$$$$')[0]
            oracle_diagnosis_icd = data['oracle_diagnosis'].strip().lower()
            first_diagnosis_icd = oracle_diagnosis_icd.split('$$$$$')[0]
            rank_dict = candidate_srrsh_analysis(candidate, first_diagnosis_str, first_diagnosis_icd, icd_dict,
                                                 call_llm, rank_llm_name)
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
                rank_dict = candidate_mimic_analysis(candidate, diagnosis_name_3, diagnosis_name_4, call_llm,
                                                     rank_llm_name)
        data_to_save = {
            'rank_dict': rank_dict,
            'original_data': data_dict[key],
            'rank_llm': rank_llm_name
        }
        logger.info(f'client_id: {client_id}, success')
        json.dump(data_to_save, open(target_file_path, 'w', encoding='utf-8-sig'))

        if parse_index % 20 == 0:
            eval_performance(target_folder)
    eval_performance(target_folder)
    logger.info('data dict length: {}'.format(len(data_dict)))


if __name__ == '__main__':
    main()
