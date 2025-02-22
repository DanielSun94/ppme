# 用于分析模型的还原能力
import os
import csv
import pickle
from itertools import islice
from evaluation_logger import logger
from doctor_serve.doctor_config import symptom_file_path
from doctor_serve.ecdai_doctor_util import read_symptom, parse_symptom, construct_question_list
from doctor_serve.util import non_streaming_call_llm
from util import read_eval_data
from evaluation_config import evaluation_result_cache_folder, structured_symptom_folder, symptom_recover_path_template


def call_llm(prompt):
    return non_streaming_call_llm("local_qwen_2__5_72b_int4_2", prompt)


def main():
    language = 'eng'
    llm_name = 'deepseek_r1_70b'
    turn_num = "10"
    filter_key_list = ['mimic_iv', 'llm', 'hospitalization']
    logger.info(f'language: {language}')
    for item in filter_key_list:
        logger.info(f'filter key: {item}')

    symptom_dict = read_symptom(symptom_file_path.format(language))
    level_1_list, level_2_list_dict = (
        construct_question_list(
            symptom_dict, language=language, maximum_questions=30))
    symptom_mapping_dict = get_symptom_list(structured_symptom_folder)
    data_dict = read_eval_data(evaluation_result_cache_folder, llm_name, turn_num, *filter_key_list)
    logger.info('data dict length: {}'.format(len(data_dict)))
    save_path = symptom_recover_path_template.format('_'.join(filter_key_list+[llm_name, turn_num]))
    if os.path.exists(save_path):
        result_dict = pickle.load(open(save_path, 'rb'))
        eval_performance(result_dict, save_path)
    else:
        result_dict = dict()
    for key in data_dict.keys():
        if key in result_dict:
            logger.info(f'key: {key} parsed')
            continue
        assert key in symptom_mapping_dict
        label_general_symptom_dict, label_specific_symptom_dict = read_symptom_file(symptom_mapping_dict[key])
        dialogue, _ = data_dict[key]
        _, symptom_info, general_symptom_dict = (
            parse_symptom(dialogue, "", symptom_dict, level_1_list, level_2_list_dict, call_llm,
                          '', language)
        )
        result = evaluation(label_general_symptom_dict, label_specific_symptom_dict, general_symptom_dict, symptom_info)

        result_dict[key] = result
        if len(result_dict) > 0 and len(result_dict) % 20 == 0:
            pickle.dump(result_dict, open(save_path, 'wb'))
            eval_performance(result_dict, save_path)
        logger.info(f'key: {key} success')
    eval_performance(result_dict, save_path)


def eval_performance(result_dict, save_path):
    logger.info(f'save path: {save_path}')
    logger.info(f'len result_dict: {len(result_dict)}')
    lv_1_tp, lv_1_tn, lv_1_fp, lv_1_fn = 0, 0, 0, 0
    lv_2_tp, lv_2_tn, lv_2_fp, lv_2_fn = 0, 0, 0, 0
    full_tp, full_tn, full_fp, full_fn = 0, 0, 0, 0
    for key in result_dict.keys():
        for level in result_dict[key]:
            if level == 'level_1':
                lv_1_tp += result_dict[key][level]['tp']
                lv_1_fn += result_dict[key][level]['fn']
                lv_1_fp += result_dict[key][level]['fp']
                lv_1_tn += result_dict[key][level]['tn']
            else:
                assert level == 'level_2'
                lv_2_tp += result_dict[key][level]['tp']
                lv_2_fn += result_dict[key][level]['fn']
                lv_2_fp += result_dict[key][level]['fp']
                lv_2_tn += result_dict[key][level]['tn']
            full_tp += result_dict[key][level]['tp']
            full_fn += result_dict[key][level]['fn']
            full_fp += result_dict[key][level]['fp']
            full_tn += result_dict[key][level]['tn']

    for prefix, (tp, tn, fp, fn) in \
        zip(['level_1', 'level_2', 'full'],
            [(lv_1_tp, lv_1_tn, lv_1_fp, lv_1_fn),
             (lv_2_tp, lv_2_tn, lv_2_fp, lv_2_fn),
             (full_tp, full_tn, full_fp, full_fn)]):
        f1 = 2 * tp / (2 * tp + fn + fp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        logger.info(f'{prefix}, precision： {precision}, recall: {recall}, f1: {f1}')
    logger.info('')


def evaluation(label_general_dict, label_specific_dict, predict_general_dict, predict_specific_dict):
    level_1_tp, level_1_fp, level_1_fn, level_1_tn = 0, 0, 0, 0
    level_2_tp, level_2_tn, level_2_fp, level_2_fn = 0, 0, 0, 0
    for key in label_general_dict.keys():
        label = label_general_dict[key]
        predict = predict_general_dict[key.lower()]
        if label == 'YES' and predict == 'YES':
            level_1_tp += 1
        elif label == "YES" and predict != 'YES':
            level_1_fn += 1
        elif label != 'YES' and predict == 'YES':
            level_1_fp += 1
        else:
            assert label != 'YES' and predict != 'YES'
            level_1_tn += 1
    for key in label_specific_dict.keys():
        for sub_key in label_specific_dict[key]:
            for sub_sub_key in label_specific_dict[key][sub_key]:
                label = label_specific_dict[key][sub_key][sub_sub_key]
                predict = predict_specific_dict[key.lower()][sub_key.lower()][sub_sub_key.lower()]
                if label == 'YES' and predict == 'YES':
                    level_2_tp += 1
                elif label == "YES" and predict != 'YES':
                    level_2_fn += 1
                elif label != 'YES' and predict == 'YES':
                    level_2_fp += 1
                else:
                    assert label != 'YES' and predict != 'YES'
                    level_2_tn += 1
    result_dict = {
        'level_1': {'tp': level_1_tp, 'fp': level_1_fp, 'fn': level_1_fn, 'tn': level_1_tn},
        'level_2': {'tp': level_2_tp, 'fp': level_2_fp, 'fn': level_2_fn, 'tn': level_2_tn},
    }
    return result_dict


def read_symptom_file(file_path):
    general_dict, specific_dict = dict(), dict()
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            symptom, relation, factor, state = line
            if relation != 'N/A':
                assert factor != 'N/A'
                if symptom not in specific_dict:
                    specific_dict[symptom] = dict()
                if relation not in specific_dict[symptom]:
                    specific_dict[symptom][relation] = dict()
                specific_dict[symptom][relation][factor] = state
            else:
                general_dict[symptom] = state
                if symptom not in specific_dict:
                    specific_dict[symptom] = dict()
    return general_dict, specific_dict

def get_symptom_list(folder):
    data_dict = dict()
    llm_name_list = os.listdir(folder)
    for llm_name in llm_name_list:
        llm_folder_path = os.path.join(folder, llm_name)
        data_source_list = os.listdir(llm_folder_path)
        for data_source in data_source_list:
            data_source_path = os.path.join(llm_folder_path, data_source)
            visit_type_list = os.listdir(data_source_path)
            for visit_type in visit_type_list:
                visit_type_path = os.path.join(data_source_path, visit_type)
                prefix_number_list = os.listdir(visit_type_path)
                for prefix_number in prefix_number_list:
                    prefix_number_path = os.path.join(visit_type_path, prefix_number)
                    file_list = os.listdir(prefix_number_path)
                    for file_name in file_list:
                        if 'symptom' in file_name:
                            file_name_path = os.path.join(prefix_number_path, file_name)
                            data_dict[file_name[:-12]] = file_name_path
    return data_dict


if __name__ == '__main__':
    main()
