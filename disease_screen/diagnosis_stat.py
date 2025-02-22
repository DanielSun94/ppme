import json

from disease_screen_logger import logger
import numpy as np
from read_data import get_symptom_index_map, read_diagnosis_symptom_embedding
from disease_screen_config import (data_file_path, symptom_file_path_template, symptom_info_dict)

category_list = [
    [0, '传染病和寄生虫病 A00-B99', 'A', 00, 'B', 99],  # Infectious parasitic diseases, 传染病和寄生虫病
    [1, '肿瘤 C00-D49', 'C', 00, 'D', 49],  # Neoplasms, 肿瘤
    [2, '血液病 D50-D89', 'D', 50, 'D', 89],
    # Diseases of the blood and blood forming organs and diseases involving the immune mechanism， 血液和造血奇怪疾病以及涉及免疫机制的疾病
    [3, '内分泌疾病 E00-E99', 'E', 00, 'E', 99],  # endocrine, nutritional and metabolic disease， 内分泌、营养和代谢病
    [4, '精神和行为障碍 F00-F99', 'F', 00, 'F', 99],  # mental and behavioural disorders 精神和行为障碍
    [5, '神经系统疾病 G00-G99', 'G', 00, 'G', 99],  # diseases of the nervous system 神经系统疾病
    [6, '眼科疾病 H00-H59', 'H', 00, 'H', 59],  # diseases of the eye and adnexa 眼科疾病
    [7, '耳部及乳突疾病 H60-H99', 'H', 60, 'H', 99],  # Diseases of the ear and mastoid process, 耳部及乳突疾病
    [8, '循环系统疾病 I00-I99', 'I', 00, 'I', 99],  # Diseases of the circulatory system, 循环系统疾病
    [9, '呼吸系统疾病 J00-J99', 'J', 00, 'J', 99],  # Diseases of the respiratory system, 呼吸系统疾病
    [10, '消化系统疾病 K00-K99', 'K', 00, 'K', 99],  # Diseases of the digestive system, 消化系统疾病
    [11, '皮肤和皮下组织疾病 L00-L99', 'L', 00, 'L', 99],  # Diseases of the skin and subcutaneous tissue, 皮肤和皮下组织疾病
    [12, '肌肉骨骼和结缔组织病 M00-M99', 'M', 00, 'M', 99],
    # Diseases of the musculoskeletal system and connective tissue, 肌肉骨骼和结缔组织病
    [13, '泌尿生殖系统疾病 N00-N99', 'N', 00, 'N', 99],  # Diseases of the genitourinary system, 泌尿生殖系统疾病
    # [14, '怀孕，产褥期疾病 O00-O9A', 'O', 00, 'O', 99]
]


def main():
    logger.info('start load')
    top_n, digit, weight = 1, 3, False
    diagnosis_lower = 20
    read_from_cache = True
    embedding_size = 1024
    use_symptom = 1
    embedding_type = 1
    filter_key = 'srrsh'
    # 两种策略，一种是只保留ICD 编码O及O以前的（后面的大部分是先天性疾病，或者不能够称为“病”），一种是所有都保留
    strategy = 'ALL'  # DISCARD_OL ALL
    data_split_strategy = 'custom'
    logger.info(f'top n: {top_n}, digit: {digit}, weight: {weight}, data_split_strategy: {data_split_strategy},'
                f'strategy: {strategy}')

    symptom_path_list = [
        [symptom_file_path_template.format('chn'), 'chn'],
        [symptom_file_path_template.format('eng'), 'eng']
    ]

    index_symptom_dict, symptom_index_dict = get_symptom_index_map(symptom_path_list)
    (train_dataset, valid_dataset, test_dataset, diagnosis_index_map, index_diagnosis_map) = (
        read_diagnosis_symptom_embedding(
            data_file_path, symptom_info_dict, symptom_index_dict, top_n, digit, weight, filter_key=filter_key,
            diagnosis_lower=diagnosis_lower, strategy=strategy, read_from_cache=read_from_cache,
            embedding_size=embedding_size, data_split_strategy=data_split_strategy, embedding_type=embedding_type,
            use_symptom=use_symptom))

    icu_cache_path = '/home/sunzhoujian/remote_development/ecdai/resource/icu_patient_idx.json'
    filter_data = json.load(open(icu_cache_path, 'r', encoding='utf-8-sig'))
    filter_set = set()
    for key, value in filter_data.items():
        if value == 1:
            filter_set.add(key)
    print('start stat')
    diagnosis_stat(index_diagnosis_map, filter_set, train_dataset, valid_dataset, test_dataset)


def parse_disease_idx(diagnosis_icd):
    diagnosis_icd = diagnosis_icd.upper()
    if not diagnosis_icd[1:3].isdigit():
        return -1
    index = -1
    for item in category_list:
        category_idx, key, category_token_start, number_start, category_token_end, number_end = item
        if diagnosis_icd[0] > category_token_start:
            if diagnosis_icd[0] > category_token_end:
                continue
            else:
                if int(diagnosis_icd[1:3]) <= number_end:
                    index = category_idx
                    break
                else:
                    continue
        elif diagnosis_icd[0] == category_token_start:
            if diagnosis_icd[0] == category_token_end:
                if number_start <= int(diagnosis_icd[1:3]) <= number_end:
                    index = category_idx
                    break
                else:
                    continue
            else:
                index = category_idx
                break
        else:
            break
    return index


def diagnosis_stat(index_diagnosis_map, filter_data=None, *dataset_list):
    diagnosis_count_dict = dict()
    for dataset in dataset_list:
        for item in dataset:
            key = item[0]
            if filter_data is not None and key not in filter_data:
                continue
            diagnosis_index = np.argmax(item[2])
            icd_code = index_diagnosis_map[diagnosis_index]
            code_idx = parse_disease_idx(icd_code)
            if code_idx not in diagnosis_count_dict:
                diagnosis_count_dict[code_idx] = 0
            diagnosis_count_dict[code_idx] += 1
    print(diagnosis_count_dict)


if __name__ == '__main__':
    main()
