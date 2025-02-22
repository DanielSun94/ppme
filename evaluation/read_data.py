import csv
import json
import random
from itertools import islice
from evaluation_config import srrsh_severe_path


srrsh_severe_dict_full = json.load(open(srrsh_severe_path, 'r', encoding='utf-8-sig'))
srrsh_severe_set = set()
for key in srrsh_severe_dict_full:
    if srrsh_severe_dict_full[key] == 1:
        srrsh_severe_set.add(key[6:])


def filter_data(data_dict, symptom_num_dict, filter_criteria, start_index, end_index):
    # 此处有领个filter标准，第一个标准是在test中positive num大于0（在symptom num dict中）
    # 其实symptom num是最终的数据集，symptom positive num为0的都删除了
    # 第二个要求是满足筛选标准
    target_info_dict = dict()
    for unified_id in symptom_num_dict:
        data_fraction, symptom_num, modify_time = symptom_num_dict[unified_id]
        if symptom_num > 0 and data_fraction == 'test':
            pure_id = '-'.join(unified_id.strip().split('-')[1:])
            target_info_dict[pure_id] = modify_time

    key_list = sorted([item for item in data_dict.keys()])
    if filter_criteria == 'mimic_iv':
        filter_func = filter_criteria_hospitalization_mimic
    elif filter_criteria == 'mimic_iii':
        filter_func = filter_criteria_hospitalization_mimic
    elif filter_criteria == 'mimic':
        filter_func = filter_criteria_hospitalization_mimic
    elif filter_criteria == 'srrsh-hospitalization':
        filter_func = filter_criteria_hospitalization_srrsh
    elif filter_criteria == 'srrsh-outpatient':
        filter_func = filter_criteria_outpatient_srrsh
    elif filter_criteria == 'srrsh-hospitalization-severe':
        filter_func = filter_criteria_hospitalization_srrsh_severe
    else:
        assert filter_criteria == 'srrsh'
        filter_func = filter_criteria_srrsh

    print(f'legal data size: {len(target_info_dict)}')
    new_data_list = []
    for key in key_list:
        sample = data_dict[key]
        if filter_func(sample, key) and key in target_info_dict:
            modify_time = target_info_dict[key]
            new_data_list.append([key, sample, modify_time])
    # 这里原先的设计是根据modify time进行筛选，后改为固定种子随机化后筛选。原因是srrsh合并，住院病人整体早于门诊。
    random.Random(715).shuffle(new_data_list)
    return_data = new_data_list[start_index:end_index]
    print(f'load data size: {len(new_data_list)}')
    return return_data


def filter_criteria_hospitalization_srrsh_severe(sample, key):
    # Filter包含以下几个要求
    # 1. 必须是住院病例数据
    # 2. 必须有直接给定的ICD编码
    filter_set = srrsh_severe_set
    icd_code = sample['oracle_diagnosis']
    if 'srrsh' not in key:
        return False

    if key not in filter_set:
        return False

    pass_flag = True
    if len(icd_code) < 3:
        pass_flag = False
    return pass_flag


def filter_criteria_srrsh(sample, key):
    # Filter包含以下几个要求
    # 1. 必须是住院病例数据
    # 2. 必须有直接给定的ICD编码
    icd_code = sample['oracle_diagnosis']
    if 'srrsh' not in key:
        return False

    pass_flag = True
    if len(icd_code) < 3:
        pass_flag = False
    return pass_flag


def filter_criteria_outpatient_srrsh(sample, key):
    # Filter包含以下几个要求
    # 1. 必须是住院病例数据
    # 2. 必须有直接给定的ICD编码
    # 3. 疾病范围限定为A00-B99，D50-N99。即所有常见疾病。但不包括肿瘤，妊娠和先天性疾病
    icd_code = sample['oracle_diagnosis']
    visit_type = sample['visit_type']
    if 'srrsh' not in key:
        return False

    pass_flag = True
    if visit_type != 'outpatient':
        pass_flag = False

    if len(icd_code) < 3:
        pass_flag = False
    # else:
    #     # 注意，按照当前的设计，只取第一个诊断前4位
    #     icd = icd_code.lower()[:4]
    #     if icd[0] == 'c':
    #         pass_flag = False
    #     if icd[0] in {'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}:
    #         pass_flag = False
    #     if icd[0] == 'd':
    #         if icd[1] in {'0', '1', '2', '3'}:
    #             pass_flag = False
    return pass_flag


def filter_criteria_hospitalization_srrsh(sample, key):
    # Filter包含以下几个要求
    # 1. 必须是住院病例数据
    # 2. 必须有直接给定的ICD编码
    # 3. 疾病范围限定为A00-B99，D50-N99。即所有常见疾病。但不包括肿瘤，妊娠和先天性疾病
    icd_code = sample['oracle_diagnosis']
    visit_type = sample['visit_type']
    if 'srrsh' not in key:
        return False

    pass_flag = True
    if visit_type != 'hospitalization':
        pass_flag = False

    if len(icd_code) < 3:
        pass_flag = False
    # else:
    #     # 注意，按照当前的设计，只取第一个诊断前4位
    #     icd = icd_code.lower()[:4]
    #     if icd[0] == 'c':
    #         pass_flag = False
    #     if icd[0] in {'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}:
    #         pass_flag = False
    #     if icd[0] == 'd':
    #         if icd[1] in {'0', '1', '2', '3'}:
    #             pass_flag = False
    return pass_flag


def filter_criteria_hospitalization_mimic(sample, key):
    # Filter包含以下几个要求
    # 1. 必须是住院病例数据
    # 2. 必须有直接给定的ICD编码
    # 3. 疾病范围限定为A00-B99，D50-N99。即所有常见疾病。但不包括肿瘤，妊娠和先天性疾病
    icd_code = sample['oracle_diagnosis']

    if 'mimic' not in key:
        return False

    pass_flag = True
    if len(icd_code) < 4:
        pass_flag = False
    else:
        if icd_code[0] in {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}:
            raise ValueError('')
    return pass_flag


def load_data(source_file, max_size=-1):
    data_dict = dict()
    with open(source_file, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            data_source = line[0]
            visit_type = line[1]
            patient_visit_id = line[2]
            unified_key = data_source + '-' + visit_type + '-' + patient_visit_id
            if len(data_dict) > max_size > -1:
                break
            assert unified_key not in data_dict
            data_dict[unified_key] = {
                'source': line[0],
                'visit_type': line[1],
                'patient_visit_id': line[2],
                'outpatient_record': line[3],
                'admission_record': line[4],
                'comprehensive_history': line[5],
                'discharge_record': line[6],
                'first_page': line[7],
                'discharge_diagnosis': line[8],
                'first_page_diagnosis': line[9],
                'table_diagnosis': line[10],
                'table_icd_code_from_data': line[11],
                'discharge_diagnosis_code': line[12],
                'first_page_diagnosis_code': line[13],
                'table_diagnosis_code_parsed': line[14],
                'affiliated_info': line[15],
                'oracle_diagnosis': line[16]
            }

    empty_set = {'', 'none', 'null', 'na', 'n/a'}
    for unified_key in data_dict:
        for item_key in data_dict[unified_key]:
            value = data_dict[unified_key][item_key]
            if value.lower() in empty_set:
                data_dict[unified_key][item_key] = ''
    return data_dict


def select_data_set(data_dict):
    # 后面再修改，现在默认返回一个数据
    return_data = None
    key = None
    for key in data_dict:
        if 'hospitalization' in key:
            return_data = data_dict[key]
            break
    assert return_data is not None
    return [[key, return_data]]
