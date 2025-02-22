import pickle

from srrsh_config import fused_diagnosis_file_path_template, diagnosis_icd_count_pkl_path
import csv
import re
import os
from itertools import islice


def read_diagnosis(file_path, filter_type, max_line=-1, read_from_cache=True):
    if os.path.exists(diagnosis_icd_count_pkl_path) and read_from_cache:
        data_dict = pickle.load(open(diagnosis_icd_count_pkl_path, 'rb'))
        return data_dict

    data_dict, diagnosis_set = dict(), set()
    with (open(file_path, 'r', encoding='utf-8-sig') as f):
        reader = csv.reader(f)
        line_index = 0
        for line in islice(reader, 1, None):
            line_index += 1
            pk_dcpvdiag, code_diag, name_diag, diag_type = line[1], line[6], line[7], line[4]
            if pk_dcpvdiag in diagnosis_set:
                continue
            if not (filter_type == 'discharge' and diag_type == '主诊断(出院)'):
                continue
            diagnosis_set.add(pk_dcpvdiag)
            if name_diag not in data_dict:
                data_dict[name_diag] = dict()
            if code_diag != "NONE" and re.match(r'^[A-Z]', code_diag):
                if code_diag not in data_dict[name_diag]:
                    data_dict[name_diag][code_diag] = 0
                data_dict[name_diag][code_diag] += 1
            if line_index > max_line > 0:
                break
    with open(diagnosis_icd_count_pkl_path, 'wb') as f:
        pickle.dump(data_dict, f)
    return data_dict


def read_label_mapping(file_path):
    diagnosis_mapping = {}
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            diagnosis, code = line[0], line[4]
            assert diagnosis not in diagnosis_mapping
            diagnosis_mapping[diagnosis] = code
    return diagnosis_mapping


def joint_analysis(diagnosis_mapping, original_data_dict, acc_level):
    success_count = 0
    full_count = 0
    inconsistent_list = []
    for diagnosis in diagnosis_mapping:
        code = diagnosis_mapping[diagnosis]
        if diagnosis not in original_data_dict:
            continue
        code_candidate_dict = original_data_dict[diagnosis]
        for original_code in code_candidate_dict:
            original_code_count = code_candidate_dict[original_code]
            full_count += original_code_count
            if acc_level == 1:
                original_code_first_three = original_code[:3]
                code = code[:3]
                if code == original_code_first_three:
                    success_count += original_code_count
                else:
                    inconsistent_list.append([diagnosis, code, original_code, original_code_count])
            else:
                assert acc_level == 2
                if len(code) == 3:
                    original_code_first_three = original_code[:3]
                    code = code[:3]
                    if code == original_code_first_three:
                        success_count += original_code_count
                    else:
                        inconsistent_list.append([diagnosis, code, original_code, original_code_count])
                else:
                    if code[3] == '.':
                        original_code_first_three = original_code[:5]
                        code = code[:5]
                        if code == original_code_first_three:
                            success_count += original_code_count
                        else:
                            inconsistent_list.append([diagnosis, code, original_code, original_code_count])
                    else:
                        original_code_first_three = original_code[:3]
                        code = code[:3]
                        if code == original_code_first_three:
                            success_count += original_code_count
                        else:
                            inconsistent_list.append([diagnosis, code, original_code, original_code_count])

    inconsistent_list = sorted(inconsistent_list, key=lambda i: i[3], reverse=True)
    for item in inconsistent_list:
        if item[3] > 10:
            print('diagnosis: {}, transformed code: {}, original code: {}, count: {}'
                  .format(item[0], item[1], item[2], item[3]))
    print('full_count: {}, success count: {}, match ratio: {}'
          .format(full_count,success_count,success_count/full_count))


def main():
    max_line = -1
    filter_type = 'discharge'
    label_mapping_save_path = os.path.abspath('../resource/{}_label_mapping.csv'.format("glm-4-9b-chat"))
    diagnosis_path = fused_diagnosis_file_path_template.format('False')
    original_data_dict = read_diagnosis(diagnosis_path, filter_type, max_line=max_line)
    print('load diagnosis success')
    diagnosis_mapping = read_label_mapping(label_mapping_save_path)
    joint_analysis(diagnosis_mapping, original_data_dict, acc_level=1)
    joint_analysis(diagnosis_mapping, original_data_dict, acc_level=2)


if __name__ == '__main__':
    main()
