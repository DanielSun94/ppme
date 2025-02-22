import os
from srrsh_config import (diagnosis_1_folder, diagnosis_2_folder, fused_diagnosis_file_path_template)
import json
import csv
import re

pattern = r'^[A-Z][0-9]|^[0-9]'

# 3
def read_diagnosis(folder_1, folder_2, use_icd):
    success_count, failed_count, icd_failed_count, json_load_error, format_error = 0, 0, 0, 0, 0
    success_dict = dict()
    head = ['pk_dcpv', 'pk_dcpvdiag', 'code_group', 'code_org', 'name_diagtype', 'name_diagsys', 'code_diag',
            'name_diag', 'date_diag', 'code_psn_diag', 'name_dept_diag']

    data_to_write = [head]
    for folder in [folder_1, folder_2]:
        file_list = os.listdir(folder)
        for file in file_list:
            file_path = os.path.join(folder, file)
            print('start parse file: {}'.format(file_path))
            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception as e:
                        # print('parse error: {}'.format(e))
                        failed_count += 1
                        json_load_error += 1
                        continue

                    if use_icd:
                        flag ='code_diag' not in data or data['code_diag'] is None or \
                              ('code_diag' in data and re.match(pattern, data['code_diag']) is None)
                    else:
                        flag = False
                    if flag:
                        failed_count += 1
                        icd_failed_count += 1

                        if 'date_diag' in data and data['date_diag'] is not None and len(data['date_diag']) > 4 \
                            and 'name_diagtype' in data and data['name_diagtype'] is not None:
                            year = data['date_diag'][:4]
                            diag_type = data['name_diagtype']
                            if year + '_' + diag_type not in success_dict:
                                success_dict[year + '_' + diag_type] = [0, 0]
                            success_dict[year + '_' + diag_type][1] += 1
                        continue

                    sample = []
                    success_flag = True
                    for key in head:
                        if key in data:
                            sample.append(data[key])
                        else:
                            if key in {'pk_dcpv', 'pk_dcpvdiag', 'name_diagtype', 'name_diag'}:
                                success_flag = False
                                format_error += 1
                                break
                            else:
                                sample.append('NONE')

                    if success_flag:
                        data_to_write.append(sample)
                        success_count += 1
                        if 'date_diag' in data and data['date_diag'] is not None and len(data['date_diag']) > 4 \
                            and 'name_diagtype' in data and data['name_diagtype'] is not None:
                            year = data['date_diag'][:4]
                            diag_type = data['name_diagtype']
                            if year + '_' + diag_type not in success_dict:
                                success_dict[year + '_' + diag_type] = [0, 0]
                            success_dict[year + '_' + diag_type][0] += 1
                    else:
                        failed_count += 1

                    if success_count > 0 and success_count % 100000 == 0:
                        print('success_count: {}, failed_count: {}, icd failed count: {}'
                              .format(success_count, failed_count, icd_failed_count))
                print('success_count: {}, failed_count: {}, icd failed count: {}, format error: {}, json_load_error: {}'
                      .format(success_count, failed_count, icd_failed_count, format_error, json_load_error))

    success_list = []
    for key in success_dict:
        success_list.append([key, success_dict[key]])
    success_list = sorted(success_list, key=lambda x: x[0])
    for item in success_list:
        print(item)

    with open(fused_diagnosis_file_path_template.format(use_icd), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def main():
    use_icd = False
    read_diagnosis(diagnosis_1_folder, diagnosis_2_folder, use_icd)



if __name__ == '__main__':
    main()
