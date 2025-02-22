import os
import csv
import random
from preprocess_config import joint_save_path, final_joint_data_folder
from itertools import islice


def diagnosis_inclusion_dict(data_list, digit_num, threshold):
    reserved_set = set()
    code_count_dict, code_sum = dict(), 0
    for i in range(len(data_list)):
        disease_code_list = data_list[i][-1].split('$$$$$')
        for item in disease_code_list:
            if len(item) == 0:
                continue
            item = item.replace('.', '')[:digit_num]
            if item not in code_count_dict:
                code_count_dict[item] = 0
            code_count_dict[item] += 1
            code_sum += 1
    code_count_list = [[key, code_count_dict[key]] for key in code_count_dict]
    code_count_list = sorted(code_count_list, key=lambda x: x[1], reverse=True)
    for i in range(50):
        print(f'code: {code_count_list[i][0]}, count: {code_count_list[i][1]}')
    reserve_list, cumulative_sum = [], 0
    for item in code_count_list:
        code, count = item
        reserved_set.add(code)
        cumulative_sum += count
        if cumulative_sum / code_sum >= threshold:
            break
    return reserved_set


def reformat_label(data_list, reserved_set, digit_num):
    final_data = []
    for i in range(len(data_list)):
        disease_code_list = data_list[i][-1].split('$$$$$')
        final_code_list = []
        for item in disease_code_list:
            if len(item) == 0:
                continue
            code = item.replace('.', '')[:digit_num]
            if code in reserved_set and code not in final_code_list:
                final_code_list.append(code)
        if len(final_code_list) > 0:
            final_code_str = '$$$$$'.join(final_code_list)
            final_line = data_list[i][0:-1] + [final_code_str]
            final_data.append(final_line)
    return final_data


def reformat_data(file_path, diagnosis_name_mapping_dict, use_discharge_mapping_code):
    data_to_write = []
    # read original data
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            discharge_diagnosis, first_page_diagnosis, table_diagnosis = line[8:11]

            if len(line) == 12:
                affiliated_info = ''
            else:
                assert len(line) == 13
                affiliated_info = line[12]

            diagnosis_code_list = []
            for diagnosis in discharge_diagnosis, first_page_diagnosis, table_diagnosis:
                if diagnosis != 'None':
                    diagnosis_list = diagnosis.split('$$$$$')
                    code_list = []
                    for single_diagnosis in diagnosis_list:
                        if len(single_diagnosis) == 0:
                            continue
                        if single_diagnosis in diagnosis_name_mapping_dict:
                            code_list.append(diagnosis_name_mapping_dict[single_diagnosis])
                        else:
                            code_list.append('NA')
                    diagnosis_code_list.append("$$$$$".join(code_list))
                else:
                    diagnosis_code_list.append('None')

            if use_discharge_mapping_code:
                oracle_diag = 'None'
                if line[11] != 'None':
                    oracle_diag = line[11].replace('.', '')
                elif diagnosis_code_list[1] != 'None':
                    oracle_diag = diagnosis_code_list[1]
                elif diagnosis_code_list[0] != 'None':
                    oracle_diag = diagnosis_code_list[0]
                elif diagnosis_code_list[2] != 'None':
                    oracle_diag = diagnosis_code_list[2]
                if oracle_diag != 'None':
                    line = line[:12] + diagnosis_code_list + [affiliated_info, oracle_diag]
                    assert len(line) == 17
                    data_to_write.append(line)
            else:
                if line[11] != 'None':
                    oracle_diag = line[11].replace('.', '')
                    line = line[:12] + diagnosis_code_list + [affiliated_info, oracle_diag]
                    assert len(line) == 17
                    data_to_write.append(line)
    return data_to_write


def final_stat(data_list):
    count_dict = dict()
    for item in data_list:
        source, visit_type = item[0], item[1]
        unified_id = source + '-' + visit_type
        if unified_id not in count_dict:
            count_dict[unified_id] = 0
        count_dict[unified_id] += 1
    for key in count_dict:
        print(key, count_dict[key])


def main():
    # read icd label
    sample_size = 10240
    digit_num = 4
    threshold = 1.00
    use_discharge_mapping_code = False
    print(f'sample size: {sample_size}, digit_num: {digit_num}, threshold: {threshold}, '
          f'use_discharge_mapping_code: {use_discharge_mapping_code}')

    diagnosis_name_mapping_dict = dict()
    label_mapping_save_path = os.path.abspath('../resource/joint_dataset_diagnosis_mapping.csv')
    with open(label_mapping_save_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            diagnosis_name, code = line[0], line[4]
            diagnosis_name_mapping_dict[diagnosis_name] = code
    print('icd loaded success')
    data_to_write = reformat_data(joint_save_path, diagnosis_name_mapping_dict, use_discharge_mapping_code)
    print('load data')
    reserved_set = diagnosis_inclusion_dict(data_to_write, digit_num, threshold)
    print('icd code stat success')
    data_to_write = reformat_label(data_to_write, reserved_set, digit_num)
    print('reformat label success')
    final_stat(data_to_write)
    print('final data size: {}'.format(len(data_to_write)))
    head = ['data_source', 'visit_type', 'patient_visit_id', 'outpatient_record', 'admission_record',
            'comprehensive_history', 'discharge_record', 'first_page', 'discharge_diagnosis', 'first_page_diagnosis',
            'table_diagnosis', 'table_icd_code_from_data', 'discharge_diagnosis_code', 'first_page_diagnosis_code',
            'table_diagnosis_code_parsed', 'affiliated_info', 'oracle_diagnosis']

    print('data load success')
    os.makedirs(final_joint_data_folder, exist_ok=True)
    file_path = os.path.join(final_joint_data_folder, f'use_code_map_{use_discharge_mapping_code}_'
                                                      f'digit_{digit_num}_fraction_{threshold}.csv')
    data = [head] + data_to_write
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data)

    sample_path = os.path.join(final_joint_data_folder, f'sample_use_code_map_{use_discharge_mapping_code}'
                                                        f'_digit_{digit_num}_fraction_{threshold}.csv')
    with open(sample_path, 'w', encoding='utf-8-sig', newline='') as f:
        index_list = [i for i in range(1, len(data))]
        random.Random(715).shuffle(index_list)
        index_list = [0] + index_list
        sample_data = [head]
        for i, idx in enumerate(index_list):
            if i >= sample_size:
                break
            sample_data.append(data_to_write[idx])
        csv.writer(f).writerows(sample_data)


if __name__ == '__main__':
    main()
