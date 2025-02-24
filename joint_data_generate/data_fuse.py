import csv
from itertools import islice
from preprocess_config import (langtong_ssrsh_path, langtong_xiamen_path, langtong_guizhou_path, langtong_wenfuyi_path,
                               langtong_xiangya_path, langtong_shengzhou_path, full_diagnosis_info_srrsh_dataset,
                               joint_save_path, mimic_iii_path, mimic_iv_path)


def main():
    path_dict = {
        'mimic_iii': mimic_iii_path,
        'mimic_iv': mimic_iv_path,
    }

    head = ['data_source', 'visit_type', 'patient_visit_id', 'outpatient_record', 'admission_record',
            'comprehensive_history', 'discharge_record', 'first_page', 'discharge_diagnosis', 'first_page_diagnosis',
            'table_diagnosis', 'icd_code', 'affiliated_info']
    data_list = [head]
    for key in path_dict:
        path = path_dict[key]
        data_list += read_data(path, key)

    with open(joint_save_path, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_list)


def read_data(file_path, key):
    data = []
    hospitalization_count, outpatient_count = 0, 0
    hospitalization_len, outpatient_len = 0, 0
    count = 0
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for line in islice(csv_reader, 1, None):
            if line[1] == 'hospitalization':
                hospitalization_count += 1
                if 'mimic' in line[0]:
                    hospitalization_len += len(line[4].split(' '))
                else:
                    hospitalization_len += len(line[4])
            else:
                assert line[1] == 'outpatient'
                outpatient_count += 1
                outpatient_len += len(line[3])
            if not (len(line) == 12 or len(line) == 13):
                print('error')
            count += 1
            data.append(line)

    print(f'key: {key}, data length: {len(data)}, hospitalization: {hospitalization_count}, '
          f'outpatient: {outpatient_count}, hospitalization len: {hospitalization_len/ count}, '
          f'outpatient len: {outpatient_len / count}')
    return data


if __name__ == '__main__':
    main()
