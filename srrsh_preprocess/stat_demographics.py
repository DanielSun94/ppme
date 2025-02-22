import json
import os
from srrsh_config import emr_ns_folder
from srrsh_logger import logger


def main():
    settings = 'full'  # severe
    file_list = []

    data_dict = dict()
    sub_folder_list = os.listdir(emr_ns_folder)
    for sub_folder in sub_folder_list:
        sub_folder_path = os.path.join(emr_ns_folder, sub_folder)
        file_name_list = os.listdir(sub_folder_path)
        for file_name in file_name_list:
            file_path = os.path.join(sub_folder_path, file_name)
            file_list.append(file_path)
    logger.info(f'len file_list: {len(file_list)}')
    failed_count, success_count = 0, 0
    for i, file_path in enumerate(file_list):
        if i > 0 and i % 100 == 0:
            temporal_stat(data_dict)

        logger.info('start parse file: {}'.format(file_path))
        with (open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f):
            for line in f:
                try:
                    data = json.loads(line)
                    success_count += 1
                except Exception as e:
                    # traceback.print_exc()
                    failed_count += 1
                    continue

                try:
                    if not isinstance(data_dict, dict) or \
                            'pname' not in data or 'inputfield' not in data or 'pk_rec_data' not in data:
                        continue
                except Exception as e:
                    # traceback.print_exc()
                    failed_count += 1
                    continue

                pk_rec_data = data['pk_rec_data']
                if pk_rec_data not in data_dict:
                    data_dict[pk_rec_data] = {'age': None, 'sex': None}
                if data['pname'] == '年龄':
                    age = data['inputfield']
                    if len(age) > 0 and len(age) <= 3:
                        try:
                            data_dict[pk_rec_data]['age'] = int(age)
                        except Exception as e:
                            failed_count += 1
                            continue
                if data['pname'] == '性别':
                    sex = data['inputfield']
                    if len(sex) > 0:
                        data_dict[pk_rec_data]['sex'] = sex.strip()
            # break
    logger.info('data loaded')
    logger.info('success_count: {}, failed_count: {}'.format(success_count, failed_count))


def temporal_stat(data_dict):
    age_list = []
    sex_dict = dict()
    age_failed, sex_failed = 0, 0
    for key in data_dict:
        age, sex = data_dict[key]['age'], data_dict[key]['sex']
        if age is None:
            age_failed += 1
        else:
            age_list.append(age)
        if sex is None:
            sex_failed += 1
        else:
            if sex not in sex_dict:
                sex_dict[sex] = 0
            else:
                sex_dict[sex] += 1
    print('sex dict')
    print(sex_dict)

    print('age dict')
    age_distribution = {"18-40": 0, "40-60": 0, "60-80": 0, "80+": 0}
    for age in age_list:
        if age < 40:
            age_distribution["18-40"] += 1
        elif age < 60:
            age_distribution["40-60"] += 1
        elif age < 80:
            age_distribution["60-80"] += 1
        else:
            age_distribution["80+"] += 1
    print(age_distribution)
    print('success')


if __name__ == '__main__':
    main()
