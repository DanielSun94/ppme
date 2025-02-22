import json
import os
import csv
from itertools import islice
from preprocess_config import diagnosis_tree_folder, disease_knowledge_path


def main():
    icd_diagnosis_dict = read_diagnosis_tree(diagnosis_tree_folder)
    id_diagnosis_dict, diagnosis_id_dict = read_knowledge(disease_knowledge_path)

    match_count, fail_count = 0, 0
    for key in icd_diagnosis_dict:
        for item in icd_diagnosis_dict[key]:
            if item in diagnosis_id_dict:
                match_count += 1
            else:
                fail_count += 1
                print('key {} not found in diagnosis'.format(item))
    print(f'match count {match_count}, fail count {fail_count}')


def read_knowledge(file_path):
    id_diagnosis_dict, diagnosis_id_dict = dict(), dict()
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            id_no, key, disease_name, description_type, context = line[:5]
            if key not in id_diagnosis_dict:
                id_diagnosis_dict[key] = disease_name.strip()
            else:
                assert disease_name == id_diagnosis_dict[key]

            if disease_name not in diagnosis_id_dict:
                diagnosis_id_dict[disease_name] = key
    return id_diagnosis_dict, diagnosis_id_dict


def read_diagnosis_tree(folder_path):
    data_dict = dict()
    file_list = os.listdir(folder_path)
    for file in file_list:
        if 'disease_list_by_icd' not in file:
            continue
        file_path = os.path.join(folder_path, file)
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))

        key = file.replace('.json', '').replace('disease_list_by_icd_', '')
        # page_no = key.strip().split('_')[-1]
        icd = '.'.join(key.strip().split('_')[0:-1])

        if icd not in data_dict:
            data_dict[icd] = []
        for item in data['result']['datas']:
            if 'ji_bing_ming' not in item:
                print('disease name null error')
                continue
            data_dict[icd].append(item['ji_bing_ming'].strip())
    return data_dict


if __name__ == '__main__':
    main()
