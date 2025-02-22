import csv
import os.path
from itertools import islice
from preprocess_config import joint_save_path


def read_data(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            table_discharge, icd_code = line[10:12]
            if table_discharge != "None" and icd_code != "None":
                data_list.append([table_discharge, icd_code])
    return data_list


data_list = read_data(joint_save_path)
with open(os.path.abspath('../resource/diagnosis_mapping.csv'), 'w', encoding='utf-8-sig') as f:
    csv.writer(f).writerows(data_list)
