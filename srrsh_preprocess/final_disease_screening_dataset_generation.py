import json
import os
import csv
from srrsh_config import (full_primary_diagnosis_dataset_path, final_disease_screening_ready_data_path)
from itertools import islice

# 6 用于乐心的工作交付相关。后续不采用这个代码
def main():
    # read original data
    data_dict = dict()
    with open(full_primary_diagnosis_dataset_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            visit_type, patient_visit_id, admission_record, history_record, outpatient_record, _, diagnosis_list, table_diagnosis = line
            data_dict[patient_visit_id] = visit_type, admission_record, history_record, \
                outpatient_record, diagnosis_list, table_diagnosis
    print('data load success')

    # read icd label
    diagnosis_name_mapping_dict = dict()
    model_id_save = 'ZhipuAI/glm-4-9b-chat'.split('/')[1]
    label_mapping_save_path = os.path.abspath('../resource/{}_label_mapping.csv'.format(model_id_save))
    with open(label_mapping_save_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            diagnosis_name, code = line[:2]
            diagnosis_name_mapping_dict[diagnosis_name] = code
    print('icd loaded success')

    # fuse
    data_to_write = [['patient_visit_id', 'visit_type', 'admission_record', 'history_record', 'outpatient_record',
                      'diagnosis_list', 'table_diagnosis', 'diagnosis_code_list']]
    for patient_visit_id in data_dict:
        visit_type, admission_record, history_record, outpatient_record, diagnosis_list, table_diagnosis = \
                data_dict[patient_visit_id]
        code_list = []
        hit_count = 0
        if diagnosis_list != 'None':
            diagnosis_list_json = json.loads(diagnosis_list)
            for diagnosis in diagnosis_list_json:
                if diagnosis in diagnosis_name_mapping_dict:
                    hit_count += 1
                    code_list.append(diagnosis_name_mapping_dict[diagnosis])
                else:
                    code_list.append("NONE")
            code_list = json.dumps(code_list)
        else:
            code_list = 'None'

        data_to_write.append([patient_visit_id, visit_type, admission_record, history_record, outpatient_record,
                              diagnosis_list, table_diagnosis, code_list])

    with open(final_disease_screening_ready_data_path, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_to_write)


if __name__ == '__main__':
    main()
