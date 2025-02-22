# 6, 用于和其它多中心数据集联合使用
import csv
import json
from srrsh_config import full_primary_diagnosis_dataset_path, final_fusion_path
from itertools import islice


def main():
    head = ['data_source', 'visit_type', 'patient_visit_id', 'outpatient_record', 'admission_record',
            'comprehensive_history', 'discharge_record', 'first_page', 'discharge_diagnosis', 'first_page_diagnosis',
            'table_diagnosis', 'icd_code']
    data_to_write = [head]
    diagnosis_contradict = 0
    discharge_count, discharge_diagnosis_count, outpatient_count, outpatient_diagnosis_count = 0, 0, 0, 0
    with open(full_primary_diagnosis_dataset_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            admission_type, patient_visit_id, admission_record, history_record, outpatient_record, discharge_record, \
                discharge_note_diagnosis, table_diagnosis = line
            # 由于大量出院诊断丢失，且大量ICD编码不存在。此处的策略是住院数据只使用从discharge record中抽取的诊断
            # 门诊数据使用初步诊断，且不使用icd
            if admission_type == 'hospitalization':
                assert admission_record != 'None' and discharge_record != 'None' and history_record != 'None'
                discharge_note_diagnosis = json.loads(discharge_note_diagnosis)
                discharge_note_diagnosis = '$$$$$'.join(discharge_note_diagnosis)

                table_diagnosis_str, icd_code_str, diagnosis_count = \
                    table_diagnosis_convert(table_diagnosis, admission_type)
                if table_diagnosis_str == 'None':
                    assert discharge_note_diagnosis != 'None'
                else:
                    discharge_count += 1
                    discharge_diagnosis_count += diagnosis_count
                line = ['srrsh', admission_type, patient_visit_id, outpatient_record, admission_record, history_record,
                        discharge_record, 'None', discharge_note_diagnosis, 'None', table_diagnosis_str, icd_code_str]
            else:
                assert outpatient_record is not None
                assert admission_type == 'outpatient'
                table_diagnosis_str, icd_code_str, diagnosis_count = table_diagnosis_convert(table_diagnosis,
                                                                                             admission_type)
                outpatient_count += 1
                outpatient_diagnosis_count += diagnosis_count
                line = ['srrsh', admission_type, patient_visit_id, outpatient_record, admission_record, history_record,
                        discharge_record, 'None', 'None', 'None', table_diagnosis_str, icd_code_str]
            data_to_write.append(line)
    print(f'outpatient count: {outpatient_count}, avg diagnosis: {outpatient_diagnosis_count/outpatient_count}')
    print(f'discharge count: {discharge_count}, avg diagnosis: {discharge_diagnosis_count / discharge_count}')
    print('outpatient contradict: {}'.format(diagnosis_contradict))
    with open(final_fusion_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def table_diagnosis_convert(table_diagnosis, visit_type):
    if table_diagnosis == 'None':
        return "None", "None", 0

    table_diagnosis = json.loads(table_diagnosis)
    if len(table_diagnosis) == 0:
        table_diagnosis_str, icd_code_str, diagnosis_count = 'None', 'None', 0
    else:
        diagnosis_list = []
        for key in table_diagnosis:
            diagnosis_list.append([key, table_diagnosis[key][0], table_diagnosis[key][1]])
        diagnosis_list = sorted(diagnosis_list, key=lambda x: x[0])
        table_diagnosis_list, icd_list = [], []
        for item in diagnosis_list:
            table_diagnosis_list.append(item[1])
            if visit_type == 'hospitalization':
                assert item[2] != 'None'
            if item[2] != 'None':
                icd_list.append(item[2])
        table_diagnosis_str = '$$$$$'.join(table_diagnosis_list)
        if len(icd_list) > 0:
            icd_code_str = '$$$$$'.join(icd_list)
        else:
            icd_code_str = 'None'
        diagnosis_count = len(icd_list)
    return table_diagnosis_str, icd_code_str, diagnosis_count


if __name__ == '__main__':
    main()
