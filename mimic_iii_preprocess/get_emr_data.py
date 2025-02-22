import csv
from itertools import islice
import numpy as np
from mimic_iii_config import diagnosis_data_path, emr_data_path, save_file, icd_mapping_file


def read_diagnosis(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            _, patient_id, visit_id, seq_num, icd_code = line
            unified_id = patient_id + '_' + visit_id
            if unified_id not in data_dict:
                data_dict[unified_id] = []
            if len(seq_num) > 0:
                data_dict[unified_id].append([int(seq_num), icd_code])
    for unified_id in data_dict:
        data_dict[unified_id] = sorted(data_dict[unified_id], key=lambda x: x[0])
    return data_dict


def read_icd_9_10_mapping(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            icd_9, icd_10 = line[:2]
            data_dict[icd_9] = icd_10
    return data_dict


def icd_convert(data_dict, mapping_dict):
    converted_dict = dict()
    icd_10_count, success_count, failure_count, total_count = 0, 0, 0, 0
    for unified_id in data_dict:
        sample = []
        for item in data_dict[unified_id]:
            if item[1] in mapping_dict:
                icd_10 = mapping_dict[item[1]]
                sample.append([item[0], icd_10])
                success_count += 1
            else:
                failure_count += 1
            total_count += 1

        converted_dict[unified_id] = sample
    print('total count: {}, success count: {}, success ratio: {}'
          .format(total_count, success_count, success_count / total_count))
    return converted_dict


def extract_admission_info(data_dict):
    # mimic-iii的出院小结不做进一步清洗（格式变动的比较复杂，收益不是特别大）
    admission_dict = dict()
    for unified_id in data_dict:
        emr = data_dict[unified_id]
        admission_dict[unified_id] = emr
    return admission_dict

def extract_discharge_info(data_dict):
    # mimic-iii的出院小结不做进一步清洗（格式变动的比较复杂，收益不是特别大）
    discharge_dict = dict()
    for unified_id in data_dict:
        emr = data_dict[unified_id]
        discharge_dict[unified_id] = emr
    return discharge_dict


# 1
def main():
    emr_dict = read_emr(emr_data_path)
    admission_info_dict = extract_admission_info(emr_dict)
    discharge_info_dict = extract_discharge_info(emr_dict)

    icd_mapping_dict = read_icd_9_10_mapping(icd_mapping_file)
    diagnosis_dict = read_diagnosis(diagnosis_data_path)
    # 这一步的icd转换几乎是无损的（99.75%的数据可以被保留）
    converted_diagnosis_dict = icd_convert(diagnosis_dict, icd_mapping_dict)

    head = ['data_source', 'visit_type', 'patient_visit_id', 'outpatient_record', 'admission_record',
            'comprehensive_history', 'discharge_record', 'first_page', 'discharge_diagnosis',
            'first_page_diagnosis',
            'table_diagnosis', 'icd_code']
    data_to_write = [head]
    total_count = 0
    for unified_id in converted_diagnosis_dict:
        if unified_id not in admission_info_dict or unified_id not in discharge_info_dict:
            continue
        total_count += 1
        admission_record = admission_info_dict[unified_id]
        discharge_record = discharge_info_dict[unified_id]
        diagnosis = converted_diagnosis_dict[unified_id]
        icd_code = '$$$$$'.join([item[1] for item in diagnosis])
        line = ['mimic_iii', 'hospitalization', unified_id, 'None', admission_record, 'None', discharge_record, 'None',
                'None', 'None', "None", icd_code]
        data_to_write.append(line)
    print('total count: {}'.format(total_count))
    with open(save_file, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)



def read_emr(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            _, patient_id, visit_id, chart_date, __, ___, category, description, cgid, is_error, text = line
            unified_id = patient_id + '_' + visit_id
            if category != 'Discharge summary':
                continue
            data_dict[unified_id] = text
    return data_dict



if __name__ == "__main__":
    main()
