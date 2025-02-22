import csv
from itertools import islice
import numpy as np
from mimic_iv_config import diagnosis_data_path, discharge_data_path, save_file, icd_mapping_file


def read_diagnosis(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, seq_num, icd_code, icd_version = line
            unified_id = patient_id + '_' + visit_id
            if unified_id not in data_dict:
                data_dict[unified_id] = []
            data_dict[unified_id].append([int(seq_num), icd_code, icd_version])

    for unified_id in data_dict:
        diagnosis_list = sorted(data_dict[unified_id], key=lambda x: x[0])
        data_dict[unified_id] = diagnosis_list
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
            if item[2] == '10':
                sample.append([item[0], item[1]])
                icd_10_count += 1
            else:
                assert item[2] == '9'
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
    print('General success ratio: {}'.format((success_count+icd_10_count)/(total_count+icd_10_count)))
    return converted_dict


def extract_admission_info(data_dict):
    admission_dict = dict()
    success_count, total_count, failure_count_1, failure_count_2 = 0, 0, 0, 0
    average_original_length, average_trimmed_length = [], []
    for unified_id in data_dict:
        total_count += 1
        emr = data_dict[unified_id][7]
        average_original_length.append(len(emr.split(' ')))
        start_idx = emr.find('Sex:')
        if start_idx == -1:
            continue
        emr = emr[start_idx:]
        end_index_1 = emr.find('Discharge')
        end_index_2 = emr.find('DISCHARGE')
        end_index = end_index_1 if end_index_1 > end_index_2 else end_index_2
        if end_index > 0:
            target_emr = emr[:end_index]
            if '\nChief Complaint' in target_emr and '\nHistory of Present Illness' and \
                '\nPast Medical History' in target_emr and '\nSocial History' in target_emr and \
                    'Physical Exam' in target_emr:
                average_trimmed_length.append(len(target_emr.split(' ')))
                admission_dict[unified_id] = target_emr
                success_count += 1
            else:
                failure_count_1 += 1
        else:
            failure_count_2 += 1
    print('admission, total count: {}, success count: {}, success ratio: {}, failure count_1: {}, 2: {}'
          .format(total_count, success_count, success_count / total_count, failure_count_1, failure_count_2))
    print('original length: {}, trimmed length: {}'
          .format(np.average(average_original_length), np.average(average_trimmed_length)))
    return admission_dict


def extract_discharge_info(data_dict, valid_unified_id_set):
    discharge_dict = dict()
    total_count, success_count, failure_count = 0, 0, 0
    average_original_length, average_trimmed_length = [], []
    for unified_id in data_dict:
        if unified_id not in valid_unified_id_set:
            continue
        total_count += 1
        emr = data_dict[unified_id][7]
        average_original_length.append(len(emr.split(' ')))
        end_idx_1 = emr.find('Discharge Medications')
        end_idx_2 = emr.find('Discharge Diagnosis')
        end_idx_3 = emr.find('Discharge Condition')
        end_idx_4 = emr.find('Discharge Disposition')

        end_idx = end_idx_1 if 0 < end_idx_1 < end_idx_2 else end_idx_2
        end_idx = end_idx if 0 < end_idx < end_idx_3 else end_idx_3
        end_idx = end_idx if 0 < end_idx < end_idx_4 else end_idx_4
        if end_idx == -1:
            failure_count += 1
        else:
            discharge_dict[unified_id] = emr[:end_idx]
            average_trimmed_length.append(len(emr[:end_idx].split(' ')))
            success_count += 1
    print('admission, total count: {}, success count: {}, success ratio: {}, failure count: {}'
          .format(total_count, success_count, success_count / total_count, failure_count))
    print('original length: {}, trimmed length: {}'
          .format(np.average(average_original_length), np.average(average_trimmed_length)))
    return discharge_dict


# 1
def main():
    icd_mapping_dict = read_icd_9_10_mapping(icd_mapping_file)
    diagnosis_dict = read_diagnosis(diagnosis_data_path)
    # 这一步的icd转换几乎是无损的（99.75%的数据可以被保留）
    converted_diagnosis_dict = icd_convert(diagnosis_dict, icd_mapping_dict)

    emr_dict = read_emr(discharge_data_path)
    admission_info_dict = extract_admission_info(emr_dict)
    valid_unified_id_set = set(admission_info_dict.keys())
    discharge_info_dict = extract_discharge_info(emr_dict, valid_unified_id_set)

    total_count = 0
    head = ['data_source', 'visit_type', 'patient_visit_id', 'outpatient_record', 'admission_record',
            'comprehensive_history', 'discharge_record', 'first_page', 'discharge_diagnosis',
            'first_page_diagnosis',
            'table_diagnosis', 'icd_code']
    data_to_write = [head]
    for unified_id in converted_diagnosis_dict:
        if unified_id not in admission_info_dict or unified_id not in discharge_info_dict:
            continue
        admission_record = admission_info_dict[unified_id]
        discharge_record = discharge_info_dict[unified_id]
        diagnosis = converted_diagnosis_dict[unified_id]
        icd_code = '$$$$$'.join([item[1] for item in diagnosis])
        line = ['mimic_iv', 'hospitalization', unified_id, 'None', admission_record, 'None', discharge_record, 'None',
                'None', 'None', "None", icd_code]
        data_to_write.append(line)
        total_count += 1
    print('total count: {}'.format(total_count))
    with open(save_file, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)



def read_emr(file_path):
    data_dict = dict()
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            note_id, patient_id, visit_id, note_type, not_seq, chart_time, store_time, text = line
            unified_id = patient_id + '_' + visit_id
            if unified_id not in data_dict:
                data_dict[unified_id] = line
            else:
                print('duplicate')
    return data_dict



if __name__ == "__main__":
    main()
