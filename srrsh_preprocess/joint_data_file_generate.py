import json
import os
import csv
import re
import pickle
from datetime import datetime
from srrsh_config import (emr_index_fuse_file_template, emr_index_fuse_cache, fused_diagnosis_file_path_template,
                          diagnosis_fuse_cache_template, fused_clinical_note_path_template,
                          clinical_note_cache_template,
                          fused_joint_admission_file_template, fused_outpatient_admission_discharge_history_note)


# 4
def read_emr_index(read_from_cache=True):
    if read_from_cache and os.path.exists(emr_index_fuse_cache):
        mapping_dict = pickle.load(open(emr_index_fuse_cache, 'rb'))
    else:
        mapping_dict = dict()
        idx_column_dict = dict()
        for i in range(1, 10):
            file_name = emr_index_fuse_file_template.format(i)
            print('file parsing start: {}'.format(file_name))
            with open(file_name, 'r', encoding='utf-8-sig') as file:
                csv_reader = csv.reader(file)
                index = 0
                for line in csv_reader:
                    if index == 0:
                        for idx, col in enumerate(line):
                            idx_column_dict[idx] = col
                        index += 1
                        continue

                    pk_dcemr, pk_dcpv = line[0: 2]
                    pk_dcemr = pk_dcemr.replace(' ', '').strip()
                    pk_dcpv = pk_dcpv.replace(' ', '').strip()
                    if pk_dcemr in mapping_dict and pk_dcpv != mapping_dict[pk_dcemr]:
                        print('Duplicate! new pk_dcpv: {}, origin pk_dcpv'.format(pk_dcpv, mapping_dict[pk_dcemr]))
                    mapping_dict[pk_dcemr] = pk_dcpv
                    if len(mapping_dict) % 1000000 == 0:
                        print('mapping dict length: {}'.format(len(mapping_dict)))
            print('file success: {}'.format(file_name))
        print('final mapping dict length: {}'.format(len(mapping_dict)))
        pickle.dump(mapping_dict, open(emr_index_fuse_cache, 'wb'))
    return mapping_dict


def read_diagnosis(file, diagnosis_type, diagnosis_use_icd, read_from_cache=True):
    # 注意，在file中已经分别存储了用ICD和不用ICD的两个版本，这里只是再过滤一遍
    assert diagnosis_type == 'discharge' or diagnosis_type == 'primary'
    success_count = 0
    diagnosis_fuse_cache = diagnosis_fuse_cache_template.format(diagnosis_type, diagnosis_use_icd)
    if read_from_cache and os.path.exists(diagnosis_fuse_cache):
        data_dict = pickle.load(open(diagnosis_fuse_cache, 'rb'))
    else:
        data_dict = dict()
        idx_column_dict = dict()
        with open(file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            index = 0
            for line in csv_reader:
                if index == 0:
                    for idx, col in enumerate(line):
                        idx_column_dict[idx] = col
                    index += 1
                    continue
                pk_dcpv, pk_dcpvdiag, diagnosis, code_diagnosis, name_diagtype = \
                    line[0], line[1], line[7], line[6], line[4]
                pk_dcpv = pk_dcpv.replace(' ', '').strip()
                if diagnosis_type == 'discharge':
                    if '主诊断' not in name_diagtype and '明确诊断' not in diagnosis_type and '补充诊断' \
                            not in name_diagtype and '鉴别诊断' not in name_diagtype:
                        continue
                if pk_dcpv not in data_dict:
                    data_dict[pk_dcpv] = []
                data_dict[pk_dcpv].append([pk_dcpvdiag, diagnosis, code_diagnosis, name_diagtype])
        for key in data_dict:
            success_count += len(data_dict[key])
        print('diagnosis read success')
        print('diagnosis data dict length: {}'.format(len(data_dict)))
        print('diagnosis avg element number: {}'.format(success_count / len(data_dict)))
        pickle.dump(data_dict, open(diagnosis_fuse_cache, 'wb'))
    return data_dict


def read_emr(filter_name, read_from_cache=True):
    file_path = fused_clinical_note_path_template.format(filter_name)
    cache_path = clinical_note_cache_template.format(filter_name)
    duplicate_count = 0
    date_format = '%Y-%m-%d %H:%M:%S'
    if read_from_cache and os.path.exists(cache_path):
        data_dict = pickle.load(open(cache_path, 'rb'))
    else:
        data_dict = dict()
        idx_column_dict = dict()
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            index = 0
            for line in csv_reader:
                if index == 0:
                    for idx, col in enumerate(line):
                        idx_column_dict[idx] = col
                    index += 1
                    continue
                pk_dcemr, name_sec, pk_sec, in_date = line[5], line[4], line[0], line[3]
                in_date = in_date.strip().split('.')[0]
                pk_dcemr = pk_dcemr.replace(' ', '').strip()
                if pk_dcemr not in data_dict:
                    data_dict[pk_dcemr] = line
                else:
                    previous_in_date = data_dict[pk_dcemr][3]
                    previous_in_date = datetime.strptime(previous_in_date, date_format)
                    current_time = datetime.strptime(in_date, date_format)
                    duplicate_count += 1
                    if previous_in_date < current_time:
                        data_dict[pk_dcemr] = line

        pickle.dump(data_dict, open(cache_path, 'wb'))
        print('emr: {} read success, len emr: {}, duplicate: {}'.format(filter_name, len(data_dict), duplicate_count))
    return data_dict


def joint_mapping_analysis(clinical_note_dict, diagnosis_dict, mapping_dict, filter_name):
    dcemr_set = set()

    patient_visit_dict = dict()
    for key in diagnosis_dict:
        patient_id, visit_id = key.strip().split('_')
        if patient_id not in patient_visit_dict:
            patient_visit_dict[patient_id] = set()
        patient_visit_dict[patient_id].add(key)
    patient_doesnt_exist_set = set()

    data_to_to_write = [['pk_dcemr', 'name_sec', 'text']]
    total_emr_count = len(clinical_note_dict)
    success_count = 0
    for pk_dcemr in clinical_note_dict:
        name_sec, text = clinical_note_dict[pk_dcemr][4], clinical_note_dict[pk_dcemr][14]
        if pk_dcemr not in mapping_dict:
            continue
        dcemr_set.add(pk_dcemr)
        pk_dcpv = mapping_dict[pk_dcemr]
        if pk_dcpv in diagnosis_dict:
            data_to_to_write.append([pk_dcemr, name_sec, text])
            success_count += 1
        else:
            patient_id = pk_dcpv.strip().split('_')[0]
            if patient_id not in patient_visit_dict:
                patient_doesnt_exist_set.add(patient_id)
            else:
                _ = patient_visit_dict[patient_id]
                # print('')
    print('len patient_doesnt_exist_set: {}'.format(len(patient_doesnt_exist_set)))
    print('success count: {}, total count: {}, fraction: {}'
          .format(success_count, total_emr_count, success_count / total_emr_count))
    print('len dcemr_set: {}'.format(len(dcemr_set)))
    with open(fused_joint_admission_file_template.format(filter_name), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_to_write)


def admission_discharge_mapping(save_path):
    filter_name = 'discharge_relevant'
    discharge_dict = read_emr(filter_name, read_from_cache=True)
    filter_name = 'admission_relevant'
    admission_dict = read_emr(filter_name, read_from_cache=True)
    mapping_dict = read_emr_index(read_from_cache=True)

    discharge_mapping_dict = dict()
    admission_mapping_dict = dict()
    for pk_dcemr in discharge_dict:
        if pk_dcemr not in mapping_dict:
            continue
        discharge_pk_dcpv = mapping_dict[pk_dcemr]
        discharge_mapping_dict[pk_dcemr] = discharge_pk_dcpv

    for pk_dcemr in admission_dict:
        if pk_dcemr not in mapping_dict:
            continue
        admission_pk_dcpv = mapping_dict[pk_dcemr]
        admission_mapping_dict[admission_pk_dcpv] = pk_dcemr

    data_to_write = []
    success_count, total_count = 0, len(discharge_dict)
    for discharge_pk_dcemr in discharge_mapping_dict:
        pk_dcpv = discharge_mapping_dict[discharge_pk_dcemr]
        if pk_dcpv in admission_mapping_dict:
            success_count += 1
            admission_pk_dcemr = admission_mapping_dict[pk_dcpv]
            admission_line = admission_dict[admission_pk_dcemr]
            discharge_line = discharge_dict[discharge_pk_dcemr]
            data_to_write.append([pk_dcpv, admission_pk_dcemr, discharge_pk_dcemr] + admission_line + discharge_line)

    with open(save_path, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_to_write)
    print('success count: {}, total count: {}, fraction: {}'
          .format(success_count, total_count, success_count / total_count))


def admission_discharge_comprehensive_mapping(save_path):
    filter_name = 'discharge_relevant'
    discharge_dict = read_emr(filter_name, read_from_cache=True)
    filter_name = 'admission_relevant'
    admission_dict = read_emr(filter_name, read_from_cache=True)
    filter_name = 'comprehensive_admission_info'
    comprehensive_history_dict = read_emr(filter_name, read_from_cache=True)
    mapping_dict = read_emr_index(read_from_cache=True)

    discharge_mapping_dict = dict()
    admission_mapping_dict = dict()
    comprehensive_mapping_dict = dict()
    for pk_dcemr in discharge_dict:
        if pk_dcemr not in mapping_dict:
            continue
        discharge_pk_dcpv = mapping_dict[pk_dcemr]
        discharge_mapping_dict[pk_dcemr] = discharge_pk_dcpv

    for pk_dcemr in admission_dict:
        if pk_dcemr not in mapping_dict:
            continue
        admission_pk_dcpv = mapping_dict[pk_dcemr]
        admission_mapping_dict[admission_pk_dcpv] = pk_dcemr

    for pk_dcemr in comprehensive_history_dict:
        if pk_dcemr not in mapping_dict:
            continue
        comprehensive_pk_dcpv = mapping_dict[pk_dcemr]
        comprehensive_mapping_dict[comprehensive_pk_dcpv] = pk_dcemr

    data_to_write = []
    success_count, total_count = 0, len(discharge_dict)
    for discharge_pk_dcemr in discharge_mapping_dict:
        pk_dcpv = discharge_mapping_dict[discharge_pk_dcemr]
        if pk_dcpv in admission_mapping_dict and pk_dcpv in comprehensive_mapping_dict:
            success_count += 1
            admission_pk_dcemr = admission_mapping_dict[pk_dcpv]
            comprehensive_pk_dcemr = comprehensive_mapping_dict[pk_dcpv]

            admission_line = admission_dict[admission_pk_dcemr]
            discharge_line = discharge_dict[discharge_pk_dcemr]
            comprehensive_line = comprehensive_history_dict[comprehensive_pk_dcemr]
            data_to_write.append([pk_dcpv, admission_pk_dcemr, discharge_pk_dcemr, comprehensive_pk_dcemr]
                                 + admission_line + discharge_line + comprehensive_line)

    with open(save_path, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_to_write)
    print('success count: {}, total count: {}, fraction: {}'
          .format(success_count, total_count, success_count / total_count))


def diagnosis_secondary_process(diagnosis_list, admission_type):
    pattern = r'^[A-Z][0-9]|^[0-9]'
    diagnosis_dict = dict()
    assert admission_type == 'hospitalization' or admission_type == 'outpatient'
    if admission_type == 'outpatient':
        for item in diagnosis_list:
            key, diagnosis, code, diagnosis_type = item
            is_code = re.match(pattern, code)
            if diagnosis_type == '初步诊断':
                if key not in diagnosis_dict:
                    diagnosis_dict[key] = diagnosis, code if is_code is not None else 'None'
                else:
                    if diagnosis != diagnosis_dict[key][0]:
                        print(f'diagnosis inconsistency: origin: {diagnosis_dict[key][0]}, new: {diagnosis}')
    else:
        for item in diagnosis_list:
            key, diagnosis, code, diagnosis_type = item
            is_code = re.match(pattern, code)
            if diagnosis_type == '主诊断(出院)' and is_code is not None:
                if key not in diagnosis_dict:
                    diagnosis_dict[key] = diagnosis, code
                else:
                    if diagnosis != diagnosis_dict[key][0]:
                        print(f'diagnosis inconsistency: origin: {diagnosis_dict[key][0]}, new: {diagnosis}')
    return diagnosis_dict


def outpatient_admission_discharge_comprehensive_mapping(save_path):
    # 注意，因为大部分门诊病历都只有入院初诊诊断，因此此处的diagnosis type必须设为primary，不然会造成大量初诊丢失的问题
    # 这份srrsh病历中似乎没有首页病史明显诊断质量更高的问题，因此就不纳入病案首页了（纳入会造成5万份病历不能完整匹配丢失）
    # 由于数据本身的特点，我们对表里的诊断的目前利用如下：如果是门诊，则只要记了就列入；如果是住院，则只记录有ICD的
    # 因为住院数据的出院病历里有完整的诊断，没有ICD是没有意义的；而门诊因为没有类似的系统，且病历里通常不写诊断，因此有没有ICD都列入
    # 注意，这里的icd筛选是在diagnosis_secondary_process处理的，一开始的use icd false和diagnosis type设为primary是正确的
    diagnosis_type = 'primary'
    diagnosis_use_icd = False
    filter_name = 'outpatient_relevant'
    outpatient_dict = read_emr(filter_name, read_from_cache=True)
    print('outpatient_dict load success')
    diagnosis_path = fused_diagnosis_file_path_template.format(diagnosis_use_icd)
    diagnosis_dict = read_diagnosis(diagnosis_path, diagnosis_type, diagnosis_use_icd, read_from_cache=True)
    print('diagnosis_dict load success')
    filter_name = 'discharge_relevant'
    discharge_dict = read_emr(filter_name, read_from_cache=True)
    print('discharge_dict load success')
    filter_name = 'admission_relevant'
    admission_dict = read_emr(filter_name, read_from_cache=True)
    print('admission_dict load success')
    filter_name = 'comprehensive_admission_info'
    comprehensive_history_dict = read_emr(filter_name, read_from_cache=True)
    print('comprehensive_history_dict load success')
    mapping_dict = read_emr_index(read_from_cache=True)
    print('mapping_dict load success')

    discharge_mapping_dict, admission_mapping_dict, comprehensive_mapping_dict, outpatient_mapping_1_dict = (
        {}, {}, {}, {})

    for pk_dcemr in outpatient_dict:
        if pk_dcemr not in mapping_dict:
            continue
        outpatient_pk_dcpv = mapping_dict[pk_dcemr]
        if outpatient_pk_dcpv not in outpatient_mapping_1_dict:
            outpatient_mapping_1_dict[outpatient_pk_dcpv] = []
        outpatient_mapping_1_dict[outpatient_pk_dcpv].append(pk_dcemr)

    outpatient_mapping_dict = dict()
    for outpatient_pk_dcpv in outpatient_mapping_1_dict:
        outpatient_pk_dcemr, _ = select_outpatient(outpatient_dict, outpatient_mapping_1_dict, outpatient_pk_dcpv)
        outpatient_mapping_dict[outpatient_pk_dcpv] = outpatient_pk_dcemr

    for pk_dcemr in discharge_dict:
        if pk_dcemr not in mapping_dict:
            continue
        discharge_pk_dcpv = mapping_dict[pk_dcemr]
        discharge_mapping_dict[pk_dcemr] = discharge_pk_dcpv

    for pk_dcemr in admission_dict:
        if pk_dcemr not in mapping_dict:
            continue
        admission_pk_dcpv = mapping_dict[pk_dcemr]
        admission_mapping_dict[admission_pk_dcpv] = pk_dcemr

    for pk_dcemr in comprehensive_history_dict:
        if pk_dcemr not in mapping_dict:
            continue
        comprehensive_pk_dcpv = mapping_dict[pk_dcemr]
        comprehensive_mapping_dict[comprehensive_pk_dcpv] = pk_dcemr

    print('start mapping')
    data_to_write = []
    success_count, hospitalization_success, outpatient_success, total_count = 0, 0, 0, len(discharge_dict)
    for outpatient_pk_dcpv in outpatient_mapping_dict:
        outpatient_pk_dcemr = outpatient_mapping_dict[outpatient_pk_dcpv]
        if outpatient_pk_dcpv in diagnosis_dict:
            success_count += 1
            outpatient_success += 1
            diagnosis = diagnosis_secondary_process(diagnosis_dict[outpatient_pk_dcpv], 'outpatient')
            diagnosis = json.dumps(diagnosis)
            outpatient_sample = outpatient_dict[outpatient_pk_dcemr]
            assert len(outpatient_sample) == 16
            data_to_write.append(
                ['outpatient', outpatient_pk_dcpv, outpatient_pk_dcemr, 'None', 'None', 'None', diagnosis] +
                outpatient_sample + ['None'] * 48
            )

    discharge_diagnosis_exist_count, discharge_diagnosis_count = 0, 0
    for discharge_pk_dcemr in discharge_mapping_dict:
        pk_dcpv = discharge_mapping_dict[discharge_pk_dcemr]
        if pk_dcpv in admission_mapping_dict and pk_dcpv in comprehensive_mapping_dict:
            success_count += 1
            hospitalization_success += 1
            admission_pk_dcemr = admission_mapping_dict[pk_dcpv]
            comprehensive_pk_dcemr = comprehensive_mapping_dict[pk_dcpv]

            admission_line = admission_dict[admission_pk_dcemr]
            discharge_line = discharge_dict[discharge_pk_dcemr]
            comprehensive_line = comprehensive_history_dict[comprehensive_pk_dcemr]

            if pk_dcpv in diagnosis_dict:
                diagnosis = diagnosis_secondary_process(diagnosis_dict[pk_dcpv], 'hospitalization')
                diagnosis_length = len(diagnosis)
                # if diagnosis_length > 4:
                #     print(diagnosis)
                diagnosis = json.dumps(diagnosis) if len(diagnosis) > 0 else 'None'
                if diagnosis != 'None':
                    discharge_diagnosis_exist_count += 1
                    discharge_diagnosis_count += diagnosis_length
            else:
                diagnosis = 'None'
            assert len(admission_line) == 16
            assert len(discharge_line) == 16
            assert len(comprehensive_line) == 16
            data_to_write.append(['hospitalization', pk_dcpv, 'None', admission_pk_dcemr, discharge_pk_dcemr,
                                  comprehensive_pk_dcemr, diagnosis] + ['None'] * 16 + admission_line + discharge_line +
                                 comprehensive_line)
    print(
        f'average discharge diagnosis (with icd) count: {discharge_diagnosis_count / discharge_diagnosis_exist_count}')
    with open(save_path, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(data_to_write)
    print('success count: {}, hospitalization success: {}, outpatient success: {}, total count: {}, fraction: {}'
          .format(success_count, hospitalization_success, outpatient_success, total_count,
                  hospitalization_success / total_count))


def select_outpatient(outpatient_data_dict, outpatient_mapping_dict, pk_dcpv):
    date_format = '%Y-%m-%d %H:%M:%S'
    outpatient_pk_dcemr_list = outpatient_mapping_dict[pk_dcpv]
    outpatient_pk_dcemr, date_time, line = None, None, None
    for key in outpatient_pk_dcemr_list:
        new_line = outpatient_data_dict[key]
        in_date = new_line[3].strip().split('.')[0]

        new_date_time = datetime.strptime(in_date, date_format)
        if date_time is None:
            date_time = new_date_time
            outpatient_pk_dcemr = key
            line = new_line
        else:
            if new_date_time < date_time:
                date_time = new_date_time
                outpatient_pk_dcemr = key
                line = new_line
    return outpatient_pk_dcemr, line


# def joint_mapping_test():
#     diagnosis_type = 'primary'
#     diagnosis_use_icd = False
#     filter_name = 'admission_relevant'
#     print('diagnosis_type: {}'.format(diagnosis_type))
#     print('diagnosis_use_icd: {}'.format(diagnosis_use_icd))
#     print('filter_name: {}'.format(filter_name))
#
#     diagnosis_path = fused_diagnosis_file_path_template.format(diagnosis_use_icd)
#     diagnosis_dict = read_diagnosis(diagnosis_path, diagnosis_type, diagnosis_use_icd, read_from_cache=True)
#     print('load diagnosis dict success')
#     clinical_note_dict = read_emr(filter_name, read_from_cache=True)
#     print('load clinical note dict success')
#     mapping_dict = read_emr_index(read_from_cache=True)
#     print('load mapping dict success')
#     joint_mapping_analysis(clinical_note_dict, diagnosis_dict, mapping_dict, filter_name)
#     print('success')


def main():
    # admission_discharge_mapping(fused_admission_discharge_note)
    # admission_discharge_comprehensive_mapping(fused_admission_discharge_history_note)
    outpatient_admission_discharge_comprehensive_mapping(fused_outpatient_admission_discharge_history_note)
    # joint_mapping_test()


if __name__ == '__main__':
    main()
