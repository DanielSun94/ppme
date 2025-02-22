import os
import csv
import pickle
from itertools import islice
from datetime import datetime
from simulate_patient_util import read_full_patient_visit_id, read_ord_mapping_dict
from srrsh_config import (original_extraction, final_fusion_path, valid_patient_visit_id_file,
                          ord_pacs_record_path, lt_ord_pacs_record_path, ord_rec_path, exam_mapping_cache_file,
                          reserved_mapping_exam_file, ord_prescription_path, ord_prescription_mapping_file,
                          valid_pat_exam_rep_code_dict_file)


# 9
# 影像学检查完整性检查
def read_exam_prescription(path, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    mapping_dict = dict()
    illegal_pk_dcpv_set = set()
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_ord_record, pk_dcord, pk_dcpv, pvcode, code_sex, birthday, name_eu_item, name_part, date_ris, \
                date_rep, code_rep, code_req, code_ord = line
            if pk_dcpv.strip() != pvcode.strip():
                print(f'pk_dcpv: {pk_dcpv} != pvcode: {pvcode}')

            if pk_ord_record not in mapping_dict:
                mapping_dict[pk_ord_record] = [pk_dcpv, code_rep, code_req, date_rep, code_ord, name_eu_item, name_part]
            else:
                print('duplicate pk_ord_record')
            if pk_dcpv[-1] == '_':
                illegal_pk_dcpv_set.add(pk_dcpv)
    print(f'len illegal pk_dcpv_set: {len(illegal_pk_dcpv_set)}')

    pickle.dump(mapping_dict, open(save_path, 'wb'))
    return mapping_dict


def read_exam_result(path_1, path_2, mapping_dict, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    new_mapping_dict = dict()
    file_1_count, file_1_success, file_2_count, file_2_success = 0, 0, 0, 0
    with open(path_2, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_ord_record, result_obj, eu_result, result_subj, note = line
            file_1_count += 1
            if pk_ord_record in mapping_dict:
                new_mapping_dict[pk_ord_record] = mapping_dict[pk_ord_record]
                file_1_success += 1
    with open(path_1, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_pacs, pk_ord_record, name_diag, result_obj, result_subj, eu_result, note, source_pk, create_time = line
            file_2_count += 1
            if pk_ord_record in mapping_dict:
                new_mapping_dict[pk_ord_record] = mapping_dict[pk_ord_record]
                file_2_success += 1
    print(f'file_1_count: {file_1_count}, file_1_success: {file_1_success}, file_2_count: {file_2_count}, '
          f'file_2_success: {file_2_success}')

    pickle.dump(new_mapping_dict, open(save_path, 'wb'))
    return new_mapping_dict


def get_patient_visit_date_dict(valid_patient_visit_id_dict):
    patient_visit_date_dict = dict()
    for key in valid_patient_visit_id_dict:
        patient_id = key.strip().split("_")[0]
        if patient_id not in patient_visit_date_dict:
            patient_visit_date_dict[patient_id] = []
        visit_date = valid_patient_visit_id_dict[key]
        patient_visit_date_dict[patient_id].append(datetime.strptime(visit_date, "%Y-%m-%d"))
    for key in patient_visit_date_dict:
        patient_visit_date_dict[key] = sorted(patient_visit_date_dict[key])
    return patient_visit_date_dict


def integrate_test(valid_patient_visit_id_dict, exam_code_mapping_dict, ord_mapping_dict, save_path, read_from_cache):
    if os.path.exists(save_path) and read_from_cache:
        return pickle.load(open(save_path, 'rb'))

    patient_exam_id_dict = dict()
    # exam的integrate test遵循如下规律。
    # exam_code_mapping_dict中针对每个pk_ord_record都有pk_dcpv, code_rep, code_req, date_rep四个结果
    # 按照pk_dcpv, code_rep, code_req的顺序进行匹配。能匹配到哪个就算哪个
    failure_count = 0
    success_count = 0
    pk_dcpv_success = 0
    rep_success_count = 0
    req_success_count = 0
    ord_success_count = 0
    success_id_set = set()
    empty_set = {'null', "", "None", 'none'}
    for pk_ord_record in exam_code_mapping_dict:
        hit_flag = False
        pk_dcpv, code_rep, code_req, date_rep, code_ord = exam_code_mapping_dict[pk_ord_record][:5]
        if pk_dcpv in valid_patient_visit_id_dict and pk_dcpv not in empty_set and pk_dcpv is not None:
            pk_dcpv_success += 1
            hit_flag = True

            if pk_dcpv not in patient_exam_id_dict:
                patient_exam_id_dict[pk_dcpv] = []
            patient_exam_id_dict[pk_dcpv].append(pk_ord_record)

        if ((not hit_flag) and code_rep in ord_mapping_dict['rep'] and code_rep not in empty_set
                and code_rep is not None):
            ord_mapping_pk_dcpv = ord_mapping_dict['rep'][code_rep]
            if ord_mapping_pk_dcpv in valid_patient_visit_id_dict:
                rep_success_count += 1
                hit_flag = True

                if pk_dcpv not in patient_exam_id_dict:
                    patient_exam_id_dict[pk_dcpv] = []
                patient_exam_id_dict[pk_dcpv].append(pk_ord_record)

        if ((not hit_flag) and code_req in ord_mapping_dict['req'] and code_req not in empty_set
                and code_req is not None):
            ord_mapping_pk_dcpv = ord_mapping_dict['req'][code_req]
            if ord_mapping_pk_dcpv in valid_patient_visit_id_dict:
                req_success_count += 1
                hit_flag = True

                if pk_dcpv not in patient_exam_id_dict:
                    patient_exam_id_dict[pk_dcpv] = []
                patient_exam_id_dict[pk_dcpv].append(pk_ord_record)

        if ((not hit_flag) and code_ord in ord_mapping_dict['ord'] and code_ord not in empty_set
                and code_ord is not None):
            ord_mapping_pk_dcpv = ord_mapping_dict['ord'][code_ord]
            if ord_mapping_pk_dcpv in valid_patient_visit_id_dict:
                ord_success_count += 1
                hit_flag = True

                if pk_dcpv not in patient_exam_id_dict:
                    patient_exam_id_dict[pk_dcpv] = []
                patient_exam_id_dict[pk_dcpv].append(pk_ord_record)
        if hit_flag:
            success_count += 1
            success_id_set.add(pk_ord_record)
        else:
            failure_count += 1
    general_count = success_count + failure_count
    print(f'general_count: {general_count}, success_count: {success_count}, failure_count: {failure_count}, '
          f'pk_dcpv_success: {pk_dcpv_success}, rep_success_count: {rep_success_count}, '
          f'req_success_count: {req_success_count}, ord_success_count: {ord_success_count}')

    pickle.dump([success_id_set, patient_exam_id_dict], open(save_path, 'wb'))
    return success_id_set, patient_exam_id_dict


def time_test(success_id_set, exam_mapping_dict):
    year_count_dict = dict()
    for success_id in exam_mapping_dict:
        time_str = exam_mapping_dict[success_id][3]
        year = time_str[:4]
        if year not in year_count_dict:
            year_count_dict[year] = [0, 0]

        year_count_dict[year][0] += 1
        if success_id in success_id_set:
            year_count_dict[year][1] += 1

    data = []
    for key in year_count_dict:
        data.append([key, year_count_dict[key][0], year_count_dict[key][1],
                     year_count_dict[key][1] / year_count_dict[key][0]])
    data = sorted(data, key=lambda x: x[2], reverse=True)
    for line in data:
        year, full_count, hit_count, ratio = line
        print(f'year: {year}, hit count: {hit_count}, full count: {full_count}, ratio: {ratio}')
    return year_count_dict


def main():
    valid_patient_visit_id_dict = read_full_patient_visit_id(final_fusion_path, original_extraction,
                                                             valid_patient_visit_id_file, True)

    exam_mapping_dict = read_exam_prescription(ord_rec_path, exam_mapping_cache_file, True)
    print(f'load_success, len_mapping_dict: {len(exam_mapping_dict)}')
    # 能够映射到报告的pk_ord_record的集合
    reserved_mapping_dict = read_exam_result(ord_pacs_record_path, lt_ord_pacs_record_path, exam_mapping_dict,
                                             reserved_mapping_exam_file, True)
    print('len reserved mapping dict: {}'.format(len(reserved_mapping_dict)))
    ord_mapping_dict = read_ord_mapping_dict(
        ord_prescription_path, ord_prescription_mapping_file, True)
    success_id_set, patient_exam_id_dict = integrate_test(
        valid_patient_visit_id_dict, reserved_mapping_dict, ord_mapping_dict, valid_pat_exam_rep_code_dict_file,
        True)
    time_test(success_id_set, exam_mapping_dict)


if __name__ == '__main__':
    main()
