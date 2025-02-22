import os
import csv
import pickle
from itertools import islice
from simulate_patient_util import read_full_patient_visit_id, read_ord_mapping_dict
from srrsh_config import (ord_detail_fuse_path, original_extraction, final_fusion_path, valid_patient_visit_id_file,
                          lab_test_rep_cache_file, ord_prescription_path, ord_prescription_mapping_file,
                          valid_pat_lab_rep_code_dict_file)

# 8
# 实验室检查完整性检查
def read_lab_test_code_rep(detail_path, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        code_rep_dict = pickle.load(open(save_path, 'rb'))
        print('len code rep set: {}'.format(len(code_rep_dict)))
        return code_rep_dict

    code_rep_dict = dict()
    count = 0
    with open(detail_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            source_pk, pk_rep_lis, code_rep, create_time, name, value, name_quanti_unit, desc_rrs = line
            if code_rep not in code_rep_dict:
                code_rep_dict[code_rep] = [0, create_time]
            code_rep_dict[code_rep][0] += 1
            count += 1
    pickle.dump(code_rep_dict, open(save_path, 'wb'))
    # 注意，这里code rep是正常的，因为一个处方就是对应了很多的检验
    print('len code rep set: {}, count: {}'.format(len(code_rep_dict), count))
    return code_rep_dict


def integrate_test(valid_patient_visit_id_dict, lab_test_code_rep_dict, ord_mapping_dict, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    success_rep_count = 0
    item_success_mapping_count = 0
    second_rep_count = 0
    second_item_success_count = 0
    req_mapping_success, ord_mapping_success = 0, 0

    success_code_rep_set = set()
    patient_lab_id_dict = dict()

    for code_rep in lab_test_code_rep_dict:
        if code_rep in ord_mapping_dict['rep']:
            pk_dcpv = ord_mapping_dict['rep'][code_rep].strip()
            if pk_dcpv in valid_patient_visit_id_dict:
                success_rep_count += 1
                item_success_mapping_count += lab_test_code_rep_dict[code_rep][0]
                second_rep_count += 1
                second_item_success_count += lab_test_code_rep_dict[code_rep][0]
                success_code_rep_set.add(code_rep)

                if pk_dcpv not in patient_lab_id_dict:
                    patient_lab_id_dict[pk_dcpv] = {}
                if 'rep' not in patient_lab_id_dict[pk_dcpv]:
                    patient_lab_id_dict[pk_dcpv]['rep'] = []
                patient_lab_id_dict[pk_dcpv]['rep'].append(code_rep)

        elif code_rep in ord_mapping_dict['req']:
            pk_dcpv = ord_mapping_dict['req'][code_rep].strip()
            if pk_dcpv in valid_patient_visit_id_dict:
                second_rep_count += 1
                req_mapping_success += 1
                second_item_success_count += lab_test_code_rep_dict[code_rep][0]

                if pk_dcpv not in patient_lab_id_dict:
                    patient_lab_id_dict[pk_dcpv] = {}
                if 'req' not in patient_lab_id_dict[pk_dcpv]:
                    patient_lab_id_dict[pk_dcpv]['req'] = []
                patient_lab_id_dict[pk_dcpv]['req'].append(code_rep)

        elif code_rep in ord_mapping_dict['ord']:
            pk_dcpv = ord_mapping_dict['ord'][code_rep].strip()
            if pk_dcpv in valid_patient_visit_id_dict:
                second_rep_count += 1
                ord_mapping_success += 1
                second_item_success_count += lab_test_code_rep_dict[code_rep][0]

                if pk_dcpv not in patient_lab_id_dict:
                    patient_lab_id_dict[pk_dcpv] = {}
                if 'ord' not in patient_lab_id_dict[pk_dcpv]:
                    patient_lab_id_dict[pk_dcpv]['ord'] = []
                patient_lab_id_dict[pk_dcpv]['ord'].append(code_rep)

    print(f'len full lab code rep set: {len(lab_test_code_rep_dict)}, '
          f'success rep count: {success_rep_count}, item_success_mapping_count: {item_success_mapping_count} '
          f'second_rep_count: {second_rep_count}, second_item_success_count: {second_item_success_count} '
          f'req_mapping_success： {req_mapping_success}, ord_mapping_success: {ord_mapping_success}')

    pickle.dump([success_code_rep_set, patient_lab_id_dict], open(valid_pat_lab_rep_code_dict_file, 'wb'))
    return success_code_rep_set, patient_lab_id_dict


def time_test(success_id_set, lab_test_code_rep_dict):
    year_count_dict = dict()
    for success_id in lab_test_code_rep_dict:
        time_str = lab_test_code_rep_dict[success_id][1]
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
    year_count_dict = dict()
    for key in valid_patient_visit_id_dict:
        year = valid_patient_visit_id_dict[key][:4]
        if year not in year_count_dict:
            year_count_dict[year] = 0
        year_count_dict[year] += 1
    print('len valid_patient_visit_id_dict: {}'.format(len(valid_patient_visit_id_dict)))
    for year in year_count_dict:
        print('year: {}, count: {}'.format(year, year_count_dict[year]))

    print('valid patient visit is set success')
    ord_mapping_dict = read_ord_mapping_dict(
        ord_prescription_path, ord_prescription_mapping_file, True)
    lab_test_code_rep_dict = read_lab_test_code_rep(ord_detail_fuse_path, lab_test_rep_cache_file, True)
    print('mapping dict loaded')
    success_code_rep_set, patient_lab_id_dict = (
        integrate_test(valid_patient_visit_id_dict, lab_test_code_rep_dict, ord_mapping_dict,
                       valid_pat_lab_rep_code_dict_file, True))
    time_test(success_code_rep_set, lab_test_code_rep_dict)


if __name__ == '__main__':
    main()
