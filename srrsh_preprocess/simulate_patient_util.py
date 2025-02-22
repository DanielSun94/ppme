import os
import csv
import pickle
import traceback
from itertools import islice
from datetime import datetime


def read_full_patient_visit_id(source_file, index_folder, save_file, read_from_cache):
    if os.path.exists(save_file) and read_from_cache:
        patient_visit_id_dict = pickle.load(open(save_file, 'rb'))
        return patient_visit_id_dict

    mapping_file_list = []
    for i in range(1, 10):
        mapping_file_list.append(os.path.join(index_folder, f'emr_index_fuse_{i}.csv'))

    time_dict = {}
    for file in mapping_file_list:
        with open(file, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                pk_dcpv = line[1]
                create_time = line[10]
                if len(pk_dcpv) < 4 or len(create_time) < 5:
                    continue

                # 取最早的时间作为Visit时间戳，迭代时要求时间戳晚于2010年，防止错误
                try:
                    if pk_dcpv not in time_dict:
                        time_dict[pk_dcpv] = datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S.%f")
                    else:
                        new_time = datetime.strptime(create_time, "%Y-%m-%d %H:%M:%S.%f")
                        if new_time.year > 2010:
                            previous_time = time_dict[pk_dcpv]
                            if new_time < previous_time:
                                time_dict[pk_dcpv] = new_time
                except Exception as e:
                    print(f"create time: {create_time}")
                    error_message = traceback.format_exc()
                    print(f'error message: {e}, message: {error_message}')

            if len(time_dict) % 10000 == 0 and len(time_dict) > 0:
                print(f'len time dict: {len(time_dict)}')
        print(f'file: {file} success')
    patient_visit_id_dict = dict()
    success_count, failure_count = 0, 0
    with open(source_file, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_visit_id = line[2]
            if patient_visit_id in time_dict:
                time_str = time_dict[patient_visit_id].strftime("%Y-%m-%d")
                patient_visit_id_dict[patient_visit_id] = time_str
                success_count += 1
            else:
                failure_count += 1
    print('success count: {}, failure count: {}'.format(success_count, failure_count))

    pickle.dump(patient_visit_id_dict, open(save_file, 'wb'))
    return patient_visit_id_dict


def read_ord_mapping_dict(source_path, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    code_ord_mapping_dict = dict()
    code_req_mapping_dict = dict()
    code_rep_mapping_dict = dict()
    empty_set = {'', 'null', "None", 'none'}
    with (open(source_path, 'r', encoding='utf-8-sig') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_dcpv, pvcode, code_ord, code_rep, code_req, name_sex, birthday, name_orditem, desc_ord, \
                create_time, date_create = line
            if len(pk_dcpv) < 2:
                continue

            if code_ord not in empty_set and code_ord is not None:
                code_ord_mapping_dict[code_ord] = pk_dcpv
            if code_rep not in empty_set and code_rep is not None:
                code_rep_mapping_dict[code_rep] = pk_dcpv
            if code_req not in empty_set and code_req is not None:
                code_req_mapping_dict[code_req] = pk_dcpv
    save_dict = {
        'ord': code_ord_mapping_dict,
        'rep': code_rep_mapping_dict,
        'req': code_req_mapping_dict
    }
    pickle.dump(save_dict, open(save_path, 'wb'))
    return save_dict

