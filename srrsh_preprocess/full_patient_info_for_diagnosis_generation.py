import pickle
import csv
import os
import random
from itertools import islice
from datetime import datetime
from simulate_patient_util import read_full_patient_visit_id
from exam_interate_test import read_exam_prescription
from labtest_exam_fuse import read_ord_rec
from srrsh_config import (
    final_fusion_path, original_extraction, valid_patient_visit_id_file, ord_rec_path, lt_ord_pacs_record_path,
    exam_mapping_cache_file, valid_pat_lab_rep_code_dict_file, ord_detail_fuse_path, exam_data_cache_file,
    lab_test_data_cache_file, ord_pacs_record_path, ord_rec_folder, age_sex_cache_file, differential_diagnosis_file,
    differential_diagnosis_sample_file, valid_pat_exam_rep_code_dict_file
)


# 10 带检验和检查的数据生成
def read_lab_test(detail_path, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    data_dict = dict()
    with open(detail_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            source_pk, pk_rep_lis, code_rep, create_time, name, value, name_quanti_unit, desc_rrs = line
            if code_rep not in data_dict:
                data_dict[code_rep] = [create_time, name, value, name_quanti_unit, desc_rrs]
    pickle.dump(data_dict, open(save_path, 'wb'))
    return data_dict


def read_exam_data(path_1, path_2, save_path, read_from_cache):
    if read_from_cache and os.path.exists(save_path):
        return pickle.load(open(save_path, 'rb'))

    data_dict = dict()
    with open(path_2, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_ord_record, result_obj, eu_result, result_subj, note = line
            data_dict[pk_ord_record] = [result_obj, eu_result, result_subj, note]
    with open(path_1, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_pacs, pk_ord_record, name_diag, result_obj, result_subj, eu_result, note, source_pk, create_time = line
            data_dict[pk_ord_record] = [result_obj, eu_result, result_subj, note]

    pickle.dump(data_dict, open(save_path, 'wb'))
    return data_dict


def get_pk_sex_age(ord_rec_folder_path, ord_rec_cache, age_sex_cache, read_from_cache):
    if os.path.exists(age_sex_cache) and read_from_cache:
        return pickle.load(open(age_sex_cache, 'rb'))

    data_list = read_ord_rec(ord_rec_folder_path, ord_rec_cache, True)
    print('read data success')
    age_sex_dict = dict()

    sex_birthday_miss_count = 0
    sex_illegal_count = 0
    sex_illegal_unknown_count = 0
    birthday_illegal_count = 0
    success_count = 0
    count = 0
    for line in data_list[1:]:
        pk_dcpv = line[2]
        code_sex = line[4].strip()
        birthday = line[5].replace('上', '').strip()
        count += 1

        if len(code_sex) == 0 or len(birthday) == 0 or code_sex == 'null' or birthday == 'null':
            sex_birthday_miss_count += 1
            continue
        if code_sex != 'F' and code_sex != "M":
            sex_illegal_count += 1
            if code_sex == '未知':
                sex_illegal_unknown_count += 1
            continue
        try:
            if '-' in birthday:
                if len(birthday) < 12:
                    birthday_date = datetime.strptime(birthday, "%Y-%m-%d")
                else:
                    birthday_date = datetime.strptime(birthday, "%Y-%m-%d %H:%M:%S")
            else:
                if len(birthday) < 12:
                    birthday_date = datetime.strptime(birthday, "%Y/%m/%d")
                else:
                    birthday_date = datetime.strptime(birthday, "%Y/%m/%d %H:%M:%S")
        except:
            birthday_illegal_count += 1
            continue

        success_count += 1
        if pk_dcpv not in age_sex_dict:
            sex = "女" if code_sex == 'F' else "男"
            age_sex_dict[pk_dcpv] = [sex, birthday_date]

    print(f'len age sex dict: {len(age_sex_dict)}, \n'
          f'all count: {count}, \n'
          f'success count: {success_count}, \n'
          f'missing_count: {sex_birthday_miss_count}, \n'
          f'sex_illegal_count: {sex_illegal_count}, sex_illegal_unknown_count: {sex_illegal_unknown_count} \n'
          f'birthday_illegal_count: {birthday_illegal_count}')
    pickle.dump(age_sex_dict, open(age_sex_cache, 'wb'))
    return age_sex_dict


def read_emr(source_file):
    data_dict = dict()
    with (open(source_file, 'r', encoding='utf-8-sig', newline='') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            pk_dcpv = line[2]
            data_dict[pk_dcpv] = line
    emr_key_list = ['data_source', 'visit_type', 'patient_visit_id', 'outpatient_record', 'admission_record',
                    'comprehensive_history', 'discharge_record', 'first_page', 'discharge_diagnosis',
                    'first_page_diagnosis', 'table_diagnosis', 'icd_code']
    return data_dict, emr_key_list


def convert_first_lab_to_str(code_rep_list, lab_data_dict):
    convert_lab = []
    for code in code_rep_list:
        if code not in lab_data_dict:
            continue
        else:
            lab_info = lab_data_dict[code]
            lab_time = datetime.strptime(lab_info[0], "%Y-%m-%d %H:%M:%S.%f")
            convert_lab.append([lab_time] + lab_info)
    convert_lab = sorted(convert_lab, key=lambda x: x[0])

    # 按发生时间读取，只取第一次
    reserve_list = []
    hit_name_set = set()
    for item in convert_lab:
        name = item[2]
        if name in hit_name_set:
            continue
        else:
            hit_name_set.add(name)
            reserve_list.append(item)

    lab_str = ''
    for item in reserve_list:
        if item[3] == '' or item[3] == 'null':
            continue
        lab_str += f'检验项目：{item[2]}，检验结果: {item[3]}，单位: {item[4]}，正常值范围：{item[5]}，检验时间 {item[1]}\n'
    return lab_str


def convert_first_exam_to_str(exam_rep_list, exam_data_dict, exam_mapping_dict):
    convert_exam = []
    time_placeholder = '2024-12-31'
    for code in exam_rep_list:
        if code not in exam_mapping_dict or code not in exam_data_dict:
            continue
        else:
            observation, impression = exam_data_dict[code][:2]
            exam_time, (exam_type, exam_sub_name) = exam_mapping_dict[code][3], exam_mapping_dict[code][5:7]
            try:
                exam_time_obj = datetime.strptime(exam_time, "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                exam_time_obj = datetime.strptime(time_placeholder, "%Y-%m-%d")
                if exam_time != 'null':
                    print(f'exam time illegal: {exam_time}')

            if observation == 'null':
                observation = ''
            if impression == 'null':
                impression = ''
            if impression == '' and observation == '':
                continue
            convert_exam.append([exam_time_obj, exam_time, exam_type, exam_sub_name, observation, impression])
    convert_exam = sorted(convert_exam, key=lambda x: x[0])

    # 按发生时间读取，只取第一次
    reserve_list = []
    hit_name_set = set()
    for item in convert_exam:
        exam_type, exam_sub_name = item[2:4]
        key = exam_type + '_' + exam_sub_name
        if key in hit_name_set:
            continue
        else:
            hit_name_set.add(key)
            reserve_list.append(item)

    exam_str = ''
    for item in reserve_list:
        key = str(item[2]) + "_" + str(item[3])
        exam_time = item[1]
        if exam_time == time_placeholder:
            exam_time = '未记录'
        exam_str += f'检查项目：{key}。观察所见: {item[4]}。结论: {item[5]}，检验时间 {exam_time} \n\n\n\n'
    return exam_str


def data_fuse(emr_dict, emr_key_list, age_sex_dict, valid_patient_visit_id_dict, patient_lab_id_dict,
              patient_exam_id_dict, exam_mapping_dict, lab_data_dict, exam_data_dict, save_file, sample_save_file,
              random_seed=715):
    head = emr_key_list + ['affiliated_info']
    data_to_write = []

    age_sex_success_count, hospitalization_age_sex_count, outpatient_age_sex_count, count = 0, 0, 0, 0
    lab_hit_count, exam_hit_count, lab_his_hit_count, lab_out_hit_count, exam_out_hit_count, exam_his_hit_count = \
        0, 0, 0, 0, 0, 0

    for key in emr_dict:
        count += 1
        line = emr_dict[key]
        assert len(line) == 12
        pk_dcpv = line[2]
        visit_data = [] + line
        affiliated_info = ''
        if pk_dcpv in age_sex_dict and pk_dcpv in valid_patient_visit_id_dict:
            sex = age_sex_dict[pk_dcpv][0]
            birthday = age_sex_dict[pk_dcpv][1]
            admit_date = datetime.strptime(valid_patient_visit_id_dict[pk_dcpv], "%Y-%m-%d")
            age = int((admit_date - birthday).days / 365)
            affiliated_info += '个人信息：年龄： {}, 性别：{}。\n\n'.format(age, sex)
            age_sex_success_count += 1

            if line[1] == 'hospitalization':
                hospitalization_age_sex_count += 1
            else:
                assert line[1] == 'outpatient'
                outpatient_age_sex_count += 1
        else:
            affiliated_info += ''

        if pk_dcpv in patient_lab_id_dict and 'rep' in patient_lab_id_dict[pk_dcpv]:
            code_rep_list = patient_lab_id_dict[pk_dcpv]['rep']
            affiliated_info += '实验室检查信息：\n{}\n\n'.format(convert_first_lab_to_str(code_rep_list, lab_data_dict))
            lab_hit_count += 1
            if line[1] == 'hospitalization':
                lab_his_hit_count += 1
            else:
                assert line[1] == 'outpatient'
                lab_out_hit_count += 1
        else:
            affiliated_info += ''

        if pk_dcpv in patient_exam_id_dict:
            exam_rep_list = patient_exam_id_dict[pk_dcpv]
            affiliated_info += ('影像学及心电、脑电等检查信息：\n{}\n\n'
                                .format(convert_first_exam_to_str(exam_rep_list, exam_data_dict, exam_mapping_dict)))
            exam_hit_count += 1
            if line[1] == 'hospitalization':
                exam_his_hit_count += 1
            else:
                assert line[1] == 'outpatient'
                exam_out_hit_count += 1
        else:
            affiliated_info += ''
        visit_data = visit_data + [affiliated_info]
        data_to_write.append(visit_data)

    print(f'data count: {count}')
    print(f'age sex success count: {age_sex_success_count}, age_sex_his: {hospitalization_age_sex_count}, '
          f'age_sex_out: {outpatient_age_sex_count}')
    print(f'lab_hit_count count: {lab_hit_count}, lab_his_hit_count: {lab_his_hit_count}, '
          f'lab_out_hit_count: {lab_out_hit_count}')
    print(f'exam_hit_count count: {exam_hit_count}, exam_his_hit_count: {exam_his_hit_count}, '
          f'exam_out_hit_count: {exam_out_hit_count}')

    random_idx_list = [i for i in range(len(data_to_write))]
    random.Random(random_seed).shuffle(random_idx_list)

    new_data_to_write = [head]
    for idx in random_idx_list:
        new_data_to_write.append(data_to_write[idx])
    with open(save_file, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(new_data_to_write)

    with open(sample_save_file, 'w', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(new_data_to_write[:5000])
    print('success')


# 注意，final_disease_screening_dataset_generation生成的数据集可以直接使用
# 该数据集中的patient_visit_id就是pk_dcpv；
# 注意，这个数据集没有筛过高风险疾病数据和做过ICD编码（相关的工作是在joint data generation上做的）
def main():
    age_sex_dict = get_pk_sex_age(ord_rec_folder, ord_rec_path, age_sex_cache_file, True)
    print('age sex loaded')
    valid_patient_visit_id_dict = read_full_patient_visit_id(final_fusion_path, original_extraction,
                                                             valid_patient_visit_id_file, True)
    print('valid_patient_visit_id_dict loaded')
    _, patient_lab_id_dict = pickle.load(open(valid_pat_lab_rep_code_dict_file, 'rb'))
    print('patient_lab_id_dict loaded')
    _, patient_exam_id_dict = pickle.load(open(valid_pat_exam_rep_code_dict_file, 'rb'))
    print('patient_exam_id_dict loaded')
    exam_mapping_dict = read_exam_prescription(ord_rec_path, exam_mapping_cache_file, True)
    print('exam mapping dict loaded')
    lab_data_dict = read_lab_test(ord_detail_fuse_path, lab_test_data_cache_file, True)
    print('lab_data_dict loaded')
    exam_data_dict = read_exam_data(
        ord_pacs_record_path, lt_ord_pacs_record_path, exam_data_cache_file, True)
    print('exam_data_dict loaded')
    emr_dict, emr_key_list = read_emr(final_fusion_path)
    print('emr loaded')
    data_fuse(emr_dict, emr_key_list, age_sex_dict, valid_patient_visit_id_dict, patient_lab_id_dict,
              patient_exam_id_dict, exam_mapping_dict, lab_data_dict, exam_data_dict, differential_diagnosis_file,
              differential_diagnosis_sample_file)
    print('final success')


if __name__ == '__main__':
    main()
