import os
import json
import csv
from srrsh_config import (clinical_note_folder_1, fused_clinical_note_path_template, clinical_note_folder_2)

# 2
head = ['pk_sec', 'create_time', 'edit_time', 'in_date', 'name_sec', 'pk_dcemr', 'data_source',
        'flag_del', 'code_psn', 'code_group', 'code_sec', 'code_org', 'source_pk', 'code_dept', 'secfs',
        'name_dept']


def fuse_clinical_note(folder_1, folder_2, filter_set=None, filtered_name=None, read_from_cache=True):
    if filter_set is not None:
        file_name = fused_clinical_note_path_template.format(filtered_name)
    else:
        file_name = fused_clinical_note_path_template.format('all')

    if not (os.path.exists(file_name) and read_from_cache):
        data_2 = fuse_clinical_note_2(folder_2, filter_set)
        data_1 = fuse_clinical_note_1(folder_1, filter_set)
        data_to_write = [head] + data_1 + data_2
        print('len data: {}'.format(len(data_to_write)))
        with open(file_name, 'w', encoding='utf-8-sig') as f:
            csv.writer(f).writerows(data_to_write)


def fuse_clinical_note_2(folder, filter_set=None):
    print('start fuse_clinical_note_2')
    data_to_write, file_list, name_sec_dict = [], [], {}

    sub_folder_list = os.listdir(folder)
    for sub_folder in sub_folder_list:
        sub_folder_path = os.path.join(folder, sub_folder)
        for file_name in os.listdir(sub_folder_path):
            file_list.append([sub_folder_path, file_name])

    failure_count, success_count = 0, 0
    for (sub_folder_path, file_name) in file_list:
        file_path = os.path.join(sub_folder_path, file_name)
        # if success_count > 10000: # for test
        #     break
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except Exception as e:
                    print('Exception: {}'.format(e))
                    continue

                if filter_set is not None:
                    if 'name_sec' not in data or data['name_sec'] is None:
                        continue
                    if data['name_sec'] not in name_sec_dict:
                        name_sec_dict[data['name_sec']] = 0
                    name_sec_dict[data['name_sec']] += 1
                    if data['name_sec'] not in filter_set:
                        continue
                try:
                    line = []
                    for key in head:
                        line.append(data[key])
                    data_to_write.append(line)
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    print('Failed Count: {}, Error: {}'.format(failure_count, e))
                if success_count % 50000 == 0:
                    print('success Count: {}, failed count: {}'.format(success_count, failure_count))
    name_sec_list = []
    for key in name_sec_dict:
        name_sec_list.append([key, name_sec_dict[key]])
    name_sec_list = sorted(name_sec_list, key=lambda x: x[1], reverse=True)
    for item in name_sec_list:
        if item[1] > 100000:
            print('emr info: ITEM: {}, COUNT: {}'.format(item[0], item[1]))
    print('success count: {}, fail count: {}'.format(success_count, failure_count))
    print('end fuse_clinical_note_2')
    return data_to_write


def fuse_clinical_note_1(folder, filter_set=None):
    print('start fuse_clinical_note_1')
    data_to_write = []
    file_list = os.listdir(folder)
    failure_count, success_count = 0, 0
    for file in file_list:
        file_path = os.path.join(folder, file)
        # if success_count > 10000: # for test
        #     break
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data_str = ''.join(f.readlines())
            data_str = '[' + data_str[1:-1] + ']'
            try:
                data = json.loads(data_str)
            except Exception as e:
                print('Exception: {}'.format(e))
                continue

            for item in data:
                if filter_set is not None:
                    if 'name_sec' not in item or item['name_sec'] not in filter_set:
                        continue
                try:
                    line = []
                    for key in head:
                        line.append(item[key])
                    data_to_write.append(line)
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    print('Failed Count: {}, Error: {}'.format(failure_count, e))
                if success_count % 50000 == 0:
                    print('Success Count: {}'.format(success_count))
    print('success count: {}, fail count: {}'.format(success_count, failure_count))
    print('end fuse_clinical_note_1')
    return data_to_write



def main():

    #
    # filtered_name = 'comprehensive_admission_info'
    # filter_set = {'大病史(女-新)', '大病史(男-新)', '大病史（产科）', '新生儿监护室大病史', '大病史(女)', '大病史(男)'}
    # fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filter_set, filtered_name, read_from_cache=True)
    #
    # filtered_name = 'first_page'
    # filter_set = {'住院病案首页（病人存档及卫生统计用）'}
    # fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filter_set, filtered_name, read_from_cache=True)
    #
    # filtered_name = 'discharge_relevant'
    # filter_set = {'出院记录'}
    # fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filter_set, filtered_name, read_from_cache=True)
    #
    # filtered_name = 'admission_relevant'
    # filter_set = {'首次病程录', '首次病程录（新版）'}
    # fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filter_set, filtered_name, read_from_cache=True)
    #
    # filtered_name = 'outpatient_relevant'
    # filter_set = {'急诊初诊病历', '门诊初诊病历', '急诊初诊病历+', '门诊初诊病历+'}
    # fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filter_set, filtered_name, read_from_cache=True)

    filter_set = {"门诊复诊病历+", '门诊初诊病历+', '门诊复诊病历', '门诊初诊病历', '急诊初诊病历', '急诊复诊病历'}
    filtered_name = 'outpatient_relevant_2'
    fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filter_set, filtered_name, read_from_cache=True)

    # filtered_name = 'all'
    # fuse_clinical_note(clinical_note_folder_1, clinical_note_folder_2, filtered_name, read_from_cache=False)


if __name__ == '__main__':
    main()
