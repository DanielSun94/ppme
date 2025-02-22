import csv
import json
from srrsh_config import (fused_outpatient_admission_discharge_history_note, full_primary_diagnosis_dataset_path,
                          distinct_diagnosis_path)


# 5
def read_data(file_path):
    data_dict = dict()
    duplicate_count = 0
    with (open(file_path, 'r', encoding='utf-8-sig') as f):
        csv_reader = csv.reader(f)
        for line in csv_reader:
            visit_type, patient_visit_id, outpatient_record, admission_record, discharge_record, history_record = \
                line[0], line[1], line[21], line[37], line[53], line[69]
            table_diagnosis = line[6]
            if visit_type == 'hospitalization':
                assert line[21] == 'None'
            else:
                assert visit_type == 'outpatient'
                assert line[37] == 'None' and line[53] == 'None' and line[69] == 'None'
            if patient_visit_id not in data_dict:
                data_dict[patient_visit_id] = dict()
            if visit_type == 'hospitalization':
                data_dict[patient_visit_id] = {'hospitalization_diagnosis': table_diagnosis}
            else:
                assert visit_type == 'outpatient'
                data_dict[patient_visit_id] = {'outpatient_diagnosis': table_diagnosis}

            if line[21] != "None":
                if 'outpatient_record' in data_dict[patient_visit_id]:
                    duplicate_count += 1
                data_dict[patient_visit_id]['outpatient_record'] = line[21]
            if line[37] != "None":
                if 'admission_record' in data_dict[patient_visit_id]:
                    duplicate_count += 1
                data_dict[patient_visit_id]['admission_record'] = line[37]
            if line[53] != "None":
                if 'discharge_record' in data_dict[patient_visit_id]:
                    duplicate_count += 1
                data_dict[patient_visit_id]['discharge_record'] = line[53]
            if line[69] != "None":
                if 'history_record' in data_dict[patient_visit_id]:
                    duplicate_count += 1
                data_dict[patient_visit_id]['history_record'] = line[69]
    print('duplicate_count:{}'.format(duplicate_count))
    return data_dict


def extract_diagnosis(data_dict):
    success_count, full_count = 0, 0
    discharge_diagnosis_dict = dict()
    for patient_visit_id in data_dict.keys():
        if 'discharge_record' not in data_dict[patient_visit_id]:
            continue
        discharge_note = data_dict[patient_visit_id]['discharge_record']
        success_flag = True
        content = ''
        start_index = discharge_note.find('[出院诊断]')
        if start_index == -1:
            success_flag = False
        if success_flag:
            start_index = start_index + 6
            sub_str = discharge_note[start_index:]
            second_start_index = sub_str.find('[')
            content = sub_str[:second_start_index]
            next_section = sub_str[second_start_index + 1: second_start_index + 5]
            if next_section != '诊治经过' and next_section != '入院诊断':
                success_flag = False
        if success_flag:
            if len(content) > 0:
                diagnosis_str_list = content.split('\n')
                content_list = []
                for item in diagnosis_str_list:
                    item = item.strip(' ').strip('\0').strip('\t')
                    if len(item) > 0:
                        content_list.append(item)
                if len(content_list) > 0:
                    discharge_diagnosis_dict[patient_visit_id] = content_list
                success_count += 1
        full_count += 1

    distinct_diagnosis_set, diagnosis_count = set(), 0
    for key in discharge_diagnosis_dict:
        for diagnosis in discharge_diagnosis_dict[key]:
            distinct_diagnosis_set.add(diagnosis)
            diagnosis_count += 1
    return distinct_diagnosis_set, discharge_diagnosis_dict


def extract_admission(data_dict):
    success_count, full_count = 0, 0
    admission_dict = dict()
    for patient_visit_id in data_dict.keys():
        if not 'admission_record' in data_dict[patient_visit_id]:
            continue
        admission_note = data_dict[patient_visit_id]['admission_record']
        individual_success = 0
        order_info_list = [['病例特点', '诊断依据', '鉴别诊断'], ['诊断依据', '病例特点', '初步诊断']]

        for order_info in order_info_list:
            success_flag = True
            start_index = admission_note.find('[' + order_info[0] + ']')
            if start_index == -1:
                success_flag = False
            if success_flag:
                start_index = start_index + 6
                next_section = admission_note[start_index:]
                second_start_index = next_section.find('[' + order_info[1])
                if second_start_index == -1:
                    success_flag = False

                content_1 = next_section[:second_start_index]
                next_section_title = next_section[second_start_index + 1: second_start_index + 5]
                if next_section_title != order_info[1]:
                    success_flag = False

                next_next_section = next_section[second_start_index + 6:]

                third_start_index = next_next_section.find('[' + order_info[2])
                content_2 = next_next_section[:third_start_index]

                if third_start_index == -1:
                    success_flag = False

                next_next_section_title = next_next_section[third_start_index + 1: third_start_index + 5]
                if next_next_section_title != order_info[2]:
                    success_flag = False

                if success_flag:
                    individual_success += 1
                    assert individual_success == 1
                    content = ('[' + order_info[0] + ']\n' + content_1 + '[' +
                               order_info[1] + ']\n' + content_2)
                    admission_dict[patient_visit_id] = content
        if individual_success == 1:
            success_count += 1
        full_count += 1
    return admission_dict


def extract_history(data_dict):
    # 大病史本身结构化不均一
    # 从结果看，病人本身的信息有些时候时倒着写的（如果有修改），很多其它信息写的也不是很规整，半结构化提取很麻烦且意义不大，拟直接交给LLM
    # 作进一步的筛选没有意义（结构化后不会保留患者隐私信息），而也不像入院病历和出院病历那样会有大量本研究不需要的文本。因此直接全量纳入
    history_dict = dict()
    for patient_visit_id in data_dict.keys():
        if 'history_record' in data_dict[patient_visit_id]:
            history_dict[patient_visit_id] = data_dict[patient_visit_id]['history_record']
    return history_dict


def extract_discharge(data_dict):
    discharge_dict = dict()
    for patient_visit_id in data_dict.keys():
        if 'discharge_record' in data_dict[patient_visit_id]:
            discharge_record = data_dict[patient_visit_id]['discharge_record']
            discharge_dict[patient_visit_id] = discharge_record
    return discharge_dict


def extract_outpatient(data_dict):
    # 不长，全量纳入
    outpatient_dict = dict()
    for patient_visit_id in data_dict.keys():
        if 'outpatient_record' in data_dict[patient_visit_id]:
            outpatient_dict[patient_visit_id] = data_dict[patient_visit_id]['outpatient_record']
    return outpatient_dict


def save_joint_data(discharge_diagnosis_dict, admission_dict, history_dict, outpatient_dict, discharge_dict,
                    data_dict, file_path):
    head = ['admission_type', 'patient_visit_id', 'admission_record', 'history_record', 'outpatient_record',
            'discharge_record', 'diagnosis_list', 'table_diagnosis']
    data_to_write = [head]
    count, hospitalization_count, outpatient_count = 0, 0, 0
    for patient_visit_id in admission_dict.keys():
        if (patient_visit_id not in history_dict.keys() or patient_visit_id not in discharge_diagnosis_dict.keys() or
                patient_visit_id not in discharge_dict):
            continue
        count += 1
        hospitalization_count += 1
        diagnosis = data_dict[patient_visit_id]['hospitalization_diagnosis']

        admission_record = admission_dict[patient_visit_id]
        diagnosis_record = json.dumps(discharge_diagnosis_dict[patient_visit_id])
        history_record = history_dict[patient_visit_id]
        discharge_record = discharge_dict[patient_visit_id]
        data_to_write.append(['hospitalization', patient_visit_id, admission_record, history_record, 'None',
                              discharge_record, diagnosis_record, diagnosis])

    for patient_visit_id in outpatient_dict.keys():
        # if patient_visit_id not in table_diagnosis_dict:
        #     continue
        diagnosis = data_dict[patient_visit_id]['outpatient_diagnosis']
        count += 1
        outpatient_count += 1
        outpatient_record = outpatient_dict[patient_visit_id]
        data_to_write.append(['outpatient', patient_visit_id, 'None', 'None', outpatient_record, "None", 'None',
                              diagnosis])
    with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)
    print('full data set size: {}, outpatient size: {}, hospitalization size: {}'
          .format(count, outpatient_count, hospitalization_count))


def main():
    data_dict = read_data(fused_outpatient_admission_discharge_history_note)
    distinct_diagnosis_set, discharge_diagnosis_dict = extract_diagnosis(data_dict)
    discharge_dict = extract_discharge(data_dict)
    history_dict = extract_history(data_dict)
    admission_dict = extract_admission(data_dict)
    outpatient_dict = extract_outpatient(data_dict)

    save_joint_data(discharge_diagnosis_dict, admission_dict, history_dict, outpatient_dict, discharge_dict, data_dict,
                    full_primary_diagnosis_dataset_path)
    json.dump(list(distinct_diagnosis_set), open(distinct_diagnosis_path, 'w', encoding='utf-8-sig'))


if __name__ == '__main__':
    main()
