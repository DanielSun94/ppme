import csv
import os
import pickle
import numpy as np
from util import parse_data_symptom
from disease_screen_logger import logger
from itertools import islice
from torch.utils.data import Dataset
from disease_screen_config import (structurize_symptom_cache_template, data_file_path,
                                   diagnosis_cache_template, history_text_embedding_folder, symptom_file_path_template,
                                   symptom_info_dict, symptom_num_path)


def get_symptom_index_map(path_list):
    # 注意，path list行idx一样的Line语义是一样的，这一点经过了人工的校核
    symptom_combine_dict = dict()
    for info in path_list:
        file_path, language = info
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for idx, line in enumerate(islice(csv_reader, 1, None)):
                symptom, factor_group, factor = line
                if idx not in symptom_combine_dict:
                    symptom_combine_dict[idx] = [language, symptom, factor_group, factor]
                else:
                    symptom_combine_dict[idx] += [language, symptom, factor_group, factor]

    index_symptom_dict, symptom_index_dict = dict(), dict()
    general_idx = 0
    for idx in symptom_combine_dict:
        new_symptom_flag = False
        split_num = len(symptom_combine_dict[idx]) // 4
        for i in range(split_num):
            language, symptom, factor_group, factor = symptom_combine_dict[idx][i*4: (i+1)*4]
            symptom = symptom.strip().lower()
            if language not in symptom_index_dict:
                symptom_index_dict[language] = dict()
            if symptom not in symptom_index_dict[language]:
                new_symptom_flag = True
                symptom_index_dict[language][symptom] = general_idx, -1
        if new_symptom_flag:
            general_idx += 1

    for idx in symptom_combine_dict:
        split_num = len(symptom_combine_dict[idx]) // 4
        triplet_flag = False
        for i in range(split_num):
            language, symptom, factor_group, factor = symptom_combine_dict[idx][i * 4: (i + 1) * 4]
            symptom, factor_group, factor = symptom.strip(), factor_group.strip(), factor.strip()

            parent_idx = symptom_index_dict[language][symptom.strip().lower()][0]
            key = ' '.join([symptom, factor_group, factor]).strip().replace('  ', ' ').lower()
            if len(factor_group) > 0:
                assert len(factor) > 0
                assert key not in symptom_index_dict[language]
                triplet_flag = True
                symptom_index_dict[language][key] = general_idx, parent_idx
        if triplet_flag:
            general_idx += 1

    for language in symptom_index_dict:
        for key in symptom_index_dict[language]:
            idx, _ = symptom_index_dict[language][key]
            if language not in index_symptom_dict:
                index_symptom_dict[language] = dict()
            index_symptom_dict[language][idx] = key
    return index_symptom_dict, symptom_index_dict


def structurize_symptom(file_path, symptom_index_dict, language):
    symptom_array = np.zeros(len(symptom_index_dict[language]) * 3)
    positive_count = 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for idx, line in enumerate(islice(csv_reader, 1, None)):
            symptom, factor_group, factor, state = line
            if factor_group == 'N/A':
                key = symptom.lower()
            else:
                key = (symptom + ' ' + factor_group + ' ' + factor).replace('  ', ' ').lower()
            mapping_idx = symptom_index_dict[language][key][0]
            if state == 'NA':
                symptom_array[mapping_idx*3] = 1
                symptom_array[mapping_idx*3+1] = 0
                symptom_array[mapping_idx*3+2] = 0
            elif state == 'YES':
                symptom_array[mapping_idx*3] = 0
                symptom_array[mapping_idx*3+1] = 0
                symptom_array[mapping_idx*3+2] = 1
                positive_count += 1
            else:
                assert state == 'NO'
                symptom_array[mapping_idx*3] = 0
                symptom_array[mapping_idx*3+1] = 1
                symptom_array[mapping_idx*3+2] = 0
    assert np.sum(symptom_array) == len(symptom_index_dict[language])
    return symptom_array, positive_count


def read_symptom_files(file_path, symptom_index_dict, language):
    folder_name = file_path.split('/')[-2]
    file_name = folder_name + '-' + os.path.basename(file_path)[:-12]
    symptom_array, positive_count = structurize_symptom(file_path, symptom_index_dict, language)
    return [file_name, symptom_array, positive_count]


def read_symptom(info_dict, filter_key, symptom_index_dict, read_from_cache=True):
    if os.path.exists(structurize_symptom_cache_template.format(filter_key)) and read_from_cache:
        data_list = pickle.load(open(structurize_symptom_cache_template.format(filter_key), 'rb'))
        return data_list

    symptom_info_list = info_dict[filter_key]
    # 目前只有中文，后续再有英文的时候这里要加英文的适配和中英文的映射约束，确保每个index的语义是一样的
    key_dict = {}
    file_path_list = []
    for info in symptom_info_list:
        folder_path, language, key = info
        key_dict[key] = {'success': 0, 'failure': 0}

        sub_folders = os.listdir(folder_path)
        count = 0
        for sub_folder in sub_folders:
            file_list = os.listdir(os.path.join(folder_path, sub_folder))
            for file_name in file_list:
                if 'symptom' not in file_name:
                    continue
                count += 1
                file_path = os.path.join(folder_path, sub_folder, file_name)
                file_path_list.append([file_path, language])
        logger.info(f'{folder_path} count: {count}')
    logger.info('total file size: {}'.format(len(file_path_list)))

    data_list, failed_count = [], 0
    for (file_path, language) in file_path_list:
        modify_time = os.path.getmtime(file_path)
        file_name, symptom_array, positive_count = read_symptom_files(file_path, symptom_index_dict, language)
        if positive_count > 0:
            data_list.append([file_name, symptom_array, modify_time])
            for key in key_dict:
                if key in file_name:
                    key_dict[key]['success'] += 1
        else:
            for key in key_dict:
                if key in file_name:
                    key_dict[key]['failure'] += 1
            failed_count += 1

        if (len(data_list) + failed_count) % 10000 == 0:
            logger.info('Processed {} symptom files'.format(len(data_list) + failed_count))
    logger.info('No positive files: {}'.format(failed_count))
    # 此处的logger代表了真的有有效症状的患者
    for key in key_dict:
        for status in key_dict[key]:
            count = key_dict[key][status]
            logger.info(f'{key}, {status}: {count}')
    pickle.dump(data_list, open(structurize_symptom_cache_template.format(filter_key), 'wb'))
    return data_list


def read_diagnosis(file_path, symptom_key_set, filter_key, lower_threshold, digit=3, top_n=1, strategy="ALL",
                   weight=False, read_from_cache=True):
    if strategy != "ALL":
        logger.info('Note, for the filter strategy you chosen, we may filter a large faction of data')
    # 这里最早的策略是可以任意选择要top n的诊断，也可以决定是否要为诊断列表赋予不同的权重
    # 现在只取Top 1，并且取消权重设计
    assert top_n == 1 and not weight
    diagnosis_cache = diagnosis_cache_template.format(digit, top_n, weight, strategy, lower_threshold, filter_key)
    if os.path.exists(diagnosis_cache) and read_from_cache:
        structurized_dict, diagnosis_index_map, index_diagnosis_map = pickle.load(open(diagnosis_cache, 'rb'))
        return structurized_dict, diagnosis_index_map, index_diagnosis_map

    diagnosis_dict = dict()
    diagnosis_count, included_count = 0, 0
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            data_source, visit_type, patient_visit_id = line[0: 3]
            oracle_diagnosis_list = line[16].strip().split('$$$$$')
            unified_id = data_source + '-' + visit_type + '-' + patient_visit_id

            if unified_id not in symptom_key_set:
                continue

            included_diagnosis_list = []
            oracle_diagnosis_list = oracle_diagnosis_list[:top_n]
            for oracle_diagnosis in oracle_diagnosis_list:
                icd = oracle_diagnosis[:digit].lower()
                if strategy == 'ALL':
                    included_diagnosis_list.append(icd)
                else:
                    assert strategy == 'DISCARD_OL'
                    if icd[0] not in {'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}:
                        included_diagnosis_list.append(icd)
                    else:
                        continue
            diagnosis_dict[unified_id] = included_diagnosis_list

    diagnosis_index_map, index_diagnosis_map = dict(), dict()
    diagnosis_count_map = dict()
    diagnosis_reserve_set = set()
    for unified_id in diagnosis_dict:
        included_diagnosis_list = diagnosis_dict[unified_id]
        for icd_code in included_diagnosis_list:
            if icd_code not in diagnosis_count_map:
                diagnosis_count_map[icd_code] = 0
            diagnosis_count_map[icd_code] += 1
            diagnosis_count += 1
    for icd_code in diagnosis_count_map:
        if diagnosis_count_map[icd_code] >= lower_threshold:
            diagnosis_reserve_set.add(icd_code)
            idx = len(diagnosis_index_map)
            diagnosis_index_map[icd_code] = idx
            index_diagnosis_map[idx] = icd_code
            included_count += diagnosis_count_map[icd_code]

    logger.info(f'diagnosis count: {diagnosis_count}, included_count: {included_count}, '
                f'ratio: {included_count/diagnosis_count}')
    structurized_dict = dict()
    for unified_id in diagnosis_dict:
        diagnosis_list = diagnosis_dict[unified_id]
        structurized_diagnosis = np.zeros(len(diagnosis_index_map))
        for i, diagnosis in enumerate(diagnosis_list):
            icd = diagnosis[:digit].lower()
            if icd not in diagnosis_index_map:
                continue
            if weight:
                structurized_diagnosis[diagnosis_index_map[icd]] = -i
            else:
                structurized_diagnosis[diagnosis_index_map[icd]] = 1
        if np.sum(structurized_diagnosis) > 0:
            structurized_dict[unified_id] = structurized_diagnosis
    pickle.dump([structurized_dict, diagnosis_index_map, index_diagnosis_map], open(diagnosis_cache, 'wb'))
    return structurized_dict, diagnosis_index_map, index_diagnosis_map


def fuse_symptom_diagnosis_embedding(diagnosis_dict, symptom_list, embedding_dict, use_symptom, max_size):
    data_list = []
    for item in symptom_list:
        unified_id_full, symptom, symptom_modify_time = item
        unified_id = '-'.join(unified_id_full.split("-")[1:])

        if unified_id not in diagnosis_dict or unified_id_full not in embedding_dict:
            continue
        diagnosis = diagnosis_dict[unified_id]
        embedding = embedding_dict[unified_id_full]
        if not use_symptom:
            symptom = np.array([0])
        sample = [unified_id_full, symptom, diagnosis, embedding, symptom_modify_time]
        data_list.append(sample)

        if 0 < max_size < len(data_list):
            break

    data_list = sorted(data_list, key=lambda x: x[0])
    return data_list


def get_fraction_part(data_list, split_strategy):
    # 所谓custom切分方法，指使用文件夹index进行切分，先排test, 然后是valid和train，确保baseline的实验可以在数据没跑完时开始
    # 其中，根据每个数据集体量的不同切分方法不同
    train_set, valid_set, test_set = set(), set(), set()
    assert split_strategy == 'custom'
    for item in data_list:
        unified_id = item[0]
        folder_idx = int(unified_id[:5])
        if 'srrsh-hospitalization' in unified_id:
            if folder_idx < 45:
                test_set.add(unified_id)
            elif folder_idx < 46:
                valid_set.add(unified_id)
            else:
                train_set.add(unified_id)
        elif 'srrsh-outpatient' in unified_id:
            if folder_idx < 175:
                test_set.add(unified_id)
            elif folder_idx < 176:
                valid_set.add(unified_id)
            else:
                train_set.add(unified_id)
        elif 'mimic_iv' in unified_id:
            if folder_idx < 20:
                test_set.add(unified_id)
            elif folder_idx < 21:
                valid_set.add(unified_id)
            else:
                train_set.add(unified_id)
        else:
            assert 'mimic_iii' in unified_id
            if folder_idx < 8:
                test_set.add(unified_id)
            elif folder_idx < 9:
                valid_set.add(unified_id)
            else:
                train_set.add(unified_id)
    return train_set, valid_set, test_set


def reorganize(data_list, split_strategy, external_key_set):
    # 注意，因为在structurize的时候已经随机化过一遍，因此，可以默认每个文件夹中包含的数据已经是随机化过的
    # 在实践中，使用文件夹的编号决定测试集（这样把test测试集放前面就可以尽可能早的确定测试集范围开始跑测试）
    train_set, valid_set, test_set = get_fraction_part(data_list, split_strategy)
    valid_dict = {'id': [], 'symptom': [], 'embedding': [], 'diagnosis': [], 'time': []}
    test_dict = {'id': [], 'symptom': [], 'embedding': [], 'diagnosis': [], 'time': []}
    train_dict = {'id': [], 'symptom': [], 'embedding': [], 'diagnosis': [], 'time': []}

    for sample in data_list:
        unified_id, symptom, diagnosis, embedding, modify_time = sample
        if external_key_set is not None:
            if unified_id not in external_key_set:
                continue
        if unified_id in train_set:
            train_dict['id'].append(unified_id)
            train_dict['symptom'].append(symptom)
            train_dict['diagnosis'].append(diagnosis)
            train_dict['embedding'].append(embedding)
            train_dict['time'].append(modify_time)
        elif unified_id in valid_set:
            valid_dict['id'].append(unified_id)
            valid_dict['symptom'].append(symptom)
            valid_dict['diagnosis'].append(diagnosis)
            valid_dict['embedding'].append(embedding)
            valid_dict['time'].append(modify_time)
        else:
            assert unified_id in test_set
            test_dict['id'].append(unified_id)
            test_dict['symptom'].append(symptom)
            test_dict['diagnosis'].append(diagnosis)
            test_dict['embedding'].append(embedding)
            test_dict['time'].append(modify_time)

    train_dataset = DiagnosisDataset(train_dict['id'], train_dict['symptom'], train_dict['diagnosis'],
                                     train_dict['embedding'], train_dict['time'])
    valid_dataset = DiagnosisDataset(valid_dict['id'], valid_dict['symptom'], valid_dict['diagnosis'],
                                     valid_dict['embedding'], valid_dict['time'])
    test_dataset = DiagnosisDataset(test_dict['id'], test_dict['symptom'], test_dict['diagnosis'],
                                    test_dict['embedding'], test_dict['time'])
    return train_dataset, valid_dataset, test_dataset


def read_embedding(id_list, embedding_type, embedding_size):
    # embedding type 0, 1, 2分别代表不使用embedding，使用不包含hpi的embedding和使用hpi的embedding
    assert embedding_type in {0, 1, 2}
    # 准备在后期加上，现在是个占位符
    embedding_dict = dict()

    file_list = os.listdir(history_text_embedding_folder)
    for file in file_list:
        file_path = os.path.join(history_text_embedding_folder, file)
        data = pickle.load(open(file_path, 'rb'))
        for item in data:
            # 注意，此处的item 1是with hpi的embedding, item 2是without hpi的embedding
            if embedding_type == 1:
                embedding_dict[item[0]] = item[2]
            elif embedding_type == 2:
                embedding_dict[item[0]] = item[1]
            else:
                assert embedding_type == 0

    data_dict = dict()
    success_count, failure_count = 0, 0
    for unified_id in id_list:
        mapping_id = '-'.join(unified_id.strip().split('-')[1:])
        if mapping_id in embedding_dict:
            if embedding_type > 0:
                data_dict[unified_id] = embedding_dict[mapping_id]
            else:
                data_dict[unified_id] = np.zeros(embedding_size)
            success_count += 1
        else:
            if embedding_type == 0:
                data_dict[unified_id] = np.zeros(embedding_size)
            failure_count += 1
    logger.info(f'embedding hit, success: {success_count}, failure_count: {failure_count}')
    return data_dict


def stat_diagnosis(data, index_diagnosis_map):
    count_dict = dict()
    for item in data:
        disease_idx = np.argmax(item[2])
        if disease_idx not in count_dict:
            count_dict[disease_idx] = 0
        count_dict[disease_idx] += 1

    data_list = []
    for key in count_dict:
        data_list.append([key, count_dict[key], index_diagnosis_map[key]])
    data_list = sorted(data_list, key=lambda x: x[1], reverse=True)
    for item in data_list:
        logger.info('{}: {}'.format(item[2], item[1]))


def read_diagnosis_symptom_embedding(
        diagnosis_path, symptom_folder_dict, symptom_index_dict, top_n, digit, weight, diagnosis_lower, strategy,
        embedding_size, filter_key, data_split_strategy, read_from_cache, embedding_type, use_symptom,
        external_key_set=None):
    assert strategy == 'ALL' or strategy == 'DISCARD_OL'

    symptom_list = read_symptom(symptom_folder_dict, filter_key, symptom_index_dict, read_from_cache=read_from_cache)
    logger.info(f'symptom loaded, list length: {len(symptom_list)}')

    symptom_key_set = set(['-'.join(item[0].split("-")[1:]) for item in symptom_list])
    structurized_diagnosis_dict, diagnosis_index_map, index_diagnosis_map = (
        read_diagnosis(diagnosis_path, symptom_key_set=symptom_key_set, strategy=strategy, digit=digit, top_n=top_n,
                       weight=weight, read_from_cache=False, lower_threshold=diagnosis_lower,
                       filter_key=filter_key))
    logger.info(f'diagnosis loaded, dict length: {len(structurized_diagnosis_dict)}')

    embedding_dict = read_embedding([item[0] for item in symptom_list], embedding_type, embedding_size)

    data_list = fuse_symptom_diagnosis_embedding(
        structurized_diagnosis_dict,
        symptom_list,
        embedding_dict,
        use_symptom,
        max_size=-1
    )
    logger.info(f'data list length: {len(data_list)}')
    stat_diagnosis(data_list, index_diagnosis_map)
    # 因为是最后reorganize，因此可以直接滤掉symptom为0和diagnosis不符合要求的，只在符合要求的数据中开展分析
    train_dataset, valid_dataset, test_dataset = reorganize(data_list, data_split_strategy, external_key_set)
    return train_dataset, valid_dataset, test_dataset, diagnosis_index_map, index_diagnosis_map


class DiagnosisDataset(Dataset):
    def __init__(self, key, symptom, diagnosis, embedding, time):
        self.key = key
        self.symptom = symptom
        self.embedding = embedding
        self.diagnosis = diagnosis
        self.time = time

    def __len__(self):
        return len(self.key)

    def __getitem__(self, idx):
        key = self.key[idx]
        symptom = self.symptom[idx]
        diagnosis = self.diagnosis[idx]
        embedding = self.embedding[idx]
        time = self.time[idx]
        return key, symptom, diagnosis, embedding, time


def dataset_origin_stat(train_dataset, valid_dataset, test_dataset, key_set):
    origin_dict = dict()
    for dataset in train_dataset, valid_dataset, test_dataset:
        for item in dataset:
            unified_id, symptom, diagnosis, embedding, _ = item
            for key in key_set:
                if key in unified_id:
                    if key not in origin_dict:
                        origin_dict[key] = 0
                    origin_dict[key] += 1

    key_count_list = [[key, origin_dict[key]]for key in origin_dict]
    key_count_list = sorted(key_count_list, key=lambda x: x[1], reverse=True)

    for item in key_count_list:
        logger.info(f'name: {item[0]}, count: {item[1]}')


def disease_stat(train_dataset, valid_dataset, test_dataset, index_diagnosis_map):
    disease_count_map = dict()
    for dataset in train_dataset, valid_dataset, test_dataset:
        for item in dataset:
            key, symptom, diagnosis, embedding, _ = item
            index_list = np.where(diagnosis == 1)
            for index in index_list[0]:
                index = int(index)
                disease_name = index_diagnosis_map[index]
                if disease_name not in disease_count_map:
                    disease_count_map[disease_name] = 0
                disease_count_map[disease_name] += 1
    disease_name_count_list = [[key, disease_count_map[key]]for key in disease_count_map]
    disease_name_count_list = sorted(disease_name_count_list, key=lambda x: x[1], reverse=True)

    count_1, count_2, count_3 = 0, 0, 0
    for item in disease_name_count_list:
        logger.info(f'name: {item[0]}, count: {item[1]}')
        count_1 += item[1]
        if item[0][0] not in {'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}:
            count_2 += item[1]
        else:
            count_3 += item[1]
    logger.info(f'count_1: {count_1}, count_2: {count_2}, count_3: {count_3}')


def main():
    logger.info('start load')
    top_n, digit, weight = 1, 3, False
    diagnosis_lower = 1
    read_from_cache = True
    embedding_size = 1024
    use_symptom = 1
    embedding_type = 1
    filter_key = 'mimic'
    # 两种策略，一种是只保留ICD 编码O及O以前的（后面的大部分是先天性疾病，或者不能够称为“病”），一种是所有都保留
    strategy = 'DISCARD_OL'  # DISCARD_OL ALL
    data_split_strategy = 'custom'
    logger.info(f'top n: {top_n}, digit: {digit}, weight: {weight}, data_split_strategy: {data_split_strategy},'
                f'strategy: {strategy}, diagnosis_lower: {diagnosis_lower}')

    symptom_path_list = [
        [symptom_file_path_template.format('chn'), 'chn'],
        [symptom_file_path_template.format('eng'), 'eng']
    ]
    index_symptom_dict, symptom_index_dict = get_symptom_index_map(symptom_path_list)
    (train_dataset, valid_dataset, test_dataset, diagnosis_index_map, index_diagnosis_map) = (
        read_diagnosis_symptom_embedding(
            data_file_path, symptom_info_dict, symptom_index_dict, top_n, digit, weight, filter_key=filter_key,
            diagnosis_lower=diagnosis_lower, strategy=strategy, read_from_cache=read_from_cache,
            embedding_size=embedding_size, data_split_strategy=data_split_strategy, embedding_type=embedding_type,
            use_symptom=use_symptom))

    save_key = "_".join([str(digit), filter_key, strategy])
    parse_data_symptom(train_dataset, valid_dataset, test_dataset, save_key, symptom_num_path)
    # stat
    key_set = {'srrsh-outpatient', 'srrsh-hospitalization', 'mimic_iv-hospitalization', 'mimic_iii-hospitalization'}
    dataset_origin_stat(train_dataset, valid_dataset, test_dataset, key_set)
    disease_stat(train_dataset, valid_dataset, test_dataset, index_diagnosis_map)


if __name__ == '__main__':
    main()
