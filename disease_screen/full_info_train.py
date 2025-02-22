import json
from torch import nn, FloatTensor
import math
import os.path
from datetime import datetime
from read_data import read_diagnosis_symptom_embedding, get_symptom_index_map
from disease_screen_config import (symptom_info_dict, data_file_path, screen_model_save_folder,
                                   symptom_file_path_template, icu_patient_idx_file)
from disease_screen_logger import logger
import numpy as np
import argparse
import random
from util import primary_full_calculate
from torch.utils.data import DataLoader, TensorDataset
import torch


def symptom_reorganize(oracle_symptom, first_level_num, symptom_num, positive_reveal, negative_reveal,
                       symptom_index_dict):
    if positive_reveal == 1 and negative_reveal == 1:
        return oracle_symptom

    positive_list, negative_list = [], []
    for j in range(first_level_num):
        if oracle_symptom[j * 3 + 2] == 1:
            positive_list.append(j)
        if oracle_symptom[j * 3 + 1] == 1:
            negative_list.append(j)
    assert len(positive_list) > 0
    next_observation = np.zeros([symptom_num * 3])
    next_observation[0::3] = 1

    # 每个症状大致都有预先规定的比例在初始化时被更新出来
    # 其中，我们要求必须至少有一个阳性症状被披露（不然就变成病人没有任何不舒服但是来医院了）。
    assert len(positive_list) > 0
    positive_hit_at_least_one, positive_hit_list, negative_hit_list = False, [], []
    for symptom_idx in positive_list:
        prob = random.uniform(0, 1)
        if prob < positive_reveal:
            next_observation[symptom_idx * 3 + 2] = 1
            next_observation[symptom_idx * 3] = 0
            positive_hit_at_least_one = True
            positive_hit_list.append(symptom_idx)
    for symptom_idx in negative_list:
        prob = random.uniform(0, 1)
        if prob < negative_reveal:
            next_observation[symptom_idx * 3 + 1] = 1
            next_observation[symptom_idx * 3] = 0
            negative_hit_list.append(symptom_idx)
    if not positive_hit_at_least_one:
        choice = int(random.choice(positive_list))
        next_observation[choice * 3 + 2] = 1
        next_observation[choice * 3] = 0
        positive_hit_list.append(choice)

    # 一级症状被确认过，才会披露二级症状。二级症状的披露比例一致
    for symptom_idx in positive_hit_list:
        for key in symptom_index_dict:
            idx, _, parent_index = symptom_index_dict[key]
            if parent_index == symptom_idx:
                if oracle_symptom[idx * 3 + 2] == 1:
                    prob = random.uniform(0, 1)
                    if prob < positive_reveal:
                        next_observation[idx * 3: (idx + 1) * 3] = oracle_symptom[idx * 3: (idx + 1) * 3]
                if oracle_symptom[idx * 3 + 1] == 1:
                    prob = random.uniform(0, 1)
                    if prob < negative_reveal:
                        next_observation[idx * 3: (idx + 1) * 3] = oracle_symptom[idx * 3: (idx + 1) * 3]
    # 一级症状如果被否认，则相应的二级症状全部置否
    for symptom_idx in negative_list:
        for key in symptom_index_dict:
            idx, _, parent_index = symptom_index_dict[key]
            if parent_index == symptom_idx:
                assert (oracle_symptom[idx * 3 + 1] == 1 and oracle_symptom[idx * 3] == 0 and
                        oracle_symptom[idx * 3 + 2] == 0)
                next_observation[idx * 3: (idx + 1) * 3] = oracle_symptom[idx * 3: (idx + 1) * 3]
    return next_observation


def get_dataset(dataset, positive_reveal, negative_reveal, data_fraction, symptom_num, first_level_num,
                symptom_index_dict, extra_filter):
    diagnosis_list = []
    observation_list = []
    data_limit = math.ceil(len(dataset) * data_fraction)
    idx_list = [i for i in range(len(dataset))]
    random.shuffle(idx_list)

    final_invalid_embedding_num = 0
    for index, item in enumerate(dataset):
        if idx_list[index] >= data_limit:
            continue
        key, symptom, diagnosis, embedding, _ = item
        assert np.sum(diagnosis) == 1
        diagnosis = np.argmax(diagnosis)
        embedding_sum = np.sum(np.abs(embedding))
        if embedding_sum == 0:
            final_invalid_embedding_num += 1
        if len(extra_filter) > 1 and extra_filter not in key:
            continue
        corrupted_symptom = symptom_reorganize(symptom, first_level_num, symptom_num, positive_reveal, negative_reveal,
                                               symptom_index_dict)
        observation_list.append(list(corrupted_symptom) + list(embedding))
        diagnosis_list.append(diagnosis)

    diagnosis_list = np.array(diagnosis_list)
    logger.info('final_invalid_embedding_num: {}'.format(final_invalid_embedding_num))
    return observation_list, diagnosis_list


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        assert len(hidden_sizes) == 2
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size)
        )

    def forward(self, x):
        return self.model(x)


def model_train(train_dataset, test_dataset, clf_iter_num, eval_epoch_interval, positive_reveal, negative_reveal,
                data_fraction, symptom_num, first_level_num, symptom_index_dict, extra_filter, batch_size,
                diagnosis_num, learning_rate, device):
    logger.info('start data reorganization ...')
    train_observation_list, train_diagnosis_list = \
        get_dataset(train_dataset, positive_reveal, negative_reveal, data_fraction, symptom_num, first_level_num,
                    symptom_index_dict, extra_filter)
    logger.info('train_split_success')
    train_observation_list = torch.tensor(train_observation_list, dtype=torch.float32)
    train_diagnosis_list = torch.tensor(train_diagnosis_list, dtype=torch.long)
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_observation_list, train_diagnosis_list)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f'data reorganization succeed, train size: {len(train_observation_list)}')
    logger.info(f'data reorganization succeed, test size: {len(test_dataset)}')

    input_size = train_observation_list.shape[1]
    hidden_sizes = [128, 128]
    model = MLP(input_size, hidden_sizes, diagnosis_num).to(device)

    criterion = nn.CrossEntropyLoss()
    learning_rate_init = learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)

    # 学习率调度器，模仿 'adaptive' 学习率策略
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    logger.info('start training')
    # 训练循环
    current_step, current_epoch = 0, 0
    while True:
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            current_step += 1
        # 计算平均损失并更新学习率
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        logger.info(f'Epoch [{current_epoch + 1}], Loss: {epoch_loss:.8f}')
        if current_step > clf_iter_num:
            break
        if current_epoch % eval_epoch_interval == 0:
            with torch.no_grad():
                logger.info(f'mediate evaluation, current_step: [{current_step}/{clf_iter_num}]')
                model_eval('test', model, test_dataset, positive_reveal, negative_reveal,
                           symptom_num, first_level_num, symptom_index_dict, extra_filter, diagnosis_num, device)
        current_epoch += 1
    model_eval('test', model, test_dataset, positive_reveal, negative_reveal,
               symptom_num, first_level_num, symptom_index_dict, extra_filter, diagnosis_num, device)
    logger.info('end training \n')
    return model


def model_eval(key_name, model, dataset, positive_reveal, negative_reveal, symptom_num, first_level_num,
               symptom_index_dict, extra_filter, diagnosis_num, device):
    observation_list, diagnosis_list = get_dataset(
        dataset, positive_reveal, negative_reveal, 1, symptom_num, first_level_num, symptom_index_dict,
        extra_filter)
    logger.info(f'test size: {len(observation_list)}')

    diagnosis_array = np.zeros([len(diagnosis_list), diagnosis_num])
    for i in range(len(diagnosis_array)):
        hot_idx = diagnosis_list[i]
        diagnosis_array[i, hot_idx] = 1
    observation_list = FloatTensor(observation_list).to(device)
    diagnosis_prob = model(observation_list)

    diagnosis_prob = diagnosis_prob.detach().to('cpu').numpy()
    logger.info('')
    for top_k in [1, 5, 10, 20]:
        top_k_hit = primary_full_calculate(diagnosis_array, diagnosis_prob, top_k)
        logger.info(f'key: {key_name}, positive_reveal: {positive_reveal}, negative_reveal: {negative_reveal}, '
                    f'top {top_k:2d} hit: {top_k_hit:5f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='', default=datetime.now().strftime('%Y%m%d%H%M%S'), type=str)
    parser.add_argument('--strategy', help='', default='DISCARD_OL', type=str)  # ALL
    parser.add_argument('--top_n', help='', default=1, type=int)
    parser.add_argument('--digit', help='', default=3, type=int)
    parser.add_argument('--diagnosis_lower', help='', default=20, type=int)
    parser.add_argument('--clf_iter_num', help='', default=100000, type=int)
    parser.add_argument('--learning_rate', help='', default=0.001, type=float)
    parser.add_argument('--positive_reveal', help='', default=1, type=float)
    parser.add_argument('--negative_reveal', help='', default=1, type=float)
    parser.add_argument('--read_from_cache', help='', default=1, type=int)
    parser.add_argument('--weight', help='', default=0, type=int)
    parser.add_argument('--data_fraction', help='', default=1.0, type=float)
    parser.add_argument('--extra_filter', help='', default="", type=str)
    parser.add_argument('--filter_key', help='', default="mimic", type=str)
    parser.add_argument('--embedding_type', help='', default=2, type=int)
    parser.add_argument('--embedding_size', help='', default=1024, type=int)
    parser.add_argument('--use_symptom', help='', default=1, type=int)
    parser.add_argument('--epoch_eval_interval', help='', default=10, type=int)
    parser.add_argument('--batch_size', help='', default=512, type=int)
    parser.add_argument('--first_level_symptom', help='', default=85, type=int)
    parser.add_argument('--data_split_strategy', help='', default="custom", type=str)
    parser.add_argument('--device', help='', default="cuda:6", type=str)
    args, args_list = vars(parser.parse_args()), []
    for key in args:
        args_list.append([key, args[key]])
    args_list = sorted(args_list, key=lambda x: x[0])
    for item in args_list:
        logger.info('{}: {}'.format(item[0], item[1]))

    top_n = args['top_n']
    digit = args['digit']
    strategy = args['strategy']
    filter_key = args['filter_key']
    model_id = args['model_id']
    clf_iter_num = args['clf_iter_num']
    learning_rate = args['learning_rate']
    diagnosis_lower = args['diagnosis_lower']
    positive_reveal = args['positive_reveal']
    negative_reveal = args['negative_reveal']
    data_fraction = args['data_fraction']
    extra_filter = args['extra_filter']
    device = args['device']
    batch_size = args['batch_size']
    epoch_eval_interval = args['epoch_eval_interval']
    embedding_size = args['embedding_size']
    first_level_symptom = args['first_level_symptom']
    data_split_strategy = args['data_split_strategy']
    embedding_type = args['embedding_type']
    weight = True if args['weight'] == 1 else False
    read_from_cache = True if args['read_from_cache'] == 1 else False
    use_symptom = True if args['use_symptom'] == 1 else False
    logger.info('start loading data')

    symptom_path_list = [
        [symptom_file_path_template.format('chn'), 'chn'],
        [symptom_file_path_template.format('eng'), 'eng']
    ]
    index_symptom_dict, symptom_index_dict = get_symptom_index_map(symptom_path_list)

    if filter_key == 'srrsh-hospitalization-severe':
        filter_key_ = 'srrsh-hospitalization'
        external_key_dict = json.load(open(icu_patient_idx_file, 'r', encoding='utf-8-sig'))
        external_key_set = set()
        for key in external_key_dict:
            if external_key_dict[key] == 1:
                external_key_set.add(key)
    else:
        filter_key_ = filter_key
        external_key_set = None
    # 注意，此处的diagnosis_index_map是全局统一的，不会因为选择项区别而变化，如果
    (train_dataset, valid_dataset, test_dataset, diagnosis_index_map, index_diagnosis_map) = (
        read_diagnosis_symptom_embedding(
            data_file_path, symptom_info_dict, symptom_index_dict, top_n, digit, weight, filter_key=filter_key_,
            read_from_cache=read_from_cache, strategy=strategy, diagnosis_lower=diagnosis_lower,
            embedding_type=embedding_type, use_symptom=use_symptom, embedding_size=embedding_size,
            data_split_strategy=data_split_strategy, external_key_set=external_key_set
        ))

    # parse_data_symptom(train_dataset, valid_dataset, test_dataset, filter_key, symptom_num_path)
    logger.info('data loaded')
    symptom_num = len(symptom_index_dict)
    # first_level_num, first_level_idx_list = 0, []
    # for key in symptom_index_dict[lan]:
    #     if ' ' not in symptom_index_dict[key]:
    #         first_level_num += 1
    #         first_level_idx_list.append(symptom_index_dict[key][0])
    # assert np.sum(np.array(first_level_idx_list) + 1) == \
    #   (1 + len(first_level_idx_list)) * len(first_level_idx_list) / 2
    logger.info(f'diagnosis number: {len(diagnosis_index_map)}')

    diagnosis_num = len(diagnosis_index_map)
    model = model_train(train_dataset, test_dataset, clf_iter_num, epoch_eval_interval, positive_reveal,
                        negative_reveal, data_fraction, symptom_num, first_level_symptom, symptom_index_dict,
                        extra_filter, batch_size, diagnosis_num, learning_rate, device)

    save_path = os.path.join(screen_model_save_folder, f'model_{filter_key}_{model_id}.pth')

    torch.save(
        {
            'index_symptom_dict': index_symptom_dict,
            'symptom_index_dict': symptom_index_dict,
            'diagnosis_index_map': diagnosis_index_map,
            'index_diagnosis_map': index_diagnosis_map,
            'model_state_dict': model.state_dict()
        },
        save_path
    )

    # data_dict = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    # # for positive_reveal_eval in (1, ):
    # #     for negative_reveal_eval in (0.5, ):
    # # for key in 'train', 'valid', 'test':
    # for positive_reveal_eval in (1, 0.9, 0.8, 0.7, 0.6):
    #     for negative_reveal_eval in (1, 0.2, 0.3, 0.4, 0.5, 0.6):
    #         for key in ('test',):
    #             dataset = data_dict[key]
    #             model_eval(key, model, dataset, positive_reveal_eval, negative_reveal_eval, 1, symptom_num,
    #                        first_level_num, symptom_index_dict, extra_filter, device)


if __name__ == '__main__':
    main()
