import csv
import copy
import random
import numpy as np
import torch
from numpy import int64
from gymnasium.spaces import Box, Discrete
from itertools import islice
from gymnasium import Env


def read_questions_mapping_info(file_path, language):
    general_question_mapping_dict, specific_question_mapping_dict = {}, {}
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            if language == 'chn':
                symptom, factor_group, factor, specific_question, general_question = line[0:5]
            else:
                assert language == 'eng'
                symptom, factor_group, factor, specific_question, general_question = line[5:10]
            if factor_group == 'NA':
                assert factor == 'NA'
                key = symptom.strip()
            else:
                key = (symptom.strip() + ' ' + factor_group.strip() + ' ' + factor.strip()).strip()
            key = key.lower().replace('  ', ' ')
            assert specific_question != 'None' and specific_question not in specific_question_mapping_dict
            specific_question_mapping_dict[specific_question] = key
            if general_question != 'None':
                if general_question not in general_question_mapping_dict:
                    general_question_mapping_dict[general_question] = set()
                assert key not in general_question_mapping_dict[general_question]
                general_question_mapping_dict[general_question].add(key)
    return specific_question_mapping_dict, general_question_mapping_dict


class PatientEnvironment(Env):
    def __init__(self, data_pool, symptom_num, diagnosis_num, symptom_index_dict, first_level_symptom_num,
                 action_mapping_dict, max_step, proactive_positive_reveal, proactive_negative_reveal,
                 passive_positive_reveal, passive_negative_reveal, random_sample, max_epoch, time_penalty,
                 screen_model, new_finding_factor, diff_sum_factor, first_level_positive_factor, language,
                 first_level_negative_factor, negative_reward_factor, positive_reward_factor, repeat_action_penalty,
                 device, rank=None, total_env=None, embedding_size=None):
        super().__init__()
        self.data_pool = data_pool
        self.symptom_index_dict = symptom_index_dict
        self.time_penalty = time_penalty
        self.symptom_num = symptom_num
        self.diagnosis_num = diagnosis_num
        self.first_level_positive_factor = first_level_positive_factor
        self.first_level_negative_factor = first_level_negative_factor
        self.first_level_num = first_level_symptom_num
        self.max_step = max_step
        self.random_sample = random_sample
        self.max_epoch = max_epoch
        self.language = language
        self.proactive_positive_reveal = proactive_positive_reveal
        self.passive_positive_reveal = passive_positive_reveal
        self.proactive_negative_reveal = proactive_negative_reveal
        self.repeat_action_penalty = repeat_action_penalty
        self.passive_negative_reveal = passive_negative_reveal
        self.screen_model = screen_model
        self.new_finding_factor = new_finding_factor
        self.diff_sum_factor = diff_sum_factor
        self.positive_reward_factor = positive_reward_factor
        self.negative_reward_factor = negative_reward_factor
        self.device = device
        # 此处的rank专指env的整体编号
        self.rank = rank
        self.total_env = total_env
        self.embedding_size = embedding_size

        # collect代表只询问症状，在询问完毕之后训练下一个模型进行诊断
        # diagnosis代表询问和诊断都纳入动作空间
        self.action_mapping_dict = action_mapping_dict

        self.action_space = Discrete(len(action_mapping_dict) + 1)
        # 三部分，已知的symptom（加一个terminate state flag） embedding和历史action。
        # 历史action的使用策略还需进一步探索，最极端的情况时列入obs space，但是actor其实不使用这部分信息
        self.observation_space = (Box(low=-10000, high=10000,
                                      shape=[self.symptom_num * 3 + 1 + embedding_size + len(action_mapping_dict) + 1]))
        self.current_key = None
        self.current_oracle_symptom = None
        self.current_oracle_diagnosis = None
        self.current_oracle_embedding = None
        self.current_symptom_observation = None
        self.current_episode_action_set = None
        self.step_count = None

        self.iteration_list = None
        self.sample_idx = None
        self.epoch_num = 0
        # self.reset()

    def epoch_end(self):
        assert self.random_sample is False
        assert self.iteration_list is not None
        assert self.sample_idx <= len(self.iteration_list)
        if self.sample_idx == len(self.iteration_list):
            return True
        else:
            return False

    def update_history_action_vector(self):
        data = np.zeros([len(self.action_mapping_dict)+1])
        for key in self.current_episode_action_set:
            data[key] = 1
        return data

    def step(self, action):
        assert 0 <= action <= len(self.action_mapping_dict)
        assert self.current_oracle_diagnosis is not None and self.current_oracle_symptom is not None
        assert self.current_symptom_observation is not None
        assert isinstance(action, int64) or isinstance(action, int)
        # 最后一个action是terminate action

        if action == len(self.action_mapping_dict):
            self.current_episode_action_set.add(action)
            action_history = self.update_history_action_vector()
            obs = np.concatenate([self.current_symptom_observation, [1], self.current_oracle_embedding, action_history])
            last_info = {
                'key': self.current_key,
                'symptom': self.current_oracle_symptom,
                'diagnosis': self.current_oracle_diagnosis,
                'embedding': self.current_oracle_embedding,
                'action_num': len(self.current_episode_action_set)
            }
            return obs, 0, True, False, last_info

        origin_observation = copy.deepcopy(self.current_symptom_observation)
        affect_symptoms = np.array([item[0] for item in self.action_mapping_dict[action][1]])

        def _update_symptom_observation(origin_observation_, affect_symptoms_):
            # 规则：
            # 阳性数据和阴性数据都按照规定的比例进行披露。
            # 当一个问题没有问到任何有效回答时（即涉及的所有Symptom的答案都是NA时）
            # 1. 如果oracle symptom中有阳性symptom，则选择找到的第一个阳性返回
            # 2. 若都没有，则返回NA，阴性不做处理
            # 3. 按照positive信息是negative信息的5倍给reward，一级症状是二级症状的5倍给reward
            # 4. 获取已经知道的信息没有reward
            # 5. 如果一个问题是极为具体的是非题，则获取结论的概率为100%

            if len(affect_symptoms_) == 1:
                pos_ratio, neg_ratio = 1, 1
            else:
                pos_ratio, neg_ratio = self.passive_positive_reveal, self.passive_negative_reveal

            is_first_level = affect_symptoms_ < self.first_level_num
            is_second_level = affect_symptoms_ >= self.first_level_num
            current_positive = origin_observation_[affect_symptoms_ * 3 + 2]
            current_negative = origin_observation_[affect_symptoms_ * 3 + 1]

            positive = self.current_oracle_symptom[affect_symptoms_ * 3 + 2]
            negative = self.current_oracle_symptom[affect_symptoms_ * 3 + 1]
            positive_reveal = np.random.uniform(0, 1, len(affect_symptoms_)) < pos_ratio
            negative_reveal = np.random.uniform(0, 1, len(affect_symptoms_)) < neg_ratio

            # 未披露的强行处理
            if np.sum(positive) > 0 and np.sum(positive_reveal) == 0:
                for i in range(len(positive)):
                    if positive[i] == 1:
                        positive_reveal[i] = 1

            new_observation_ = copy.deepcopy(origin_observation_)
            positive_info = (new_observation_[affect_symptoms_ * 3 + 2] + positive_reveal * positive) > 0
            negative_info = (new_observation_[affect_symptoms_ * 3 + 1] + negative_reveal * negative) > 0
            assert np.sum(positive * negative) == 0
            new_observation_[affect_symptoms_ * 3 + 1] = negative_info
            new_observation_[affect_symptoms_ * 3 + 2] = positive_info
            unknown = 1 - (positive_info + negative_info)
            new_observation_[affect_symptoms_ * 3] = unknown

            new_positive = new_observation_[affect_symptoms_ * 3 + 2]
            new_negative = new_observation_[affect_symptoms_ * 3 + 1]
            positive_reward = (((new_positive - current_positive) * is_second_level +
                               (new_positive - current_positive) * is_first_level * self.first_level_positive_factor) *
                               self.positive_reward_factor)
            negative_reward = (((new_negative - current_negative) * is_second_level * +
                               (new_negative - current_negative) * is_first_level * self.first_level_negative_factor) *
                               self.negative_reward_factor)
            new_finding_reward = np.sum(positive_reward) + np.sum(negative_reward)

            if self.diff_sum_factor > 0:
                # 计算原先的模型预测结果
                origin_model_input = np.concatenate([origin_observation_, self.current_oracle_embedding], axis=0)
                origin_model_input = torch.FloatTensor(origin_model_input.reshape(1, -1)).to(self.device)
                origin_pred = self.screen_model(origin_model_input).detach().to('cpu').numpy()
                # 新的模型计算结果
                new_model_input = np.concatenate([new_observation_, self.current_oracle_embedding], axis=0)
                new_model_input = torch.FloatTensor(new_model_input.reshape(1, -1)).to(self.device)
                new_pred = self.screen_model(new_model_input).detach().to('cpu').numpy()

                origin_pred_prob = 1 / (1 + np.exp(origin_pred * -1))
                new_pred_prob = 1 / (1 + np.exp(new_pred * -1))
                diff_sum = np.sum(np.abs(origin_pred_prob - new_pred_prob))
            else:
                diff_sum = 0

            # diff_list.append(diff_sum)
            # new_finding_list.append(new_finding_reward)
            # if len(diff_list) % 1000 == 0 and len(diff_list) > 0:
            #     logger.info(f'diff list len: {len(diff_list)}, avg value: {np.average(diff_list)}')
            #     logger.info(f'new finding reward list len: {len(new_finding_list)}, '
            #                 f'avg value: {np.average(new_finding_list)}')

            reward_ = self.time_penalty + self.new_finding_factor * new_finding_reward + self.diff_sum_factor * diff_sum
            return new_observation_, reward_

        new_observation, reward = \
            _update_symptom_observation(origin_observation, affect_symptoms)
        if action in self.current_episode_action_set:
            reward -= self.repeat_action_penalty

        self.current_episode_action_set.add(action)
        action_history = self.update_history_action_vector()
        terminate = False
        # reward = reward - 0.1
        self.step_count += 1
        if self.step_count == self.max_step:
            truncate = True
            return_info = {
                'key': self.current_key,
                'symptom': self.current_oracle_symptom,
                'diagnosis': self.current_oracle_diagnosis,
                'embedding': self.current_oracle_embedding,
                'action_num': len(self.current_episode_action_set)
            }
        elif self.step_count < self.max_step:
            truncate = False
            return_info = {}
        elif self.step_count > self.max_step:
            raise ValueError('')
        else:
            raise ValueError('')
        self.current_symptom_observation = new_observation

        return_obs = np.concatenate([self.current_symptom_observation, [1], self.current_oracle_embedding,
                                     action_history])
        # if reward != 1 and reward!=0:
        #     raise ValueError('')
        return return_obs, reward, terminate, truncate, return_info

    def reset(self, seed=None, options=None):
        if self.sample_idx is None or self.sample_idx == len(self.iteration_list):
            if self.random_sample:
                self.iteration_list = [i for i in range(len(self.data_pool))]
                random.Random().shuffle(self.iteration_list)
                self.sample_idx = 0
            else:
                assert isinstance(self.total_env, int) and isinstance(self.rank, int)
                start_idx = len(self.data_pool) // self.total_env * self.rank
                end_idx = len(self.data_pool) // self.total_env * (self.rank + 1)
                self.iteration_list = [i for i in range(start_idx, end_idx)]
                self.sample_idx = 0

        # 重设已经获知的symptom
        next_sample = self.data_pool[self.iteration_list[self.sample_idx]]
        self.current_key = next_sample[0]
        self.current_oracle_symptom = next_sample[1]
        self.current_oracle_diagnosis = next_sample[2]
        self.current_oracle_embedding = next_sample[3]
        self.current_episode_action_set = set()

        # 确认一级症状会在最前面
        positive_list, negative_list = [], []
        for j in range(self.first_level_num):
            if self.current_oracle_symptom[j * 3 + 2] == 1:
                positive_list.append(j)
            if self.current_oracle_symptom[j * 3 + 1] == 1:
                negative_list.append(j)

        next_observation = np.zeros([self.symptom_num * 3])
        next_observation[0::3] = 1

        # 每个症状都根据预先规定的比例在初始化时被更新出来
        # 其中，我们要求必须至少有一个阳性症状被披露（不然就变成病人没有任何不舒服但是来医院了）。
        assert len(positive_list) > 0
        positive_hit_at_least_one, positive_hit_list, negative_hit_list = False, [], []
        for symptom_idx in positive_list:
            prob = random.uniform(0, 1)
            if prob < self.proactive_positive_reveal:
                next_observation[symptom_idx * 3 + 2] = 1
                next_observation[symptom_idx * 3] = 0
                positive_hit_at_least_one = True
                positive_hit_list.append(symptom_idx)
        for symptom_idx in negative_list:
            prob = random.uniform(0, 1)
            if prob < self.proactive_negative_reveal:
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
            for key in self.symptom_index_dict[self.language]:
                idx, parent_index = self.symptom_index_dict[self.language][key]
                if parent_index == symptom_idx:
                    if self.current_oracle_symptom[idx * 3 + 2] == 1:
                        prob = random.uniform(0, 1)
                        if prob < self.proactive_positive_reveal:
                            next_observation[idx * 3: (idx+1) * 3] = self.current_oracle_symptom[idx * 3: (idx+1) * 3]
                    if self.current_oracle_symptom[idx * 3 + 1] == 1:
                        prob = random.uniform(0, 1)
                        if prob < self.proactive_negative_reveal:
                            next_observation[idx * 3: (idx+1) * 3] = self.current_oracle_symptom[idx * 3: (idx+1) * 3]
        # 一级症状如果被否认，则相应的二级症状全部置否
        for symptom_idx in negative_list:
            for key in self.symptom_index_dict[self.language]:
                idx, parent_index = self.symptom_index_dict[self.language][key]
                if parent_index == symptom_idx:
                    assert self.current_oracle_symptom[idx * 3 + 1] == 1
                    next_observation[idx * 3: (idx + 1) * 3] = self.current_oracle_symptom[idx * 3: (idx + 1) * 3]

        self.sample_idx += 1
        self.current_symptom_observation = next_observation
        self.step_count = 0
        self.epoch_num += 1
        assert self.max_epoch is None or self.epoch_num <= self.max_epoch

        # 重设embedding
        embedding = np.array(self.current_oracle_embedding)
        action_history = self.update_history_action_vector()
        assert np.sum(action_history) == 0
        return_obs = np.concatenate([next_observation, [0], embedding, action_history])
        return return_obs, {}

    def render(self):
        print('call render, current key: {}'.format(self.current_key))

    def close(self):
        print('close, current key: {}'.format(self.current_key))
