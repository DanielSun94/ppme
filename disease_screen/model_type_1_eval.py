import os
import numpy as np
from disease_screen_logger import logger
from stable_baselines3.common.callbacks import BaseCallback
import copy
import torch
from datetime import datetime
from util import primary_full_calculate

#
# class DiagnosisFullDiagnosisEvalCallback(BaseCallback):
#     def __init__(self, envs_train, envs_valid, envs_test, model, language, eval_per_step, episode_max_len,
#                  save_path, verbose: int = 0):
#         super().__init__(verbose)
#         self.eval_model = model
#         self.eval_envs_train = envs_train
#         self.eval_envs_valid = envs_valid
#         self.eval_envs_test = envs_test
#         self.eval_per_step = eval_per_step
#         self.save_path = save_path
#         self.language = language
#         self.episode_max_len = episode_max_len
#
#     def performance_eval(self, current_step):
#         model = self.model
#         max_step = self.eval_envs_train.envs[0].env.max_step
#
#         result = []
#         for eval_envs in (self.eval_envs_train, self.eval_envs_valid, self.eval_envs_test):
#             obs_list = []
#             symptom_list = []
#             diagnosis_list = []
#             with torch.no_grad():
#                 obs = eval_envs.reset()
#                 while not eval_envs.envs[0].unwrapped.epoch_end():
#                     for env_monitor in eval_envs.envs:
#                         diagnosis_list.append(copy.deepcopy(env_monitor.env.current_oracle_diagnosis))
#                         symptom_list.append(copy.deepcopy(env_monitor.env.current_oracle_symptom))
#
#                     for i in range(max_step):
#                         action, _states = model.predict(obs, deterministic=True)
#                         obs, rewards, dones, info = eval_envs.step(action)
#                         if i < max_step - 1:
#                             assert np.sum(dones) == 0
#                         else:
#                             assert np.sum(dones) == eval_envs.num_envs
#                     # 这里因为自动重载，不能用final obs,并且要先注入diagnosis_list和symptom_list
#                     for idx in range(len(eval_envs.buf_infos)):
#                         obs_list.append(copy.deepcopy(eval_envs.buf_infos[idx]['terminal_observation']))
#
#             # assert eval_envs.epoch_end()
#             observation = np.array(obs_list)
#             symptom = np.array(symptom_list)
#             result.append([diagnosis_list, observation, symptom])
#         logger.info('test environment constructed')
#         # disease_num = self.eval_envs_train.envs[0].env.diagnosis_num
#         clf = full_disease_predict_eval(result, 64, 269, 16, 20, 2151)
#         first_level_action = self.eval_envs_train.envs[0].env.first_level_num
#         symptom_hit_eval(result, first_level_action, current_step, 2151)
#         return clf
#
#     def model_save(self, clf, current_step):
#         model = self.model
#         now = datetime.now().strftime('%Y%m%d%H%M%S')
#         policy_path = 'model_{}_{}_{}_{}_policy.pth'.format(self.language, self.episode_max_len, current_step, now)
#         other_path = 'model_{}_{}_{}_{}_clf_dict.pkl'.format(self.language, self.episode_max_len, current_step, now)
#         policy_path = os.path.join(self.save_path, policy_path)
#         other_path = os.path.join(self.save_path, other_path)
#         torch.save(model.policy.state_dict(), policy_path)
#         pickle.dump(
#             [
#                 clf,
#                 model.policy_kwargs['symptom_index_dict']
#             ],
#             open(other_path, 'wb')
#         )
#
#     def _on_training_start(self) -> None:
#         clf = self.performance_eval(0)
#         self.model_save(clf, 0)
#
#     def _on_rollout_start(self) -> None:
#         pass
#
#     def _on_step(self) -> bool:
#         """
#         This method will be called by the model after each call to `env.step()`.
#
#         For child callback (of an `EventCallback`), this will be called
#         when the event is triggered.
#
#         :return: If the callback returns False, training is aborted early.
#         """
#         current_step = self.num_timesteps
#         if current_step % self.eval_per_step != 0 and current_step != 0:
#             return True
#         clf = self.performance_eval(current_step)
#         self.model_save(clf, current_step)
#         return True
#
#     def _on_rollout_end(self) -> None:
#         pass
#
#     def _on_training_end(self) -> None:
#         self.performance_eval(-1)
#         pass


# def full_disease_predict_eval(result, hidden_size, output_size, batch_size, epochs, obs_dim):
#     diagnosis_list_train = np.array(result[0][0])
#     observation_list_train = np.array(result[0][1])
#     diagnosis_list_test = np.array(result[2][0])
#     observation_list_test = np.array(result[2][1])
#
#     train_obs_symptom = observation_list_train[:, :obs_dim][:, 2::3]
#     train_obs_history = observation_list_train[:, obs_dim:]
#     train_obs = np.concatenate([train_obs_symptom, train_obs_history], axis=1)
#     test_obs_symptom = observation_list_test[:, :obs_dim][:, 2::3]
#     test_obs_history = observation_list_test[:, obs_dim:]
#     test_obs = np.concatenate([test_obs_symptom, test_obs_history], axis=1)
#     ranking_model = ranking_model_training(train_obs, diagnosis_list_train, test_obs, diagnosis_list_test,
#                                            hidden_size, output_size, batch_size, epochs)
#     return ranking_model


def symptom_hit_eval(result, first_level_action, step_num):
    symptom_size = len(result[0][1][0])
    symptom_list_test = np.array(result[0][1])
    observation_list_test = np.array(result[0][0])[:, :symptom_size]

    test_match = observation_list_test * symptom_list_test

    symptom_num = symptom_size // 3
    test_first_level_hit, test_all_hit = 0, 0
    test_symptom_all, test_symptom_first_level = 0, 0
    for i in range(symptom_num):
        idx = i * 3 + 2
        if i < first_level_action:
            test_first_level_hit += np.sum(test_match[:, idx])
            test_symptom_first_level += np.sum(symptom_list_test[:, idx])
        test_all_hit += np.sum(test_match[:, idx])
        test_symptom_all += np.sum(symptom_list_test[:, idx])

    test_all_hit_average = test_all_hit / len(test_match)
    test_f_hit_average = test_first_level_hit / len(test_match)

    test_f_symptom_average = test_symptom_first_level / len(test_match)
    test_all_symptom_average = test_symptom_all / len(test_match)
    logger.info(f'step num: {step_num:10d}, test avg positive sym: {test_all_symptom_average:3.5f}, '
                f'test first level positive sym: {test_f_symptom_average:4f}')
    logger.info(f'step num: {step_num:10d}, test avg positive hit: {test_all_hit_average:3.5f}, '
                f'test first level positive hit: {test_f_hit_average:5f}')


class DiagnosisDiagnosisEvalCallback(BaseCallback):
    def __init__(self, envs_train, envs_valid, envs_test, model, language, eval_per_step, episode_max_len,
                 save_path, symptom_index_dict, diagnosis_index_map, filter_key, device, verbose: int = 0):
        super().__init__(verbose)
        self.eval_model = model
        self.eval_envs_train = envs_train
        self.eval_envs_valid = envs_valid
        self.eval_envs_test = envs_test
        self.eval_per_step = eval_per_step
        self.save_path = save_path
        self.language = language
        self.episode_max_len = episode_max_len
        self.symptom_index_dict = symptom_index_dict
        self.diagnosis_index_map = diagnosis_index_map
        self.filter_key = filter_key
        self.device = device

    def performance_eval(self, current_step):
        model = self.model
        result = []
        logger.info('start generating observation')
        for eval_envs in (self.eval_envs_test,):
            observation_fraction, symptom_fraction, diagnosis_fraction, embedding_fraction, len_list = (
                [], [], [], [], [])
            action_num_list = []
            epoch_end_test = np.zeros(len(eval_envs.envs))
            with torch.no_grad():
                obs = eval_envs.reset()
                # while not eval_envs.envs[0].unwrapped.epoch_end():
                while not (np.sum(epoch_end_test) == len(epoch_end_test)):
                    action, _states = model.predict(obs)
                    obs, rewards, dones, infos = eval_envs.step(action)
                    for j in range(len(infos)):
                        if epoch_end_test[j] == 1:
                            continue
                        if 'terminal_observation' in infos[j]:
                            assert 'key' in infos[j]
                            # 这里因为自动重载，所以所有必须信息都要从info里面获取
                            unified_id = infos[j]['key']
                            diagnosis = infos[j]['diagnosis']
                            symptom = infos[j]['symptom']
                            observation = copy.deepcopy(infos[j]['terminal_observation'])
                            embedding = infos[j]['embedding']
                            length = int(infos[j]['episode']['l'])
                            action_num = infos[j]['action_num']
                            observation_fraction.append(observation)
                            symptom_fraction.append(symptom)
                            diagnosis_fraction.append(diagnosis)
                            embedding_fraction.append(embedding)
                            action_num_list.append(action_num)
                            len_list.append(length)
                        if eval_envs.envs[j].unwrapped.epoch_end():
                            epoch_end_test[j] = 1

            logger.info('observation fraction size: {}'.format(len(observation_fraction)))
            logger.info(f'avg episode length: {np.average(len_list)}, unique action_num: {np.average(action_num_list)}')
            result.append([observation_fraction, symptom_fraction, diagnosis_fraction, embedding_fraction, unified_id])

        prediction_model = eval_envs.envs[0].unwrapped.screen_model
        disease_predict_eval(result, prediction_model, self.device)
        first_level_action = self.eval_envs_train.envs[0].env.first_level_num
        symptom_hit_eval(result, first_level_action, current_step)

    def model_save(self, current_step):
        model = self.model
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        policy_path = ('model_{}_{}_{}_{}_{}_policy.pth'
                       .format(self.filter_key, self.language, self.episode_max_len, current_step, now))
        policy_path = os.path.join(self.save_path, policy_path)
        torch.save(model.policy.state_dict(), policy_path)

    def _on_training_start(self) -> None:
        self.performance_eval(0)
        # self.model_save(0)
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        current_step = self.num_timesteps
        if current_step % self.eval_per_step != 0 and current_step != 0:
            return True
        self.performance_eval(current_step)
        self.model_save(current_step)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        self.performance_eval(-1)
        self.model_save(-1)
        pass


def disease_predict_eval(result, model, device):
    symptom_size = len(result[0][1][0])
    diagnosis_list_test = np.array(result[0][2])
    observation_list_test = np.array(result[0][0])[:, :symptom_size]
    embedding_list_test = np.array(result[0][3])

    test_input = np.concatenate([observation_list_test, embedding_list_test], axis=1)
    test_input = torch.FloatTensor(test_input).to(device)
    test_diagnosis_prob = model(test_input)
    test_diagnosis_prob = test_diagnosis_prob.detach().to('cpu').numpy()
    average_test_label = np.sum(diagnosis_list_test) / len(diagnosis_list_test)
    logger.info('average test label: {:4f}'.format(average_test_label))
    for top_k in [1, 3, 5, 10]:
        test_top_k_hit = primary_full_calculate(diagnosis_list_test, test_diagnosis_prob, top_k)
        logger.info(f'test top {top_k:2d} hit: {test_top_k_hit:5f}')
    logger.info('\n')
