import os
import torch
from model_type_1_env import PatientEnvironment, read_questions_mapping_info
from read_data import read_diagnosis_symptom_embedding, get_symptom_index_map
import argparse
from disease_screen_logger import logger
from datetime import datetime
from stable_baselines3 import PPO, A2C, DQN
import torch as th
from util import make_vec_env, LinearSchedule, set_action_mapping_dict
from model_type_1_eval import DiagnosisDiagnosisEvalCallback
from model_type_1_policy_model import SymptomInquireActorCriticPolicy
from full_info_train import MLP
from disease_screen_config import (symptom_info_dict, data_file_path, screen_model_save_folder,
                                   question_file_path, symptom_file_path_template)

model_id = datetime.now().strftime('%Y%m%d%H%M%S')

parser = argparse.ArgumentParser()
parser.add_argument('--screen_model_name', help='',
                    default='model_mimic_20250211103028_DISCARD_OL.pth', type=str)
parser.add_argument('--language', help='', default='eng', type=str)
parser.add_argument('--model_name', help='', default='ppo', type=str)
parser.add_argument('--filter_key', help='', default='mimic', type=str)
parser.add_argument('--model_id', help='', default=model_id, type=str)
parser.add_argument('--first_level', help='', default=85, type=int)
parser.add_argument('--top_n_diagnosis', help='', default=1, type=int)
parser.add_argument('--icd_digit', help='', default=3, type=int)
parser.add_argument('--learning_rate', help='', default=0.001, type=float)
parser.add_argument('--diagnosis_weight', help='', default=0, type=int)
parser.add_argument('--value_weight', help='', default=1.0, type=float)
parser.add_argument('--entropy_weight', help='', default=0.04, type=float)
parser.add_argument('--proactive_positive_reveal', help='', default=0.7, type=float)
parser.add_argument('--proactive_negative_reveal', help='', default=1.0, type=float)
parser.add_argument('--passive_positive_reveal', help='', default=0.7, type=float)
parser.add_argument('--passive_negative_reveal', help='', default=1.0, type=float)
parser.add_argument('--positive_reward_factor', help='', default=1, type=float)
parser.add_argument('--negative_reward_factor', help='', default=0.2, type=float)
parser.add_argument('--first_level_positive_factor', help='', default=5, type=float)
parser.add_argument('--first_level_negative_factor', help='', default=1, type=float)
parser.add_argument('--embedding_size', help='', default=1024, type=int)
parser.add_argument('--mask_history_action', help='', default=1, type=int)
parser.add_argument('--repeat_action_penalty', help='', default=0, type=float)
parser.add_argument('--time_penalty', help='', default=-0.4, type=float)
parser.add_argument('--new_finding_factor', help='', default=1, type=float)
parser.add_argument('--diff_sum_factor', help='', default=2, type=float)
parser.add_argument('--embedding_type', help='', default=2, type=int)
parser.add_argument('--use_symptom', help='', default=1, type=int)
parser.add_argument('--n_envs', help='', default=256, type=int)
parser.add_argument('--episode_max_len', help='', default=10, type=int)
parser.add_argument('--update_per_step', help='', default=20, type=int)
parser.add_argument('--symptom_num', help='', default=1264, type=int)
parser.add_argument('--data_split_strategy', help='', default='custom', type=str)
parser.add_argument('--diagnosis_lower_threshold', help='', default=20, type=int)
parser.add_argument('--diagnosis_filter_strategy', help='', default="DISCARD_OL", type=str)
parser.add_argument('--device', help='', default='cuda:0', type=str)
args = vars(parser.parse_args())
args_list = []
for key in args:
    args_list.append([key, args[key]])
args_list = sorted(args_list, key=lambda x: x[0])
for item in args_list:
    logger.info('{}: {}'.format(item[0], item[1]))
model_name = args['screen_model_name']
screen_model_path = os.path.join(screen_model_save_folder, model_name)
logger.info('{}: {}'.format('screening model path', screen_model_path))


def main():
    n_envs = args['n_envs']
    top_n_diagnosis = args['top_n_diagnosis']
    icd_digit = args['icd_digit']
    diagnosis_weight = args['diagnosis_weight']
    init_learning_rate = args['learning_rate']
    first_level = args['first_level']
    episode_max_len = args['episode_max_len']
    device = args['device']
    update_per_step = args['update_per_step']
    value_weight = args['value_weight']
    entropy_weight = args['entropy_weight']
    language = args['language']
    diagnosis_lower_threshold = args['diagnosis_lower_threshold']
    diagnosis_filter_strategy = args['diagnosis_filter_strategy']
    first_level_positive_factor = args['first_level_positive_factor']
    first_level_negative_factor = args['first_level_negative_factor']
    time_penalty = args['time_penalty']
    diff_sum_factor = args['diff_sum_factor']
    new_finding_factor = args['new_finding_factor']
    positive_reward_factor = args['positive_reward_factor']
    negative_reward_factor = args['negative_reward_factor']
    embedding_size = args['embedding_size']
    # 对应一开始的主动陈述（主动告知阳性症状和阴性症状的概率），以及后续的被动开放性提问的主动告知概率
    # 注意，如果被动的不是开放性提问（也就是针对某个特定症状的是非题），则回答概率为100%，此处不做设置
    proactive_positive_reveal = args['proactive_positive_reveal']
    proactive_negative_reveal = args['proactive_negative_reveal']
    passive_positive_reveal = args['passive_positive_reveal']
    passive_negative_reveal = args['passive_negative_reveal']
    embedding_type = args['embedding_type']
    mask_history_action = args['mask_history_action']
    repeat_action_penalty = args['repeat_action_penalty']
    filter_key = args['filter_key']
    use_symptom = args['use_symptom']
    data_split_strategy = args['data_split_strategy']

    assert language == 'chn' or language == 'eng'
    if 'srrsh' in filter_key:
        assert language == 'chn'
    else:
        assert language == 'eng' and 'mimic' in filter_key
    assert mask_history_action == 1 or mask_history_action == 0
    assert embedding_type in {0, 1, 2}
    mask_history_action_bool = True if mask_history_action == 1 else False
    assert ((mask_history_action_bool and repeat_action_penalty == 0) or
            (not mask_history_action_bool and repeat_action_penalty > 0))

    assert diagnosis_weight == 1 or diagnosis_weight == 0
    diagnosis_weight = True if diagnosis_weight == 1 else False

    model_info_dict = torch.load(screen_model_path)
    input_size = 4816
    hidden_sizes = [128, 128]
    output_size = len(model_info_dict['index_diagnosis_map'])
    model = MLP(input_size, hidden_sizes, output_size)
    state_dict = model_info_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)

    symptom_path_list = [
        [symptom_file_path_template.format('chn'), 'chn'],
        [symptom_file_path_template.format('eng'), 'eng']
    ]
    index_symptom_dict, symptom_index_dict = get_symptom_index_map(symptom_path_list)
    specific_question_mapping_dict, general_question_mapping_dict = (
        read_questions_mapping_info(question_file_path, language))
    symptom_index_dict = symptom_index_dict[language]
    action_mapping_dict = set_action_mapping_dict(
        specific_question_mapping_dict, general_question_mapping_dict, symptom_index_dict)

    symptom_path_list = [
        [symptom_file_path_template.format('chn'), 'chn'],
        [symptom_file_path_template.format('eng'), 'eng']
    ]
    index_symptom_dict, symptom_index_dict = get_symptom_index_map(symptom_path_list)
    (train_dataset, valid_dataset, test_dataset, diagnosis_index_map, index_diagnosis_map) = (
        read_diagnosis_symptom_embedding(
            data_file_path, symptom_info_dict, symptom_index_dict, top_n_diagnosis, icd_digit, diagnosis_weight,
            filter_key=filter_key, diagnosis_lower=diagnosis_lower_threshold, strategy=diagnosis_filter_strategy,
            read_from_cache=True, embedding_size=embedding_size, data_split_strategy=data_split_strategy,
            embedding_type=embedding_type, use_symptom=use_symptom))


    logger.info('data read success')
    symptom_dim = len(train_dataset[0][1]) // 3
    diagnosis_dim = len(train_dataset[0][2])

    envs_kwarg = {
        'first_level_symptom_num': first_level,
        'max_step': episode_max_len,
        'symptom_num': symptom_dim,
        'diagnosis_num': diagnosis_dim,
        'embedding_size': embedding_size,
        'symptom_index_dict': symptom_index_dict,
        'repeat_action_penalty': repeat_action_penalty,
        "action_mapping_dict": action_mapping_dict,
        'screen_model': model,
        "proactive_positive_reveal": proactive_positive_reveal,
        "proactive_negative_reveal": proactive_negative_reveal,
        'language': language,
        "passive_positive_reveal": passive_positive_reveal,
        "passive_negative_reveal": passive_negative_reveal,
        "first_level_positive_factor": first_level_positive_factor,
        "first_level_negative_factor": first_level_negative_factor,
        'time_penalty': time_penalty,
        'max_epoch': None,
        'diff_sum_factor': diff_sum_factor,
        'new_finding_factor': new_finding_factor,
        'positive_reward_factor': positive_reward_factor,
        'negative_reward_factor': negative_reward_factor,
        'device': device
    }

    envs_kwarg_train = {
        'data_pool': train_dataset,
        'random_sample': True,
        **envs_kwarg
    }

    envs_kwarg_eval_train = {
        'data_pool': train_dataset,
        'random_sample': False,
        **envs_kwarg
    }

    envs_kwarg_eval_valid = {
        'data_pool': valid_dataset,
        'random_sample': False,
        **envs_kwarg
    }

    envs_kwarg_eval_test = {
        'data_pool': test_dataset,
        'random_sample': False,
        **envs_kwarg
    }
    vec_env = make_vec_env(PatientEnvironment, n_envs=n_envs, env_kwargs=envs_kwarg_train)

    model_name = args['model_name']

    if model_name == 'dqn':
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=[256, 128, 128],
            action_mapping_dict=action_mapping_dict,
        )
        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_starts=n_envs * episode_max_len * 10,
            buffer_size=819200,
            batch_size=n_envs * episode_max_len,  # envs * 10 epoch * 40 steps * 5 batch per epoch
            policy_kwargs=policy_kwargs,
            learning_rate=LinearSchedule(init_learning_rate),
            device=device,
            verbose=1
        )
    elif model_name == 'ppo':
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[256, 256, 256], vf=[256, 128, 128]),
            symptom_index_dict=symptom_index_dict,
            action_mapping_dict=action_mapping_dict,
            symptom_num=symptom_dim,
            mask_history_action=mask_history_action_bool
        )
        model = PPO(
            SymptomInquireActorCriticPolicy,
            vec_env,
            batch_size=n_envs * update_per_step,
            n_steps=update_per_step,
            policy_kwargs=policy_kwargs,
            learning_rate=LinearSchedule(init_learning_rate),
            ent_coef=entropy_weight,
            vf_coef=value_weight,
            device=device,
            verbose=1
        )
    elif model_name == 'a2c':
        policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=dict(pi=[256, 128, 128], vf=[256]),
            symptom_index_dict=symptom_index_dict,
            action_mapping_dict=action_mapping_dict,
            symptom_num=symptom_dim
        )
        model = A2C(
            SymptomInquireActorCriticPolicy,
            vec_env,
            n_steps=update_per_step,
            learning_rate=LinearSchedule(init_learning_rate),
            policy_kwargs=policy_kwargs,
            vf_coef=value_weight,
            ent_coef=entropy_weight,
            device=device,
            verbose=1
        )
    else:
        raise ValueError('')
    # 2048000约为10个epoch的结果 1024 env, 40 step, 5 batch per epoch, 10 epoch
    callback_interval = update_per_step * n_envs * 5 * 10
    
    vec_env_eval_train = make_vec_env(PatientEnvironment, n_envs=n_envs, env_kwargs=envs_kwarg_eval_train)
    vec_env_eval_valid = make_vec_env(PatientEnvironment, n_envs=n_envs, env_kwargs=envs_kwarg_eval_valid)
    vec_env_eval_test = make_vec_env(PatientEnvironment, n_envs=n_envs, env_kwargs=envs_kwarg_eval_test)
    eval_callback = DiagnosisDiagnosisEvalCallback(
        vec_env_eval_train, vec_env_eval_valid, vec_env_eval_test, model, language, callback_interval, episode_max_len,
        screen_model_save_folder, symptom_index_dict, diagnosis_index_map, filter_key, device
    )
    logger.info('start training')
    model.learn(
        total_timesteps=callback_interval * 4,
        log_interval=3,
        callback=eval_callback
    )
    logger.info('finish')


if __name__ == '__main__':
    main()
