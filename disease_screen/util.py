import gymnasium as gym
import os
from stable_baselines3.common.monitor import Monitor
from typing import Any, Callable, Dict, Optional, Type, Union
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env
import csv
from itertools import islice
import pickle
import numpy as np
from disease_screen_logger import logger
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LinearSchedule:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        if progress_remaining > 0.9:
            lr = (1 - progress_remaining) * 10 * self.initial_value
        elif progress_remaining > 0.6:
            lr = (1 - (0.9 - progress_remaining) / 3 * 9) * self.initial_value
        else:
            lr = (1 - (0.6 - progress_remaining) / 6 * 10) * 0.1 * self.initial_value
        return lr


def make_vec_env(
        env_callable: Union[str, Callable[..., gym.Env]],
        n_envs: int = 1,
        monitor_dir: Optional[str] = None,
        seed: Optional[int] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        vec_env_cls: Optional[Type[Union[DummyVecEnv,]]] = None,
) -> VecEnv:
    """
    Create a wrapped, monitored ``VecEnv``.
    By default, it uses a ``DummyVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_callable: either the env ID, the env class or a callable returning an env
    :param seed:
    :param n_envs: the number of environments you wish to have in parallel
    :param monitor_dir: Path to a folder where the monitor files will be saved.
        If None, no file will be written, however, the env will still be wrapped
        in a Monitor wrapper to provide additional information about training.
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``VecEnv`` class constructor. Default: None.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            env_kwargs['rank'] = rank
            env_kwargs['total_env'] = n_envs

            env = env_callable(**env_kwargs)
            env = _patch_env(env)

            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv
    vec_env = vec_env_cls([make_env(i) for i in range(n_envs)])
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


def set_action_mapping_dict(specific_question_mapping_dict, general_question_mapping_dict, symptom_index_dict):
    # 规定action空间的前面len(symptom_index_dict)位与特征是一一对应的，后面的则是一对多的action
    action_mapping = dict()
    assert len(specific_question_mapping_dict) == len(symptom_index_dict)
    for question in specific_question_mapping_dict:
        key = specific_question_mapping_dict[question]
        index, parent_idx = symptom_index_dict[key]
        action_mapping[index] = [question, [], set()]
        action_mapping[index][1].append([index, key])
        action_mapping[index][2].add(parent_idx)

    for idx in range(len(action_mapping)):
        assert idx in action_mapping

    action_idx = len(action_mapping)
    for question in general_question_mapping_dict:
        key_set = general_question_mapping_dict[question]
        # 后面两个set分别代指影响的symptom, trigger symptom,，general question均无trigger action
        action_mapping[action_idx] = [question, [], set()]
        for key in key_set:
            symptom_index, parent_symptom_idx = symptom_index_dict[key]
            action_mapping[action_idx][1].append([symptom_index, key])
            action_mapping[action_idx][2].add(parent_symptom_idx)
        action_idx += 1
    return action_mapping


def primary_full_calculate(label, pred, top_k):
    data_size = len(label)
    top_k_hit_count = 0
    for i in range(data_size):
        pair_list = []
        for j in range(len(pred[i])):
            pair_list.append([label[i, j], pred[i, j]])

        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        for j in range(top_k):
            if pair_list[j][0] == 1:
                top_k_hit_count += 1
    top_k_hit = top_k_hit_count / data_size
    return top_k_hit


def get_data(file_path, valid_key_set, visit_type_filter_key, data_source_filter_key, max_data_size):
    data_list = []
    if data_source_filter_key == 'srrsh':
        language = 'chn'
    else:
        assert data_source_filter_key in {'mimic_iv', 'mimic_iii', 'mimic'}
        language = 'eng'

    data_dict = read_context(file_path, visit_type_filter_key, data_source_filter_key,
                             max_data_size=max_data_size)
    for unified_id in data_dict.keys():
        if unified_id in valid_key_set:
            data_list += [[unified_id, data_dict[unified_id], language]]
    data_list = sorted(data_list, key=lambda x: x[0])
    return data_list


def read_context(file_path, visit_type_filter, data_source_filter, max_data_size=-1):
    failed_count, admission_failed_count, outpatient_failed_count = 0, 0, 0
    success_count, admission_success_count, outpatient_success_count = 0, 0, 0
    data_dict = dict()
    with (open(file_path, 'r', encoding='utf-8-sig', newline='') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            data_source, visit_type, patient_visit_id, outpatient_record, admission_record = line[:5]
            unified_id = data_source + '-' + visit_type + '-' + patient_visit_id

            if 0 < max_data_size < len(data_dict):
                break

            if visit_type_filter != 'all':
                if visit_type_filter != visit_type:
                    continue
            if data_source_filter != 'all':
                if data_source_filter not in data_source:
                    continue
            if visit_type == 'outpatient':
                # assert admission_record == 'None'
                if len(outpatient_record) <= 20:
                    # logger.info('unified_id: {}, outpatient record too short, skip.'.format(unified_id))
                    failed_count += 1
                    outpatient_failed_count += 1
                else:
                    success_count += 1
                    outpatient_success_count += 1
                data_dict[unified_id] = outpatient_record
            else:
                assert visit_type == 'hospitalization'
                if len(admission_record) <= 50:
                    # logger.info('unified_id: {}, admission record too short, skip. '.format(unified_id))
                    failed_count += 1
                    admission_failed_count += 1
                else:
                    success_count += 1
                    admission_success_count += 1
                    data_dict[unified_id] = admission_record
    logger.info('failed count (data too short): {}, admission: {}, outpatient： {}'.format(
        failed_count, admission_failed_count, outpatient_failed_count))
    logger.info('success count: {}, admission: {}, outpatient： {}'.format(
        success_count, admission_success_count, outpatient_success_count))
    return data_dict


def get_model(max_token, model_split, model_id, enforce_eager, gpu_utilization, cache_folder, vllm_type):
    if len(cache_folder) > 0:
        model_path = str(os.path.join(cache_folder, model_id))
    else:
        model_path = model_id
    max_model_len, tp_size = max_token, model_split

    logger.info('local model, model id: {}'.format(model_path))
    llm_model = LLM(
        model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_utilization,
        trust_remote_code=True,
        enforce_eager=enforce_eager,
        quantization=vllm_type
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        padding_side='left',
        trust_remote_code=True,
    )
    return llm_model, tokenizer


def get_local_llm_model_tokenizer(model_id, max_output_token, llm_cache, max_token, tensor_split,
                                  enforce_eager, gpu_utilization, vllm_type):
    if model_id == 'qwen/Qwen2-72B-Instruct':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'qwen/Qwen2___5-32B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'qwen/Qwen2___5-72B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'qwen/Qwen2-72B-Instruct-GPTQ-Int4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'qwen/Qwen2-72B-Instruct-GPTQ-Int8':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'qwen/Qwen2-7B-Instruct':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'LLM-Research/Meta-Llama-3___1-70B-Instruct-GPTQ-INT4':
        sampling_strategy = SamplingParams(max_tokens=max_output_token)
    elif model_id == 'tclf90/Yi-1___5-34B-Chat-16K-GPTQ-Int4':
        sampling_strategy = SamplingParams(temperature=0.95, max_tokens=max_output_token)
    elif model_id == 'tclf90/Qwen2-72B-Instruct-GPTQ-Int3':
        sampling_strategy = SamplingParams(temperature=0.95, max_tokens=max_output_token)
    else:
        assert model_id == 'ZhipuAI/glm-4-9b-chat'
        sampling_strategy = SamplingParams(max_tokens=max_output_token)

    logger.info('model id: {}'.format(model_id))
    llm_model, llm_tokenizer = get_model(cache_folder=llm_cache, max_token=max_token, enforce_eager=enforce_eager,
                                         model_split=tensor_split, model_id=model_id, gpu_utilization=gpu_utilization,
                                         vllm_type=vllm_type)

    def call_llm(prompt):
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            prompt = llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = llm_model.generate(prompt, sampling_strategy)
            result = outputs[0].outputs[0].text
        else:
            prompt_list = []
            for prompt_ in prompt:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_}
                ]
                prompt_ = llm_tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompt_list.append(prompt_)
            outputs = llm_model.generate(prompt_list, sampling_strategy)
            result = []
            for output in outputs:
                text = output.outputs[0].text
                result.append(text)
        return result

    def tokenizer_tokenize(input_info):
        result = llm_tokenizer(input_info)['input_ids']
        return result

    def tokenizer_reverse_tokenize(input_info):
        result = llm_tokenizer.decode(input_info)
        return result
    return call_llm, tokenizer_tokenize, tokenizer_reverse_tokenize


def parse_data_symptom(train_dataset, valid_dataset, test_dataset, save_key, file_path):
    data_dict = dict()
    for key, dataset in zip(['train', 'valid', 'test'], [train_dataset, valid_dataset, test_dataset]):
        for item in dataset:
            unified_id, symptom, diagnosis, embedding, modify_time = item
            positive_symptom = symptom[2::3]
            positive_symptom_num = np.sum(positive_symptom)
            data_dict[unified_id] = [key, positive_symptom_num, modify_time]
    pickle.dump(data_dict, open(file_path.format(save_key), 'wb'))
    logger.info('symptom_num saved')
