import numpy as np
import torch
import re
import logging
from .ecdai_screening_model import MLP
from .ecdai_doctor_prompt import ecdai_prompt_dict
from .doctor_config import (symptom_file_path, question_file_path, args, chn_icd_mapping_path, eng_icd_mapping_file)
from .ecdai_doctor_util import (read_symptom, set_action_mapping_dict, procedure_dict, eng_index_diagnosis_name_mapping,
                                read_planner, read_questions_mapping_info, construct_question_list, parse_symptom,
                                chn_index_diagnosis_name_mapping)
from .doctor_config import bge_model_file_path
from FlagEmbedding import BGEM3FlagModel


logger = logging.getLogger('evaluation_logger')
environment_language = args['ecdai_language']
maximum_structurize_items = args['ecdai_maximum_structurize_items']
embedding_size_ = args['ecdai_embedding_size']
code_missing_strategy = args['ecdai_code_missing_strategy']
call_llm_retry_time = args['call_llm_retry_time']
planner_device = args['ecdai_planner_device']
embedding_device = args['ecdai_embedding_device']
mask_history_action_flag = args['ecdai_mask_history_action_flag']
ecdai_disease_filter_flag = args['ecdai_disease_filter_flag']
disease_screening_model_path = args['disease_screening_model_path']
planner_file_path = args['planner_file_path']
assert mask_history_action_flag == 1 or mask_history_action_flag == 0
mask_history_action_bool_ = True if mask_history_action_flag == 1 else False

embedding_type = args['ecdai_embedding_type']
assert embedding_type in {0, 1, 2}

symptom_dict = read_symptom(symptom_file_path.format(environment_language))
screen_model_info = torch.load(disease_screening_model_path, map_location=torch.device('cpu'))
index_diagnosis_code_map = screen_model_info['index_diagnosis_map']
if 'chn' == environment_language:
    index_diagnosis_name_map = chn_index_diagnosis_name_mapping(chn_icd_mapping_path, index_diagnosis_code_map)
else:
    assert environment_language == 'eng'
    index_diagnosis_name_map = eng_index_diagnosis_name_mapping(eng_icd_mapping_file, index_diagnosis_code_map)
# icd_specific_dict = read_icd_specific_info(index_diagnosis_code_map, environment_language)

output_size = len(screen_model_info['index_diagnosis_map'])
classification_model_input_size = 4816
classification_model_hidden_info = [128, 128]
disease_classification_model = MLP(classification_model_input_size, classification_model_hidden_info, output_size)
state_dict = screen_model_info['model_state_dict']
disease_classification_model.load_state_dict(state_dict)
disease_classification_model = disease_classification_model.to(planner_device)

symptom_index_dict = screen_model_info['symptom_index_dict']
level_1_list, level_2_list_dict = (
    construct_question_list(symptom_dict,
                            language=environment_language, maximum_questions=maximum_structurize_items))

specific_question_mapping_dict, general_question_mapping_dict = (
    read_questions_mapping_info(question_file_path, environment_language))
action_mapping_dict = (
    set_action_mapping_dict(specific_question_mapping_dict, general_question_mapping_dict, symptom_index_dict,
                            environment_language))


policy_model_actor_net_hidden_info = [256, 256, 256]
policy_model_value_net_hidden_info = [256, 128, 128]
policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=policy_model_actor_net_hidden_info, vf=policy_model_value_net_hidden_info),
    symptom_index_dict=symptom_index_dict,
    action_mapping_dict=action_mapping_dict,
    symptom_num=len(symptom_index_dict[environment_language]),
    mask_history_action=mask_history_action_bool_
)
planner = read_planner(planner_file_path, policy_kwargs, embedding_size_, environment_language, planner_device)
embedding_model = BGEM3FlagModel(bge_model_file_path, device=embedding_device)
embedding_model.max_seq_length = 1024


def screen_prompt_generation(dialogue_string, demographic_flag, history_flag, present_flag,
                             answered_questions, call_llm, client_id, screening_maximum_question,
                             top_n_differential_diagnosis_disease, language):
    assert language == environment_language
    # 注意，此处的previous_answered_screen_questions只计算症状问题
    previous_answered_screen_questions = []
    for item in answered_questions:
        if item[0] != 'S':
            continue
        idx = item.split('-')[1]
        if idx.isdecimal():
            previous_answered_screen_questions.append(int(idx))

    action_array = get_previous_actions(previous_answered_screen_questions)
    symptom_array = obtain_current_symptom_info(dialogue_string, call_llm, client_id, language)
    embedding = obtain_current_embedding(dialogue_string, call_llm, embedding_type, client_id, language)
    question_str, question_candidate_str, screening_decision_flag, question_idx = (
        predict_next_screening_question(symptom_array, embedding, action_array))
    candidate_disease_list, high_risk_diagnoses_str = predict_diagnosis(
        symptom_array, embedding, index_diagnosis_name_map, index_diagnosis_code_map)

    # 超过规定数量强制结束。如果history present 和人口学信息没问到，直接不做screening decision，不受限制
    if len(previous_answered_screen_questions) >= screening_maximum_question:
        screening_decision_flag = True
    if not (history_flag and present_flag and demographic_flag):
        screening_decision_flag = False

    if not demographic_flag and not history_flag:
        question_type = "screening_ask_demographic_medical_history"
        prompt = ecdai_prompt_dict[question_type][language]
        new_screening_question_key = 'S-HD'
        logger.info(f'client: {client_id}, history and demographic info lost, retry')
    elif not demographic_flag:
        question_type = "screening_ask_demographic"
        prompt = ecdai_prompt_dict[question_type][language]
        new_screening_question_key = 'S-D'
        logger.info(f'client: {client_id}, demographic info lost, retry')
    elif not history_flag:
        question_type = "screening_ask_medical_history"
        prompt = ecdai_prompt_dict[question_type][language]
        new_screening_question_key = 'S-H'
        logger.info(f'client: {client_id}, history info lost, retry')
    elif not present_flag:
        question_type = "screening_ask_chief_complaint"
        prompt = ecdai_prompt_dict[question_type][language]
        new_screening_question_key = 'S-P'
        logger.info(f'client: {client_id}, chief compliant info lost, retry')
    elif screening_decision_flag:
        prompt = generate_diagnosis_prompt(candidate_disease_list, top_n_differential_diagnosis_disease,
                                           language).replace('\n', ' ')
        new_screening_question_key = 'S-DIAG'
        logger.info(f'client: {client_id}, generate diagnosis prompt: {prompt}')
    else:
        question_type = "screening_asking_template"
        new_screening_question_key = 'S-{}'.format(question_idx)
        prompt = ecdai_prompt_dict[question_type][language].format(question_str).replace('\n', ' ')
        assert new_screening_question_key not in answered_questions
        logger.info(f'client: {client_id}, generate screen question prompt: {prompt}')

    return (prompt, new_screening_question_key, question_str, question_candidate_str, high_risk_diagnoses_str,
            candidate_disease_list, screening_decision_flag)


def preprocess_diagnosis_list(candidate_disease_list, discard_dict):
    # 本函数的功能有两个
    # 1. 为了防止直接输出一些很严重的疾病把别人吓到。或者是潜在的在未来要做专科版，直接屏蔽一部分疾病。
    # 在前端显示时直接滤过一些疾病。目前的做法是直接删掉所有肿瘤（ICD 10 开头为C-D的相关疾病）
    # 2. 部分初筛的高危疾病可能没有在鉴别诊断阶段有合适对应。这部分疾病可以提前滤掉
    reserve_list_1, discard_count = [], 0
    for item in candidate_disease_list:
        idx = item[0]
        if idx not in discard_dict:
            reserve_list_1.append(item)
        else:
            discard_count += 1

    reserve_list = []
    for item in reserve_list_1:
        if code_missing_strategy == 'end':
            reserve_list.append(item)
        else:
            assert code_missing_strategy == 'skip'
            code = item[1]
            if code in procedure_dict:
                reserve_list.append(item)
    return reserve_list


def get_previous_actions(previous_actions):
    # 生成历史动作流，+1是指最后一个动作是中止提问
    action_array = np.zeros(len(action_mapping_dict) + 1)
    for previous_action in previous_actions:
        # 即便设置了action mask，最终的终止问题还是允许重复做的，确保可以多次做出终止判断
        if previous_action < len(action_mapping_dict):
            action_array[previous_action] = 1
    return action_array


def obtain_current_symptom_info(messages, call_llm, client_id, language):
    _, symptom_info, general_symptom_dict = (
        parse_symptom(messages, "patient_id", symptom_dict, level_1_list, level_2_list_dict, call_llm,
                      client_id, language)
    )
    symptom_index_dict_language = symptom_index_dict[language]
    symptom_array = format_symptom(general_symptom_dict, symptom_index_dict_language, symptom_info).reshape(1, -1)
    return symptom_array


def obtain_current_embedding(dialogue_string, call_llm, embedding_type_, client_id, language):
    if embedding_type_ == 0:
        embedding = np.zeros([1, embedding_size_])
        return embedding

    prompt_key = 'embedding_input_generation_template'
    prompt = ecdai_prompt_dict[prompt_key][language] + dialogue_string
    response = call_llm(prompt)
    try:
        response_str = parse_response(response, language, embedding_type_).replace('。。', '。')
    except:
        response_str = response
        logger.info('ERROR RESPONSE')
    logger.info(f'embedding prompt: {prompt}')
    logger.info(f'embedding response: {response_str}')
    embedding = embedding_model.encode([response_str])['dense_vecs']
    return embedding


def parse_response(response, language, embedding_type_):
    result = {}
    assert embedding_type_ in {1, 2}
    if language == 'eng':
        start_idx = response.find('#Start#')
        end_idx = response.find('#End#')
    else:
        assert language == 'chn'
        start_idx = response.find('#回答开始#')
        end_idx = response.find('#回答结束#')
    response = response[start_idx + 6: end_idx]
    matches = re.findall(r"#(\d+)#: (.+?)(?=#\d+|$)", response, re.DOTALL)
    assert len(matches) == 4
    for match in matches:
        number, content = match
        if number == '1':
            result['基本信息'] = content
        elif number == '2':
            result['既往史'] = content
        elif number == '3':
            result['家族史'] = content
        else:
            assert number == '4'
            result['现病史'] = content

    output = ''
    if "基本信息" in result:
        if language == 'chn':
            key = '基本信息'
        else:
            key = 'Demographic Info'
        output += f'{key}：{result["基本信息"]} \n'
    if "既往史" in result:
        if language == 'chn':
            key = '既往史'
        else:
            key = 'Previous Medical History'
        output += f'{key}：{result["既往史"]} \n'
    if '家族史' in result:
        if language == 'chn':
            key = '家族史'
        else:
            key = 'Family History'
        output += f'{key}：{result["家族史"]}\n'

    if embedding_type_ == 2:
        if '现病史' in result:
            if language == 'chn':
                key = '现病史'
            else:
                key = 'History of Present Illness'
            output += f'{key}：{result["现病史"]}'
    return output


def predict_diagnosis(symptom, embedding, index_diagnosis_name_dict, index_diagnosis_code_dict, top_n=20):
    model_input = np.concatenate([symptom, embedding], axis=1)
    model_input = torch.FloatTensor(model_input).to(planner_device)
    prediction_list = disease_classification_model(model_input).detach().to('cpu').numpy()[0]
    prediction_prob_list = 1 / (1 + np.exp(prediction_list * -1))

    candidate_disease_list = \
        [[idx, index_diagnosis_name_dict[idx], index_diagnosis_code_dict[idx], prediction_prob_list[idx]]
         for idx in index_diagnosis_name_dict]

    candidate_disease_list = sorted(candidate_disease_list, key=lambda x: x[3], reverse=True)
    output_list = []
    for diagnosis in candidate_disease_list:
        if len(output_list) > top_n:
            break
        # index, code, name, prob
        output_list.append([diagnosis[0], diagnosis[2], diagnosis[1], float(diagnosis[3])])

    output_str = '\n'.join(f'{item[1]}: {item[3]:.3f}' for item in output_list)
    return output_list, output_str


def generate_diagnosis_prompt(candidate_disease_list, top_n_differential_diagnosis_disease, language):
    _, top_diagnosis, _, _ = candidate_disease_list[0]
    prompt_type = 'screening_decision_template'
    prompt = ecdai_prompt_dict[prompt_type][language].format(top_diagnosis)

    candidate_list = []
    for idx in range(1, len(candidate_disease_list)):
        if idx >= top_n_differential_diagnosis_disease:
            break
        candidate_list.append(candidate_disease_list[idx][1])
    if language == 'chn':
        if len(candidate_list) > 0:
            candidate_str = '也请告诉用户可能存在风险较高的其它疾病还包括：' + '，'.join(candidate_list) + '\n'
        else:
            candidate_str = ''
    else:
        if len(candidate_list) > 0:
            candidate_str = 'Please inform the user there are some other high risk diseases: ' + ','.join(candidate_list) + '\n'
        else:
            candidate_str = ''
    prompt = prompt + candidate_str
    return prompt


def format_symptom(general_symptom_dict, symptom_index_mapping, symptom_info):
    symptom_info_array = np.zeros(len(symptom_index_mapping) * 3)
    symptom_info_array[0::3] = 1
    for key in general_symptom_dict:
        assert key in symptom_index_mapping
        idx = symptom_index_mapping[key][0]
        if general_symptom_dict[key] == 'YES':
            symptom_info_array[idx * 3: (idx + 1) * 3] = [0, 0, 1]
        elif general_symptom_dict[key] == 'NO':
            symptom_info_array[idx * 3: (idx + 1) * 3] = [0, 1, 0]
        else:
            assert general_symptom_dict[key] == 'NA'
    for first_level in symptom_info:
        for group in symptom_info[first_level]:
            for factor in symptom_info[first_level][group]:
                key = (first_level + ' ' + group + ' ' + factor).lower().replace('  ', ' ')
                assert key in symptom_index_mapping
                idx = symptom_index_mapping[key][0]
                if symptom_info[first_level][group][factor] == 'YES':
                    symptom_info_array[idx * 3: (idx + 1) * 3] = [0, 0, 1]
                elif symptom_info[first_level][group][factor] == 'NO':
                    symptom_info_array[idx * 3: (idx + 1) * 3] = [0, 1, 0]
                else:
                    assert symptom_info[first_level][group][factor] == 'NA'
    return symptom_info_array


def predict_next_screening_question(symptom, embedding, previous_actions, top_n=5):
    model_input = np.concatenate([symptom, [[0]], embedding, previous_actions[np.newaxis, :]], axis=1)
    model_input = torch.FloatTensor(model_input)
    data = planner.get_distribution(model_input)
    probs = list(data.distribution.probs.detach().cpu().numpy()[0])
    action_dict = planner.action_mapping_dict
    action_list = ([[index, action_dict[index][0]] for index in action_dict] +
                   [[len(action_dict), 'screen_inquiry_end']])
    action_list = sorted(action_list, key=lambda x: x[0])
    return_question_list = []
    for prob, action in zip(probs, action_list):
        probability = float(prob)
        return_question_list.append(action + [probability])

    return_question_list = sorted(return_question_list, key=lambda x: x[2], reverse=True)
    question_index, question, _ = return_question_list[0]

    # action idx的最后一位是初筛决策action
    screening_decision_flag = return_question_list[0][0] == (len(return_question_list) - 1)
    question_candidate = []
    for return_question in return_question_list[1:]:
        if len(question_candidate) >= top_n:
            break
        question_candidate.append(return_question[1])
    question_candidate_str = '\n'.join(question_candidate)
    return question, question_candidate_str, screening_decision_flag, question_index


