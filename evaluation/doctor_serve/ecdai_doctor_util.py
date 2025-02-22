import copy
import re
import csv
import json
import torch
import threading
import logging
from itertools import islice
from gymnasium.spaces import Box, Discrete
from .ecdai_doctor_prompt import ecdai_prompt_dict
from .ecdai_screening_model import SymptomInquireActorCriticPolicy
from .doctor_config import differential_diagnosis_procedure_file_path, args
from .util import parse_patient_response_intention


logger = logging.getLogger('evaluation_logger')
call_llm_retry_time = args['call_llm_retry_time']


def read_symptom(symptom_path):
    symptom_dict = dict()
    with open(symptom_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            symptom, factor_group, factor = line
            if symptom not in symptom_dict:
                symptom_dict[symptom] = dict()
            if len(factor_group) > 0:
                assert len(factor_group) > 0
                if factor_group not in symptom_dict[symptom]:
                    symptom_dict[symptom][factor_group] = []
                symptom_dict[symptom][factor_group].append(factor)
    return symptom_dict


def read_planner(planner_path, policy_kwargs, embedding_size, language, device):
    symptom_num = len(policy_kwargs['symptom_index_dict'][language])
    action_mapping_dict = policy_kwargs['action_mapping_dict']
    policy_weight = torch.load(planner_path, map_location=torch.device('cpu'), weights_only=True)
    action_space = Discrete(len(policy_kwargs['action_mapping_dict']) + 1)
    observation_space = Box(low=-10000, high=10000,
                            shape=[symptom_num * 3 + 1 + embedding_size + len(action_mapping_dict) + 1])
    policy_model = SymptomInquireActorCriticPolicy(
        action_space=action_space,
        observation_space=observation_space,
        lr_schedule=LinearSchedule(0.001),
        **policy_kwargs,
    )
    policy_model.load_state_dict(policy_weight)
    policy_model = policy_model.to(device)
    return policy_model


class LinearSchedule:
    def __init__(self, initial_value: float):
        self.initial_value = initial_value

    def __call__(self, progress_remaining: float) -> float:
        if progress_remaining > 0.9:
            lr = (1-progress_remaining) * 10 * self.initial_value
        elif progress_remaining > 0.6:
            lr = (1 - (0.9 - progress_remaining) / 3 * 9) * self.initial_value
        else:
            lr = (1 - (0.6 - progress_remaining) / 6 * 10) * 0.1 * self.initial_value
        return lr


def set_action_mapping_dict(specific_question_mapping_dict, general_question_mapping_dict, symptom_index_dict,
                            language):
    # 规定action空间的前面len(symptom_index_dict)位与特征是一一对应的，后面的则是一对多的action
    action_mapping = dict()
    assert len(specific_question_mapping_dict) == len(symptom_index_dict[language])
    for question in specific_question_mapping_dict:
        key = specific_question_mapping_dict[question]
        symptom_index, parent_symptom_idx = symptom_index_dict[language][key]
        assert symptom_index not in specific_question_mapping_dict
        action_mapping[symptom_index] = [question, [], set()]
        action_mapping[symptom_index][1].append([symptom_index, key])
        action_mapping[symptom_index][2].add(parent_symptom_idx)

    for idx in range(len(action_mapping)):
        assert idx in action_mapping

    action_idx = len(action_mapping)
    for question in general_question_mapping_dict:
        key_set = general_question_mapping_dict[question]
        # 后面两个set分别代指影响的symptom, trigger symptom,，general question均无trigger action
        action_mapping[action_idx] = [question, [], set()]
        for key in key_set:
            symptom_index, parent_symptom_idx = symptom_index_dict[language][key]
            action_mapping[action_idx][1].append([symptom_index, key])
            action_mapping[action_idx][2].add(parent_symptom_idx)
        action_idx += 1
    return action_mapping


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


def construct_question_list(symptom_dict, language, maximum_questions):
    level_one_symptom_list = sorted(list(symptom_dict.keys()))
    level_1_list = []

    split_num = len(level_one_symptom_list) // maximum_questions \
        if len(level_one_symptom_list) % maximum_questions == 0 \
        else len(level_one_symptom_list) // maximum_questions + 1

    for i in range(split_num):
        start_index = i * maximum_questions
        if i < len(level_one_symptom_list) - 1:
            end_index = (i + 1) * maximum_questions
        else:
            assert i == len(level_one_symptom_list) - 1
            end_index = len(level_one_symptom_list) - 1

        symptom_list = level_one_symptom_list[start_index: end_index]
        prompt_1 = construct_level_one_question_list(symptom_list, language)
        level_1_list.append([prompt_1, symptom_list])

    level_2_list_dict = {}
    for key in symptom_dict:
        secondary_questions_dict = symptom_dict[key]
        if len(secondary_questions_dict) > 0:
            level_2_list_dict[key.lower()] = construct_level_two_question_list(
                key, secondary_questions_dict, language, maximum_questions)
        else:
            level_2_list_dict[key.lower()] = None
    return level_1_list, level_2_list_dict


def construct_level_two_question_list(symptom, factor_dict, language, maximum_questions):
    if language == 'eng':
        prompt_prefix = ('Please answer whether the below factors exist when the give symptom: {} is existing. \n'
                         'There are three possible answers for each factor. YES means a factor exists or ever '
                         'exits based on the given context, NO means a '
                         'factor does not exist, NA means a factor is not mentioned in the context\n'
                         'PLEASE NOTE:\n'
                         '1. "deny" or "denies" a symptom means NO. \n'
                         '2. a factor need to be treated as NA when it is not mentioned\n'
                         '3. a factor need to be treated as exist (YES) when it directly relates or cause the current '
                         'hospital admission, if a factor exists but does not cause the current admission, '
                         'Please treats the symptom as NA\n '
                         '4. fever means patient body temperature larger or equal than 99 F\n').format(symptom.upper())
    else:
        assert language == 'chn'
        prompt_prefix = (
            '请回答以下因素在给定症状：{} 存在时是否存在。'
            '如果症状中包括对发热的判断，患者体温大于37.2度或文本中明确包含近期发热的字样时，可视为患者存在发热症状。\n'
            .format(symptom))

    factor_group_list = sorted(list(factor_dict.keys()))

    full_list = []
    for factor_group in factor_group_list:
        factor_list = sorted(factor_dict[factor_group])
        for item in factor_list:
            full_list.append([factor_group, item])

    prompt_list = []
    split_num = len(full_list) // maximum_questions \
        if len(full_list) % maximum_questions == 0 else len(full_list) // maximum_questions + 1
    for i in range(split_num):
        start_index = i * maximum_questions
        if i < split_num - 1:
            end_index = (i + 1) * maximum_questions
        else:
            end_index = len(full_list)
        batch = full_list[start_index:end_index]
        prompt = '' + prompt_prefix
        for j, item in enumerate(batch):
            if language == 'chn':
                prompt += '#{}#: {}{}{}\n'.format(j + 1, symptom, item[0], item[1])
            else:
                assert language == 'eng'
                prompt += '#{}#: {} {} {}\n'.format(j + 1, symptom.strip(), item[0].strip(), item[1].strip())

        if language == 'eng':
            prompt = prompt + ('\nPlease answer the question strictly according to the following format, '
                               'without any other content. You MUST pick up one to answer (Yes or No or NA).'
                               'Please answer the question in order with the format #Number#: YES/NO/NA,'
                               'and without any explain.\n')
        else:
            assert language == 'chn'
            prompt += (
                '\n'
                '请在回复的第一个字符就开始回答，不要做任何补充说明。\n'
                '请严格按照以下格式和数字顺序回答： #问题编号# YES/NO/NA。从第一个症状到最后一个症状逐一回答，不要有任何遗漏。\n'
                '除了YES NO NA的回答外，不要添加任何其他内容，也无需对答案进行任何解释。\n'
                '请在回答前先空一行，并严格按照每个症状占据一行的方式回复，即在每个症状的答案后面加一个换行符。\n'
                '每个症状有三种可能的答案。YES 表示根据给定的上下文因素存在或曾经存在。NO '
                '表示症状不存在。NA 表示上下文中未提及该因素或信息不足无法判断。\n请注意：病人否认一个症状表示 NO。'
                '当某因素直接相关或导致当前入院时，应视为存在（YES），如果某因素存在但未导致当前入院，请将其视为 NA。'
                '每个症状只能归类为三个情况中的一个，不能同时输出两个答案。\n'
            )
        # for j, item in enumerate(batch):
        #     prompt += '#{}#: YES/NO/NA\n'.format(j+1)
        prompt_list.append([prompt, batch])
    return prompt_list


def construct_level_one_question_list(symptom_list, language):
    if language == 'eng':
        prompt = 'Please answer whether the below symptoms are existing. \n' \
                 'There are three possible answers for each symptom. YES means a symptom exists, NO means a ' \
                 'symptom does not exist, NA means a symptom is not mentioned in the context\n' \
                 'PLEASE NOTE:\n' \
                 '1. "deny" or "denies" a symptom means NO. \n' \
                 '2. a factor need to be treated as NA when it is not mentioned\n' \
                 '3. a factor need to be treated as exist (YES) when it directly relates or cause the current ' \
                 'hospital admission, if a factor exists but does not cause the current admission, ' \
                 'Please treats the symptom as NA\n ' \
                 '4. fever means patient body temperature larger or equal than 99 F\n'
    else:
        assert language == 'chn'
        prompt = ('请回答以下症状是否存在。'
                  '患者体温大于37.2度或文本中明确包含近期发烧的字样时，可视为患者存在发烧症状。'
                  '\n')

    for i, item in enumerate(symptom_list):
        prompt += '#{}#: {}\n'.format(i + 1, item)

    if language == 'eng':
        prompt = prompt + ('\nPlease answer the question strictly according to the following format, '
                           'without any other content. You MUST pick up one to answer (Yes or No or NA).'
                           'Please answer the question in order with the format #Number#: YES/NO/NA,'
                           'and without any explain.\n')
    else:
        assert language == 'chn'
        prompt += (
            '\n'
            '请在回复的第一个字符就开始回答，不要做任何补充说明。\n'
            '请严格按照以下格式和数字顺序回答： #问题编号# YES/NO/NA。从第一个症状到最后一个症状逐一回答，不要有任何遗漏。\n'
            '除了YES NO NA的回答外，不要添加任何其他内容，也无需对答案进行任何解释。\n'
            '请在回答前先空一行，并严格按照每个症状占据一行的方式回复，即在每个症状的答案后面加一个换行符。\n'
            '每个症状有三种可能的答案。YES 表示根据给定的上下文因素存在或曾经存在。NO '
            '表示症状不存在。NA 表示上下文中未提及该因素或信息不足无法判断。\n请注意：病人否认一个症状表示 NO。'
            '当某因素直接相关或导致当前入院时，应视为存在（YES），如果某因素存在但未导致当前入院，请将其视为 NA。'
            '每个症状只能归类为三个情况中的一个，不能同时输出两个答案。\n'
        )
    # for i, item in enumerate(symptom_list):
    #     prompt += '#{}#, #{}#: YES/NO/NA\n'.format(i+1, item)
    return prompt


def content_reorganize(content, phase, language):
    dialogue_list, dialogue_string = [], ''
    assert len(content) % 2 == 0
    for i in range(len(content)):
        flag_idx, end_idx = (
            content[i]['full_response'].find('<RESPONSE>'), content[i]['full_response'].find('</RESPONSE>'))

        assert end_idx > 0 and flag_idx >= 0
        start_idx = flag_idx + len('<RESPONSE>')
        turn_content = content[i]['full_response'][start_idx: end_idx]

        if i % 2 == 0:
            if language == 'chn':
                turn_dialogue_str = f'第{i // 2 + 1}轮，医生说：\n {turn_content} \n'
            else:
                assert language == 'eng'
                turn_dialogue_str = f'Turn #{i // 2 + 1}, Doctor Said：\n {turn_content} \n'
            dialogue_string = dialogue_string + turn_dialogue_str
            assert content[i]['role'] == 'doctor'
        else:
            if language == 'chn':
                turn_dialogue_str = f'第{i//2+1}轮，病人说：\n {turn_content} \n\n'
            else:
                assert language == 'eng'
                turn_dialogue_str = f'Turn #{i//2+1}, Patient Said：\n {turn_content} \n\n'
            dialogue_string = dialogue_string + turn_dialogue_str
            assert content[i]['role'] == 'patient'
        dialogue_list.append(turn_dialogue_str)

    if len(content) < 1:
        # content len < 1即为第一轮
        # initialize
        # 如果是第一轮对话，则重新初始化state
        if phase == 'DIAGNOSIS':
            screen_flag = 0
        else:
            assert phase == 'SCREEN' or phase == 'ALL'
            screen_flag = 1
        state = {
            'diagnosis_state': {},
            'answered_questions': [],
            'end_flag': 0,
            'screen_flag': screen_flag,
            'candidate_disease_list': None,
            'question_key': None,
            'high_risk_diagnoses_str': None,
            'screen_question_str': None,
            'screen_question_candidate_str': None,
        }
    else:
        # 如果并非第一轮对话，则基于上一轮对话的state进行初始化
        start_idx = (content[-2]['full_response'].find('<AFFILIATED-INFO>') +
                     len('<AFFILIATED-INFO>'))
        end_idx = content[-2]['full_response'].find('</AFFILIATED-INFO>')
        state = json.loads(content[-2]['full_response'][start_idx: end_idx])

    turn_num = len(content) // 2
    # 如果上一轮的问题并没有在本轮中被正面回复，则视为没有询问，不列入previous screen actions
    previous_questions = state['answered_questions']
    answered_questions = list()
    for action in previous_questions:
        answered_questions.append(action)
    return dialogue_string, dialogue_list, state, turn_num, answered_questions


def ecdai_parse_previous_interact(dialogue_str: str, call_llm, client_id, language):
    # 取消之前的正面回答的解析设计，不管有没有正面回答，都完整的跑一次解析流程然后问下一个问题
    success_flag = False
    failed_time = 0
    direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag = (
        False, False, False, False, False, False)

    key = 'last_interaction_parse_template'
    prompt_template = ecdai_prompt_dict[key][language]
    while not success_flag:
        try:
            prompt = prompt_template.format(dialogue_str)
            response = call_llm(prompt)

            direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag = (
                parse_patient_response_intention(response, target_length=6, language=language))
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, parse previous interact failed')
                break
    return direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag


def load_diagnosis_dict(language):
    logger.info('The diagnosis part of ecdai only support Chinese')
    data_dict = json.load(open(differential_diagnosis_procedure_file_path, 'r', encoding='utf-8-sig'))
    structured_procedure_dict = dict()
    for key in data_dict:
        procedure_list = data_dict[key]
        structured_procedure_dict[key] = list()
        for procedure in procedure_list:
            question_dict, start_index = parse_questions(procedure[1], language)
            legal_flag = structure_validity_test(question_dict, start_index)
            assert legal_flag
            structured_procedure_dict[key].append([procedure[0], procedure[1], question_dict, start_index])
    return structured_procedure_dict


def parse_questions(text, language):
    question_dict = dict()
    start_index = text.find('Start') + 5
    text = text[start_index:]
    pattern = r"#QUESTION\s+#\d+#"
    # Using findall to find all occurrences in the text
    matches = re.findall(pattern, text)
    for i, _ in enumerate(matches):
        start_index = text.find(matches[i]) + len(matches[i])
        if i < len(matches) - 1:
            end_index = text.find(matches[i + 1])
        else:
            end_ = text[start_index:]
            if 'End' in end_:
                end_index = end_.find('End') + start_index
            else:
                end_index = len(text)
        condition_str = text[start_index: end_index]
        question_content = parse_condition(condition_str, language)

        question = matches[i].strip().split('#')
        question_index = int(question[-2])
        question_dict[question_index] = question_content
    return question_dict, int(matches[0].strip().split('#')[-2])


def parse_condition(text, language):
    assert "- Yes:" in text and "- No:" in text
    condition = text[: text.find('- Yes:')]
    yes = text[text.find('- Yes:') + len("- Yes:"): text.find('- No:')]
    no = text[text.find('- No:') + len("- No:"):]

    # nested question error
    assert '- Yes:' not in yes and '- No:' not in yes
    assert '- Yes:' not in no and '- No:' not in no
    condition_obj = Condition(condition, yes, no, language)
    return condition_obj


class Condition(object):
    def __init__(self, condition, yes, no, language):
        self.condition = condition
        self.language = language
        self.yes = self.parse(yes)
        self.no = self.parse(no)

    def yes_next(self):
        return self.yes

    def no_next(self):
        return self.no

    def parse(self, text):
        if self.language == 'chn':
            confirm_word = '你确诊'
            exclude_word = '你没有'
        else:
            assert self.language == 'eng'
            confirm_word = 'YOU HAVE'
            exclude_word = 'YOU DON\'T HAVE'
        if confirm_word in text:
            return text, 'DIAGNOSIS CONFIRM'
        elif exclude_word in text:
            return text, "DIAGNOSIS EXCLUDE"
        else:
            assert 'PROCEED' in text
            question = text.strip().split('#')
            question_index = int(question[-2])
            return question_index, 'NAVIGATE'


def structure_validity_test(question_dict, start_index):
    # 测试包括几个要求
    # 从第一个问题开始，必须能够遍历所有节点
    # 无环
    # 必须有路径能够得到确诊，也必须有路径能够得到排除
    # 终止决策必须是诊断
    path_list = []
    legal_flag, exclude_flag, confirm_flag, traversed_node_set = (
        traverse_node(question_dict, start_index, [], path_list))

    legal_flag = legal_flag and exclude_flag and confirm_flag and len(traversed_node_set) == len(question_dict)
    return legal_flag


def traverse_node(question_dict, index, current_path, path_list):
    legal_flag, exclude_flag, confirm_flag = True, False, False
    traversed_node_set = set()
    traversed_node_set.add(index)
    for item in current_path:
        traversed_node_set.add(item)

    new_path = copy.deepcopy(current_path)
    if index in new_path:
        legal_flag = False
        return legal_flag, exclude_flag, confirm_flag, traversed_node_set
    new_path.append(index)

    if index not in question_dict:
        return False, exclude_flag, confirm_flag, traversed_node_set

    node = question_dict[index]
    yes_next, yes_next_type = node.yes
    no_next, no_next_type = node.no
    if yes_next_type == 'DIAGNOSIS EXCLUDE':
        exclude_flag = True
    elif yes_next_type == 'DIAGNOSIS CONFIRM':
        confirm_flag = True
    else:
        if yes_next_type == 'NAVIGATE' and isinstance(yes_next, int):
            legal_flag, exclude_flag, confirm_flag, new_traversed_node_set = (
                traverse_node(question_dict, yes_next, new_path, path_list))
            traversed_node_set = traversed_node_set.union(new_traversed_node_set)
        else:
            legal_flag = False
            return legal_flag, exclude_flag, confirm_flag, traversed_node_set

    if no_next_type == 'DIAGNOSIS EXCLUDE':
        exclude_flag = True
        path_list.append(new_path)
    elif no_next_type == 'DIAGNOSIS CONFIRM':
        confirm_flag = True
        path_list.append(new_path)
    else:
        if no_next_type == 'NAVIGATE' and isinstance(no_next, int):
            legal_flag, exclude_flag, confirm_flag, new_traversed_node_set = (
                traverse_node(question_dict, no_next, new_path, path_list))
            traversed_node_set = traversed_node_set.union(new_traversed_node_set)
        else:
            legal_flag = False
            return legal_flag, exclude_flag, confirm_flag, traversed_node_set
    return legal_flag, exclude_flag, confirm_flag, traversed_node_set


def read_icd_specific_info(file_path, index_diagnosis_code_map):
    # 这个函数的设计中，给出了第四位的疾病诊断的下属细则，是为了实践中的输出做准备的
    code_name_list_dict = dict()
    with (open(file_path, 'r', encoding='utf-8-sig', newline='') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            category_code, category_name, icd_code, disease_name, sub_disease_name = \
                line[5], line[6], line[7], line[8], line[10]
            icd_code = icd_code.replace('.', '').lower()
            category_code = category_code[0:3].lower()
            if icd_code not in code_name_list_dict:
                code_name_list_dict[icd_code] = {'description': disease_name, 'specific_disease': []}
            if category_code not in code_name_list_dict:
                code_name_list_dict[category_code] = {'description': category_name, 'specific_disease': []}
            code_name_list_dict[icd_code]['specific_disease'].append(sub_disease_name)

    idx_name_diagnosis_dict = dict()
    for index in index_diagnosis_code_map:
        code = index_diagnosis_code_map[index]
        if code in code_name_list_dict:
            idx_name_diagnosis_dict[index] = code_name_list_dict[code]
        # 邵逸夫医院的编码到第四位和医保局的并不完全对应，如果出现了无法匹配的问题就回退一位
        elif code[0:3] in code_name_list_dict:
            idx_name_diagnosis_dict[index] = code_name_list_dict[code[0:3]]
        else:
            logger.info(f'code: {code} mapping failed')
    return idx_name_diagnosis_dict



def eng_index_diagnosis_name_mapping(file_path, index_diagnosis_code_map):
    code_mapping = dict()
    with (open(file_path, 'r', encoding='utf-8-sig', newline='') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            icd, version, name = line
            icd_code = icd.lower().replace('.', '')
            code_mapping[icd_code] = name

    index_diagnosis_name_map = dict()
    illegal_set = set()
    for index in index_diagnosis_code_map:
        code = index_diagnosis_code_map[index]
        category_code = code[0:3]
        if code in code_mapping:
            name = code_mapping[code]
        else:
            name = ''
            illegal_set.add(category_code)
        index_diagnosis_name_map[index] = name
    logger.info(f'illegal set length: {len(illegal_set)}')
    # 注意，此处的mapping failed不代表icd code弃用，他只是声明一下
    for key in illegal_set:
        logger.info(f'illegal category code: {key}')
    return index_diagnosis_name_map


def chn_index_diagnosis_name_mapping(file_path, index_diagnosis_code_map):
    category_code_name_mapping = dict()
    code_name_mapping = dict()
    code_sub_name_mapping = dict()
    with (open(file_path, 'r', encoding='utf-8-sig', newline='') as f):
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            category_code, category_name, icd_code, disease_name, sub_disease_name = \
                line[5], line[6], line[7], line[8], line[10]
            icd_code = icd_code.lower().replace('.', '')
            category_code = category_code.lower()
            if category_code not in category_code_name_mapping:
                category_code_name_mapping[category_code] = category_name
            if icd_code not in code_name_mapping:
                code_name_mapping[icd_code] = disease_name
            if icd_code not in code_sub_name_mapping:
                code_sub_name_mapping[icd_code] = []
            code_sub_name_mapping[icd_code].append(sub_disease_name)

    index_diagnosis_name_map = dict()
    illegal_set = set()
    for index in index_diagnosis_code_map:
        code = index_diagnosis_code_map[index]
        category_code = code[0:3]
        if code in code_name_mapping:
            name = code_name_mapping[code]
        elif category_code in category_code_name_mapping:
            name = category_code_name_mapping[category_code]
        else:
            name = ''
            illegal_set.add(category_code)
        index_diagnosis_name_map[index] = name
    logger.info(f'illegal set length: {len(illegal_set)}')
    # 注意，此处的mapping failed不代表icd code弃用，他只是声明一下
    for key in illegal_set:
        logger.info(f'illegal category code: {key}')
    return index_diagnosis_name_map



def parse_symptom(content, unified_id, symptom_dict_, question_one_list, question_two_list_dict, call_llm,
                  client_id, language):
    dialogue_list = []
    if language == 'eng':
        init_prompt = "Please assume you are a senior doctor, given the below conversation history:\n\n"
    else:
        assert language == 'chn'
        init_prompt = "请假设您是一位高级医生，基于以下对话：\n\n"

    symptom_info, general_symptom_dict = initialize_symptom_info(symptom_dict_)

    logger.info('start parsing symptom')
    parse_level_1_symptom(unified_id, content, init_prompt, general_symptom_dict, symptom_info, question_one_list,
                          call_llm, dialogue_list, client_id)

    parse_level_2_symptom(symptom_info, question_two_list_dict, general_symptom_dict, unified_id, content,
                          init_prompt, dialogue_list, call_llm, client_id)
    logger.info('symptom parsing end')
    return dialogue_list, symptom_info, general_symptom_dict


def parse_level_2_symptom(symptom_info, question_two_list_dict, general_symptom_dict, unified_id, content_str,
                          init_prompt, dialogue_list, call_llm, client_id):
    level_2_info_list = []
    for symptom in symptom_info:
        second_level_prompt_list = question_two_list_dict[symptom]
        if second_level_prompt_list is None:
            continue
        if general_symptom_dict[symptom] != 'YES':
            continue
        logger.info(f'client: {client_id} start parsing: {unified_id}, level 2 symptom {symptom}')
        for (second_level_prompt, symptom_list) in second_level_prompt_list:
            level_2_info_list.append([symptom, second_level_prompt, symptom_list])

    logger.info(f'client: {client_id}, level 2 info parsing list length: {len(level_2_info_list)}')
    threads = []
    for (symptom, second_level_prompt, symptom_list) in level_2_info_list:
        symptom = symptom.lower()
        thread = threading.Thread(
            target=parse_level_2_symptom_detail,
            args=(content_str, symptom, symptom_info, second_level_prompt, symptom_list,
                  init_prompt, dialogue_list, call_llm, client_id))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()



def parse_level_1_symptom(unified_id, content, init_prompt, general_symptom_dict, symptom_info, question_one_list,
                          call_llm, dialogue_list, client_id):
    logger.info(f'client: {client_id}, start parsing: {unified_id}, level 1 symptom')
    info_list = []
    for sub_level_1_prompt, sub_level_1_symptom_list in question_one_list:
        prompt = init_prompt + content + '\n' + sub_level_1_prompt
        info_list.append([prompt, sub_level_1_symptom_list])
    logger.info(f'client: {client_id}, level 1 info parsing list length: {len(info_list)}')
    threads = []
    for (prompt, sub_level_1_symptom_list) in info_list:
        thread = threading.Thread(
            target=parse_level_1_symptom_detail,
            args=(prompt, sub_level_1_symptom_list, general_symptom_dict, symptom_info, call_llm, dialogue_list,
                  client_id))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return symptom_info, general_symptom_dict


def parse_level_1_symptom_detail(prompt, sub_level_1_symptom_list, general_symptom_dict, symptom_info,
                                 call_llm, dialogue_list, client_id):
    success_flag = False
    result_str = ''
    failure_time = 0
    while not success_flag:
        try:
            result_str = call_llm(prompt)
            dialogue_list.append([prompt, result_str])
            result_list = result_preprocess(result_str, len(sub_level_1_symptom_list), client_id)

            for result, symptom in zip(result_list, sub_level_1_symptom_list):
                symptom = symptom.lower()
                assert symptom in general_symptom_dict
                if 'NO' in result:
                    general_symptom_dict[symptom] = 'NO'
                    if symptom_info[symptom] is not None:
                        for factor_group in symptom_info[symptom]:
                            for factor in symptom_info[symptom][factor_group]:
                                symptom_info[symptom][factor_group][factor] = 'NO'
                elif 'NA' in result:
                    general_symptom_dict[symptom] = 'NA'
                    if symptom_info[symptom] is not None:
                        for factor_group in symptom_info[symptom]:
                            for factor in symptom_info[symptom][factor_group]:
                                symptom_info[symptom][factor_group][factor] = 'NA'
                else:
                    if 'YES' not in result:
                        logger.info('Answer illegal')
                        raise ValueError('')
                    general_symptom_dict[symptom.lower()] = 'YES'
                    logger.info(f'client: {client_id}, {symptom} confirm')
            success_flag = True
        except:
            # 部分文本过长的情况下可能导致解析不完整，这里如果出现错误超过重试次数就直接跳出
            failure_time += 1
            logger.info(f'client: {client_id} failed, failure time: {failure_time}')
            if failure_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, exceed maximum retry time ({call_llm_retry_time}), exit')
                logger.info(f'client: {client_id}, prompt: {prompt}, response: {result_str}')
                success_flag = True


def parse_level_2_symptom_detail(content, symptom, symptom_info, second_level_prompt, symptom_list, init_prompt,
                                 dialogue_list, call_llm, client_id):
    prompt = init_prompt + content + '\n' + second_level_prompt
    success_flag = False
    failure_time = 0
    while not success_flag:
        result_str = ''
        try:
            result_str = call_llm(prompt)
            result_list = result_preprocess(result_str, len(symptom_list), client_id)
            dialogue_list.append([prompt, result_str, symptom])
            for result, (factor_group, factor) in zip(result_list, symptom_list):
                factor_group, factor = factor_group.lower(), factor.lower()
                assert factor_group in symptom_info[symptom] and factor in symptom_info[symptom][factor_group]
                if 'NO' in result:
                    symptom_info[symptom][factor_group][factor] = 'NO'
                elif 'NA' in result:
                    symptom_info[symptom][factor_group][factor] = 'NA'
                else:
                    if 'YES' not in result:
                        logger.info(f'client: {client_id}, answer illegal')
                        raise ValueError('')
                    symptom_info[symptom.lower()][factor_group.lower()][factor.lower()] = 'YES'
                    logger.info(f'client: {client_id}, {symptom} {factor_group} {factor} confirm')
            success_flag = True
        except:
            failure_time += 1
            logger.info(f'client: {client_id} failed, failure time: {failure_time}')
            if failure_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, prompt: {prompt}, response: {result_str}')
                logger.info(f'client: {client_id}, exceed maximum retry time ({call_llm_retry_time}), exit')
                success_flag = True


def initialize_symptom_info(symptom_dict_):
    symptom_info, general_symptom_dict = dict(), dict()
    for key in symptom_dict_:
        symptom_info[key.lower()] = dict()
        general_symptom_dict[key.lower()] = "NA"
        for factor_group in symptom_dict_[key]:
            symptom_info[key.lower()][factor_group.lower()] = dict()
            for factor in symptom_dict_[key][factor_group]:
                symptom_info[key.lower()][factor_group.lower()][factor.lower()] = "NA"
    return symptom_info, general_symptom_dict


def result_preprocess(result_str, target_length, client_id):
    result_str = result_str.strip()
    start_index = result_str.find('#1#')
    if start_index == -1:
        # 之所以有这个设置，是因为01 AI的大模型特别喜欢空一格，原因不明。也只有这个模型有这个现象
        start_index = result_str.find('#1 #')
    if start_index == -1:
        logger.info(f'client: {client_id}, format illegal')
        raise ValueError('Invalid result')

    # 滤过一些空行
    result_list_ = result_str[start_index:].split('\n')
    result_list = []
    for item in result_list_:
        if len(item) > 3:
            result_list.append(item)
    if len(result_list) < target_length:
        # logger.info('Illegal result: {}'.format(result_str))
        logger.info(f'client: {client_id}, count illegal, too short')
        raise ValueError('Invalid result')

    if len(result_list) >= target_length:
        result_list = result_list[:target_length]

    for i, result in enumerate(result_list):
        flag_2 = ('NO' in result and 'NA' not in result and 'YES' not in result) or \
                 ('NO' not in result and 'NA' in result and 'YES' not in result) or \
                 ('NO' not in result and 'NA' not in result and 'YES' in result)
        flag_1 = '#{}#'.format(i + 1) in result or '#{} #'.format(i + 1) in result
        if not (flag_1 and flag_2):
            if not flag_1:
                logger.info(f'client: {client_id}, count illegal')
            if not flag_2:
                logger.info(f'client: {client_id}, format illegal')
            raise ValueError('Invalid result')
    return result_list


procedure_dict = load_diagnosis_dict(language='chn')
