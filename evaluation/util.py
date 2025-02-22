import json
import re
import os
import csv
import openai
from scipy.stats import norm
import numpy as np
import traceback
from itertools import islice
from evaluation_logger import logger
from evaluation_config import call_llm_retry_time, eng_icd_mapping_file, model_info_json, chn_icd_mapping_path


endpoint_info = json.load(open(model_info_json, 'r', encoding='utf-8-sig'))

rerank_prompt_dict = {
    'chn': '你现在拿到了一个疾病列表清单，内容是:\n{}。请你判断，列表中的每一个疾病是否可以和{}被视为同一疾病。'
           '你的回答必须是YES或NO，不可以是别的值。注意，只有两种疾病明确是同一疾病，或是列表中的疾病是{}的某个子类型时，'
           "输出格式应当遵循如下格式（#ANSWER START#是开始标记，用于定位真正的回答开始位置，请严格输出#ANSWER START#）"
           '\n#ANSWER START#\n#1#: 疾病名称\n#2#: 疾病名称\n#3#: 疾病名称\n#4#: 疾病名称\n#5#: 疾病名称\n',
    'eng': "You now have a list of diseases as follows:\n{}. Please determine whether each disease in the list can "
           "be considered the same as {} (true diagnosis). Your answer must be either YES or NO, without any other "
           "values. Note that you should only respond with YES if the two diseases are exactly the same, or the "
           "disease in the list is a subtype of {}. Your output format should follow (#ANSWER START# is the start "
           "token of the answer, it is used to navigate the real start of the answer, "
           "please strictly output #ANSWER START#):\n#ANSWER START#\n#1#: Disease Name\n#2#: Disease Name\n"
           "#3#: Disease Name\n #4#: Disease Name\n#5#: Disease Name\n"
}

diagnosis_prompt_dict = {
    'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据和病人的对话历史，推测他有什么疾病。你们的对话历史是：\n{}\n\n。"
           "你需要基于这一堆话，列出造成病人本次入院的最可能的5种疾病，注意，越高风险的疾病应当排在越前面，"
           "你不能输出重复的疾病，疾病的粒度应当尽可能细致。\n你可以按照你的习惯思考，"
           "但在思考完毕后，输出格式应当遵循如下格式（#ANSWER START#是开始标记，用于定位真正的回答开始位置，请严格输出#ANSWER START#）：\n"
           "#ANSWER START#\n#1#: 疾病名称\n#2#: 疾病名称\n#3#: 疾病名称\n#4#: 疾病名称\n#5#: 疾病名称\n",
    'eng': "Please assume you are a doctor conducting a medical consultation with a patient. "
           "Based on the conversation history with the patient, you need to infer their possible diseases. "
           "The conversation history is:\n{}\n\n. Based on this dialogue, you are required to list the five most "
           "likely diseases causing the patient's current hospitalization. Note that higher-risk diseases should "
           "be listed first. You must not output duplicate diseases, and the granularity of the diseases should "
           "be as detailed as possible.\nYour output format should follow (#ANSWER START# is the start token of the "
           "answer, it is used to navigate the real start of the answer, please strictly output #ANSWER START#):"
           "\n#ANSWER START#\n#1#: Disease Name\n#2#: Disease Name\n#3#: Disease Name\n#4#: Disease Name\n"
           "#5#: Disease Name\n"
}


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


def content_reorganize(content):
    dialogue_list, dialogue_string, content = [], '', content[1:]
    assert len(content) % 2 == 0
    for i in range(len(content)):
        if i % 2 == 0:
            assert content[i]['role'] == 'assistant'
            flag_idx = content[i]['fullResponse'].find('<SEC-1-START>')
            if flag_idx != -1:
                end_idx = content[i]['fullResponse'].find('<SEC-1-END>')
                assert end_idx > 0
                start_idx = flag_idx + len('<SEC-1-START>')
                turn_content = content[i]['fullResponse'][start_idx: end_idx]
                turn_dialogue_str = f'第{i // 2 + 1}轮，医生说：\n {turn_content} \n'
                dialogue_string = dialogue_string + turn_dialogue_str
                dialogue_list.append(turn_dialogue_str)
            else:
                assert i == 0
                turn_dialogue_str = f'第{i//2+1}轮，医生说：\n {content[i]["fullResponse"]} \n'
                dialogue_string = dialogue_string + turn_dialogue_str
                dialogue_list.append(turn_dialogue_str)
        else:
            assert content[i]['role'] == 'user'
            turn_dialogue_str = f'第{i//2+1}轮，病人说：\n {content[i]["fullResponse"]} \n\n'
            dialogue_string = dialogue_string + turn_dialogue_str
            dialogue_list.append(turn_dialogue_str)

    assert len(content) >= 2
    if len(content) < 3:
        # content len < 3即为第一轮
        assert '<AFFILIATED-INFO-START>' not in content[-2]['fullResponse']
        # initialize
        # 如果是第一轮对话，则重新初始化state
        state = {
            'diagnosis_state': {},
            'answered_questions': [],
            'asking_question': None,
            'end_flag': None,
            'diagnosis_list': None,
            'question_key': None,
            'high_risk_diagnoses_str': None,
            'screen_question_str': None,
            'screen_question_candidate_str': None,
        }
    else:
        # 如果并非第一轮对话，则基于上一轮对话的state进行初始化
        start_idx = (content[-2]['fullResponse'].find('<AFFILIATED-INFO-START>') +
                     len('<AFFILIATED-INFO-START>'))
        end_idx = content[-2]['fullResponse'].find('<AFFILIATED-INFO-END>')
        state = json.loads(content[-2]['fullResponse'][start_idx: end_idx])

    turn_num = len(content) // 2
    # 如果上一轮的问题并没有在本轮中被正面回复，则视为没有询问，不列入previous screen actions
    previous_questions = state['answered_questions']
    answered_questions = list()
    for action in previous_questions:
        answered_questions.append(action)
    return dialogue_string, dialogue_list, state, turn_num, answered_questions


def parse_previous_interact(dialogue_str: str, call_llm, client_id):
    # 取消之前的正面回答的解析设计，不管有没有正面回答，都完整的跑一次解析流程然后问下一个问题
    success_flag = False
    failed_time = 0
    direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag = (
        False, False, False, False, False, False)
    while not success_flag:
        try:
            prompt = ("请假设你是一位医生，正在对一名用户进行诊断对话。你们的对话记录是：\n{}\n"
                      "你需要用YES或NO回答以下问题，但我们不允许同时回答YES/NO，你只能选一个回复："
                      "问题 1：用户是否回答了对话中的最后一个问题？。注意，用户如果说他不知道，应当视为回答了问题，回复YES。"
                      "但如果用户对问题的内容不理解而进行提问，应当视为没有回答，回复NO。\n"
                      "问题 2：在最后一轮对话中，用户是否询问了与医学相关的问题？如果用户的回复中包含与医学相关的任何问题，包括但不限于药物的服用方式，"
                      "疾病的症状等信息，对既往问题中的内容不理解而提问，请判定为YES，如果患者没有提及任何医学相关的问题，则判定为NO。)\n"
                      "问题 3：在最后一轮对话中，用户是否询问或发起了与医学无关的问题或指令？如果用户的发起了与医学无关的问题，请返回YES。"
                      "如果用户发起了指令让模型做一些与医学无关的任务，请返回YES。"
                      "问题 4：用户是否已经在这些对话中介绍了自己的性别和年龄？\n"
                      "问题 5：用户是否已经在这些对话中介绍了自己的既往病史信息？\n"
                      "问题 6：用户是否已经在这些对话中介绍了自己当前的症状（任何现有症状或者体征）？\n"
                      "请返回NO,若用户的回复不涉及与医学无关的问题，也不涉及与医学无关的指令，请返回NO。请根据以下格式作答:"
                      "#问题 1#：YES/NO\n"
                      "#问题 2#：YES/NO\n"
                      "#问题 3#：YES/NO\n"
                      "#问题 4#：YES/NO\n"
                      "#问题 5#：YES/NO\n"
                      "#问题 6#：YES/NO\n"
                      ).format(dialogue_str)
            response = call_llm(prompt)

            direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag = (
                parse_patient_response_intention(response, target_length=6))
            success_flag = True
        except:
            failed_time += 1
            logger.info(u"Error Trance {}".format(traceback.format_exc()))
            if failed_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, parse previous interact failed')
                break
    return direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag


def response_medical_question(message, dialogue_string):
    # 取消之前的正面回答的解析设计，不管有没有正面回答，都完整的跑一次解析流程然后问下一个问题
    prompt = ("请假设你是一位医生，正在对一名用户进行诊断对话。在上一轮对话中，用户的回复是：{}。\n"
              "既往的完整对话是：\n{}\n。请你仔细思考，并简明扼要的回答用户在上一轮中提出的医学问题。回复不要太长，直接回答问题即可。"
              "请不要反问用户问题，直接给出回答。若用户的问题很难精准回答，给出宽泛的回答即可。针对患者提出的非医学问题和诉求，"
              "请不要做任何回复。如果用户的问题是要求你诊断疾病，请不要回答，回复：请回答诊断系统的问题，我们会适时给出诊断建议。"
              ).format(message[-1]['fullResponse'], dialogue_string)
    return prompt


def response_non_medical_question(message):
    # 取消之前的正面回答的解析设计，不管有没有正面回答，都完整的跑一次解析流程然后问下一个问题
    prompt = ("请假设你是一位医生，正在对一名用户进行诊断对话。\n"
              "请告诉用户其诉求中和医学无关的问题不会被回答。回复不要太长。"
              ).format(message[-1]['fullResponse'])
    return prompt


def get_already_terminate_prompt():
    prompt = ("请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户用户已经终止，不会再就任何信息进行回复。"
              "注意：回复不要太长，不要透露你在扮演一名医生，直接告诉用户这个信息即可，不要回复其他任何内容")
    return prompt


def get_data_insufficient_prompt():
    prompt = ("请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户用户已经终止，不会再就任何信息进行回复，"
              "原因是用户无法提供请求的信息或用户主动要求终止对话。"
              "注意：回复不要太长，不要透露你在扮演一名医生，直接告诉用户这个信息即可，不要回复其他任何内容")
    return prompt


def parse_patient_response_intention(response, target_length):
    result_str = response.strip()
    start_index = result_str.find('#问题 1#')
    if start_index == -1:
        logger.info('format illegal')
        raise ValueError('Invalid result')

    result_str_list = result_str[start_index:].split('\n')
    if len(result_str_list) < target_length:
        # logger.info('Illegal result: {}'.format(result_str))
        logger.info('count illegal, too short')
        raise ValueError('Invalid result')
    else:
        result_str_list = result_str_list[:target_length]

    result_list = []
    for i, result in enumerate(result_str_list):
        flag_1 = '#问题 {}#'.format(i + 1) in result
        flag_2 = ('NO' in result and 'YES' not in result) or \
                 ('NO' not in result and 'YES' in result)
        if not (flag_1 and flag_2):
            if not flag_1:
                logger.info('count illegal')
            if not flag_2:
                logger.info('format illegal')
            raise ValueError('Invalid result')
        result_list.append(False if 'NO' in result and 'YES' not in result else True)
    return result_list


def read_chinese_icd():
    code_name_mapping = dict()
    with open(chn_icd_mapping_path, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            category_code, category_name, icd_code, disease_name, sub_disease_name = \
                line[5], line[6], line[7], line[8], line[10]
            icd_code = icd_code.lower().replace('.', '')
            category_code = category_code.lower()
            if category_code not in code_name_mapping:
                code_name_mapping[category_code] = category_name
            if icd_code not in code_name_mapping:
                code_name_mapping[icd_code] = disease_name
            if icd_code not in code_name_mapping:
                code_name_mapping[icd_code] = []
    return code_name_mapping


def read_english_icd():
    icd_dict = dict()
    with open(eng_icd_mapping_file, 'r', encoding='utf-8-sig', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            icd, version, name = line
            icd_dict[icd.lower()] = name
    return icd_dict


eng_icd_mapping_dict = read_english_icd()
chn_icd_mapping_dict = read_chinese_icd()


def llm_react_mimic_diagnosis_match_parse(history, selected_data, call_llm, target_llm_name):
    oracle_diagnosis = selected_data['oracle_diagnosis'].strip().lower()
    first_diagnosis = oracle_diagnosis.split('$$$$$')[0]
    icd_dict = eng_icd_mapping_dict
    if first_diagnosis[:3] not in icd_dict or first_diagnosis[:4] not in icd_dict:
        logger.info('ERROR')
        return {3: 100, 4: 100}

    diagnosis_name_3 = icd_dict[first_diagnosis[:3]]
    diagnosis_name_4 = icd_dict[first_diagnosis[:4]]
    start_key, end_key = '<AFFILIATED-INFO>', "</AFFILIATED-INFO>"
    start_idx = history[-2]['full_response'].find(start_key) + len(start_key)
    end_idx = history[-2]['full_response'].find(end_key)
    candidate_list = json.loads(history[-2]['full_response'][start_idx:end_idx])['candidate_disease_list']
    rank_dict = candidate_mimic_analysis(candidate_list, diagnosis_name_3, diagnosis_name_4, call_llm, target_llm_name)
    return rank_dict


def candidate_mimic_analysis(candidate_list, diagnosis_name_3, diagnosis_name_4, call_llm, target_llm_name):
    candidate_str = ''
    for idx, candidate in enumerate(candidate_list):
        candidate_str += f'#{idx + 1}#: {candidate}\n'

    rank_dict = {}
    for true_diagnosis, digit in zip((diagnosis_name_3, ), (3, )):
        prompt = rerank_prompt_dict['eng'].format(candidate_str, true_diagnosis, true_diagnosis)
        logger.info(f'parse rank prompt: \n: {prompt}')
        rank = parse_rank(call_llm, prompt, target_llm_name)
        logger.info(f'candidate list: {candidate_list}, true diagnosis: {true_diagnosis}, rank: {rank}')
        rank_dict[digit] = rank
    return rank_dict


def llm_react_srrsh_diagnosis_match_parse(history, selected_data, call_llm, target_llm_name):
    icd_dict = chn_icd_mapping_dict
    oracle_diagnosis_str = selected_data['table_diagnosis'].strip().lower()
    first_diagnosis_str = oracle_diagnosis_str.split('$$$$$')[0]

    oracle_diagnosis_icd = selected_data['oracle_diagnosis'].strip().lower()
    first_diagnosis_icd = oracle_diagnosis_icd.split('$$$$$')[0]

    start_key, end_key = '<AFFILIATED-INFO>', "</AFFILIATED-INFO>"
    end_idx = history[-2]['full_response'].find(end_key)
    start_idx = history[-2]['full_response'].find(start_key) + len(start_key)
    candidate_list = json.loads(history[-2]['full_response'][start_idx:end_idx])['candidate_disease_list']
    result_dict = candidate_srrsh_analysis(candidate_list, first_diagnosis_str, first_diagnosis_icd, icd_dict,
                                           call_llm, target_llm_name)
    return result_dict


def candidate_srrsh_analysis(candidate_list, first_diagnosis_str, first_diagnosis_icd, icd_dict, call_llm,
                             target_llm_name):
    candidate_str = ''
    for idx, candidate in enumerate(candidate_list):
        candidate_str += f'#{idx + 1}#: {candidate}\n'

    if len(candidate_list) != 5 or len(first_diagnosis_str) == 0:
        logger.info('candidate info illegal: {}'.format(candidate_list))
        name_rank = 100
    else:
        prompt = rerank_prompt_dict['chn'].format(candidate_str, first_diagnosis_str, first_diagnosis_str)
        name_rank = parse_rank(call_llm, prompt, target_llm_name)
    # 需要可以后补
    # if first_diagnosis_icd[0:4] in icd_dict:
    #     diagnosis_str = icd_dict[first_diagnosis_icd[0:4]]
    #     prompt = prompt_start + prompt_template.format(diagnosis_str, diagnosis_str)
    #     rank_4 = parse_rank(call_llm, prompt, target_llm_name)
    # else:
    #     rank_4 = 100
    #
    # if first_diagnosis_icd[0:3] in icd_dict:
    #     diagnosis_str = icd_dict[first_diagnosis_icd[0:3]]
    #     prompt = prompt_start + prompt_template.format(diagnosis_str, diagnosis_str)
    #     rank_3 = parse_rank(call_llm, prompt, target_llm_name)
    # else:
    #     rank_3 = 100
    logger.info(f'candidate list: {candidate_list}, true diagnosis: {first_diagnosis_str}, name_rank: {name_rank}, '
                f'rank_4: Skip, rank_3: Skip')
    return {'name': name_rank, 'rank_4': 100, 'rank_3': 100}


def parse_rank(call_llm, prompt, target_llm_name):
    rank = 100
    success_flag, failure_time = False, 0
    while not success_flag:
        try:
            response = call_llm(prompt, target_llm_name)
            logger.info(f'rank parse response: {response}')
            assert '#ANSWER START#' in response
            start_idx = response.find('#ANSWER START#') + len('#ANSWER START#')
            response_reserve = response[start_idx:]
            start_index = response_reserve.find('#1#')
            assert start_index >= 0
            response_list = response_reserve[start_index:].split('\n')
            assert len(response) >= 5

            for idx, content in enumerate(response_list[:5]):
                content = content.lower()
                assert ('yes' in content and 'no' not in content) or ('yes' not in content and 'no' in content)
                if 'yes' in content:
                    rank = idx + 1
                    break
            success_flag = True
        except Exception:
            failure_time += 1
            logger.info(u"Error Trance {}".format(traceback.format_exc()))
            if failure_time >= 5:
                success_flag = True
    return rank


def ecdai_diagnosis_match_parse(history, selected_data, data_source):
    start_key, end_key = '<AFFILIATED-INFO>', "</AFFILIATED-INFO>"
    end_idx = history[-2]['full_response'].find(end_key)
    start_idx = history[-2]['full_response'].find(start_key) + len(start_key)
    candidate_list = json.loads(history[-2]['full_response'][start_idx:end_idx])['candidate_disease_list']
    predict_diagnosis_list = [item[1] for item in candidate_list]

    oracle_diagnosis = selected_data['oracle_diagnosis'].strip().lower()
    true_diagnosis_icd = oracle_diagnosis.split('$$$$$')[0]

    rank_dict = {3: 100, 4: 100}
    for digit in (3, 4):
        for idx, candidate in enumerate(predict_diagnosis_list):
            if candidate[:digit] == true_diagnosis_icd[:digit]:
                rank_dict[digit] = idx + 1
                break
    return rank_dict


def moderator_judge_request(history, selected_data, call_llm, doctor_type, target_llm_name):
    data_source = selected_data['source']
    if 'ecdai' not in doctor_type:
        if 'mimic' in data_source:
            rank_dict = llm_react_mimic_diagnosis_match_parse(history, selected_data, call_llm, target_llm_name)
        else:
            assert 'srrsh' in data_source
            rank_dict = llm_react_srrsh_diagnosis_match_parse(history, selected_data, call_llm, target_llm_name)
    else:
        rank_dict = ecdai_diagnosis_match_parse(history, selected_data, data_source)
    return rank_dict


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


def read_history_text_data(folder, filter_key):
    file_list = os.listdir(folder)
    data_list = []
    for file in file_list:
        file_path = os.path.join(folder, file)
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        data_list = data_list + data

    return_data = dict()
    error_count = 0
    for item in data_list:
        key_1, key_2 = item[0], item[2]
        if filter_key != 'all':
            if filter_key not in key_1:
                continue

        if 'mimic' in key_1:
            language = 'eng'
        else:
            assert 'srrsh' in key_1
            language = 'chn'
        content_with_hpi, content_without_hpi = '', ''
        if "基本信息" in item[1]:
            if language == 'chn':
                key = '基本信息'
            else:
                key = 'demographic info'
            content_without_hpi += f'{key}：{item[1]["基本信息"]}\n'
        else:
            error_count += 1
        if "既往史" in item[1]:
            if language == 'chn':
                key = '既往史'
            else:
                key = 'Previous Medical History'
            content_without_hpi += f'{key}：{item[1]["既往史"]}\n'
        else:
            error_count += 1
        if '家族史' in item[1]:
            if language == 'chn':
                key = '家族史'
            else:
                key = 'Family History'
            content_without_hpi += f'{key}：{item[1]["家族史"]}\n'
        else:
            error_count += 1
        if '现病史' in item[1]:
            if language == 'chn':
                key = '现病史'
            else:
                key = 'History of Present Illness'
            content_with_hpi = content_without_hpi + f'{key}：{item[1]["现病史"]}'
        else:
            error_count += 1
        return_data[key_1] = [key_1, key_2, content_with_hpi, content_without_hpi]
    print(f'error count: {error_count}')
    return return_data


def read_conversation(dialogue, language):
    assert len(dialogue) % 2 == 0
    dialogue_str = ''
    for i, utterance in enumerate(dialogue[:-2]):
        turn_num = i // 2 + 1
        if i % 2 == 0:
            assert utterance['role'] == 'doctor'
            if language == 'eng':
                prefix_str = f'Turn {turn_num}, Doctor Said:\n'
            else:
                assert language == 'chn'
                prefix_str = f'第{turn_num}轮，医生说:\n'
        else:
            assert utterance['role'] == 'patient'
            if language == 'eng':
                prefix_str = f'Turn {turn_num}, Patient Said:\n'
            else:
                prefix_str = f'第{turn_num}轮，病人说:\n'
        dialogue_str += prefix_str + utterance['show_response'] + '\n'
    return dialogue_str


def read_eval_data(folder, llm_name, turn_num, *filter_key_list):
    folder = os.path.join(folder, llm_name)
    data_dict = dict()
    file_list = os.listdir(folder)
    for file in file_list:
        file_path = os.path.join(folder, file)
        json_file = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        data_source = json_file['data']['source']
        visit_type = json_file['data']['visit_type']
        patient_visit_id = json_file['data']['patient_visit_id']
        dialogue = json_file['dialogue']
        if 'mimic' in data_source:
            language = 'eng'
        else:
            assert 'srrsh' in data_source
            language = 'chn'
        dialogue_str = read_conversation(dialogue, language)
        unified_id = data_source + '-' + visit_type + '-' + patient_visit_id

        flag = True

        for filter_key in filter_key_list:
            if filter_key not in file_path:
                flag = False
        if file[:2] != turn_num:
            flag = False
        if flag:
            data_dict[unified_id] = dialogue_str, json_file
    return data_dict


def non_streaming_call_llm(model_name, content):
    client = openai.OpenAI(
        api_key=endpoint_info[model_name]['api_key'],
        base_url=endpoint_info[model_name]['url'],
    )

    completion = client.chat.completions.create(
        model=endpoint_info[model_name]['model_name'],
        messages=[{'role': 'assistant', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': content}],
        # max_completion_tokens=256,
        # temperature=0.0,
    )
    message = completion.choices[0].message.content
    return message


def parse_result_disease_think(response):
    assert '#ANSWER START#' in response
    start_idx = response.find('#ANSWER START#') + len('#ANSWER START#')
    response_reserve = response[start_idx:]
    response_list = response_reserve.split('\n')
    contents = []
    for s in response_list:
        match = re.match(r"#\d+#: (.+)", s)
        if match:
            contents.append(match.group(1).strip())
    assert len(contents) >= 5
    return contents[:5]


def recall_ci_eval_dict(key_list, data_dict):
    for key in key_list:
        logger.info(f'key_name: {key}, length: {len(data_dict[key])}')
        rank_list = np.array(data_dict[key])
        top_1 = np.sum(rank_list < 2) / len(rank_list)
        top_2 = np.sum(rank_list < 3) / len(rank_list)
        top_3 = np.sum(rank_list < 4) / len(rank_list)
        top_4 = np.sum(rank_list < 5) / len(rank_list)
        top_5 = np.sum(rank_list < 6) / len(rank_list)

        top_1_ci = ci_calculate(rank_list < 2)
        top_2_ci = ci_calculate(rank_list < 3)
        top_3_ci = ci_calculate(rank_list < 4)
        top_4_ci = ci_calculate(rank_list < 5)
        top_5_ci = ci_calculate(rank_list < 6)
        logger.info('top 1: {:.4f}, 95% CI: {:.4f}'.format(top_1, top_1_ci))
        logger.info('top 2: {:.4f}, 95% CI: {:.4f}'.format(top_2, top_2_ci))
        logger.info('top 3: {:.4f}, 95% CI: {:.4f}'.format(top_3, top_3_ci))
        logger.info('top 4: {:.4f}, 95% CI: {:.4f}'.format(top_4, top_4_ci))
        logger.info('top 5: {:.4f}, 95% CI: {:.4f}'.format(top_5, top_5_ci))
        logger.info('')


def eval_performance(target_folder, filter_failed, max_size):
    file_list = os.listdir(target_folder)
    file_path_list = [os.path.join(target_folder, file) for file in file_list]
    file_path_list = sorted(file_path_list)[:max_size]
    logger.info(f'len result_dict: {len(file_path_list)}')
    performance_dict = dict()
    success_count = 0
    for file_path in file_path_list:
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        if filter_failed:
            last_utterance = data['original_data'][1]['dialogue'][-2]['full_response']
            start_index = last_utterance.find('<AFFILIATED-INFO>') + len("<AFFILIATED-INFO>")
            end_index = last_utterance.find('</AFFILIATED-INFO>')
            candidate = json.loads(last_utterance[start_index:end_index])['candidate_disease_list']
            if len(candidate) == 0:
                continue
        success_count += 1
        rank_dict = data['rank_dict']
        for k, v in rank_dict.items():
            if k not in performance_dict.keys():
                performance_dict[k] = []
            performance_dict[k].append(v)

    key_list = sorted(performance_dict.keys())
    recall_ci_eval_dict(key_list, performance_dict)
    logger.info('success_count: {:.4f}'.format(success_count))


def ci_calculate(data):
    p_hat = np.mean(data)
    n = len(data)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    z = norm.ppf(0.975)  # 95% confidence level
    return z * se


def parse_disease_idx(diagnosis, category_list):
    idx = -1
    if len(diagnosis) < 3 or not diagnosis[1:3].isdigit():
        print(f'error, diagnosis: {diagnosis}')
        return idx

    first_diagnosis = diagnosis[:3]
    for item in category_list:
        category_idx, key, category_token_start, number_start, category_token_end, number_end = item
        if first_diagnosis[0] > category_token_start:
            if first_diagnosis[0] > category_token_end:
                continue
            else:
                if int(first_diagnosis[1:3]) <= number_end:
                    idx = category_idx
        elif first_diagnosis[0] == category_token_start:
            if first_diagnosis[0] == category_token_end:
                if number_start <= int(first_diagnosis[1:3]) <= number_end:
                    idx = category_idx
            else:
                if number_start < int(first_diagnosis[1:3]):
                    idx = category_idx
        else:
            continue
    return idx