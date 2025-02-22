import json
import logging
import traceback
import re
from .react_doctor_prompt import llm_prompt_dict
from .doctor_config import call_llm_retry_time
from .util import (parse_patient_response_intention, parse_thought, parse_question, parse_diagnosis)

logger = logging.getLogger('evaluation_logger')


def get_response(prompt, call_llm, client_id, question_type):
    # 取消之前的正面回答的解析设计，不管有没有正面回答，都完整的跑一次解析流程然后问下一个问题
    success_flag = False
    failed_time = 0
    response = "None"
    while not success_flag:
        try:
            response = call_llm(prompt)
            success_flag = True
        except:
            failed_time += 1
            trace = traceback.format_exc()
            if failed_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, parse previous interact failed, trace: {trace}')
                break
    logger.info(f'client: {client_id}, {question_type}, prompt: {prompt}')
    return response


def parse_result_disease(response):
    response_list = response.split('\n')
    contents = []
    for s in response_list:
        match = re.match(r"#\d+#: (.+)", s)
        if match:
            contents.append(match.group(1).strip())
    assert len(contents) == 5
    return contents


def generate_high_risk_diseases(dialogue_string, call_llm, client_id, language):
    question_type = 'screening_decision_template'
    prompt = llm_prompt_dict[question_type][language].format(dialogue_string)
    success_flag = False
    failed_time = 0
    response, diagnosis_list, thought = '', [], ''
    while not success_flag:
        try:
            response = get_response(prompt, call_llm, client_id, question_type)
            thought = parse_thought(response, language)
            diagnosis_list = parse_result_disease(response)
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, high risk diseases parse failed')
                break
    return response, thought, diagnosis_list, question_type


def screen_information_sufficiency_analyze(dialogue_string, call_llm, client_id, language):
    question_type = 'screening_sufficiency_analysis_template'
    prompt = llm_prompt_dict[question_type][language].format(dialogue_string)
    success_flag = False
    failed_time = 0
    sufficient_thought = "none"
    flag = False
    while not success_flag:
        try:
            response = get_response(prompt, call_llm, client_id, question_type)
            flag, = parse_patient_response_intention(response, target_length=1, language=language)
            sufficient_thought = parse_thought(response, language)
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, sufficiency analysis failed')
                break
    sufficient_flag = 1 if flag else 0
    logger.info(f'client: {client_id}, screen sufficiency test, result: {sufficient_flag}')
    return sufficient_thought, sufficient_flag


def screen_question_generation(dialogue_string, call_llm, client_id, language):
    if len(dialogue_string) == 0:
        question_type = 'opening_question'
        prompt = llm_prompt_dict[question_type][language]
        question = get_response(prompt, call_llm, client_id, question_type)
        thought = 'first question, no thought'
    else:
        question_type = 'screening_asking_template'
        prompt = llm_prompt_dict[question_type][language].format(dialogue_string)

        success_flag = False
        failed_time = 0
        thought = 'none'
        question = 'none'
        while not success_flag:
            try:
                response = get_response(prompt, call_llm, client_id, question_type)
                thought = parse_thought(response, language)
                question = parse_question(response, language)
                success_flag = True
            except:
                failed_time += 1
                if failed_time > call_llm_retry_time:
                    logger.info(f'client: {client_id}, screen question generation failed')
                    break
    # logger.info(f'client: {client_id}, question_type: {question_type}, response: {response}')
    return thought, question, question_type


def diagnosis_procedure(dialogue_string, call_llm, client_id, disease_name, question_num,
                        maximum_question_per_differential_diagnosis_disease, language):
    question_type = "diagnosis_conversation_template"
    prompt = llm_prompt_dict[question_type][language].format(disease_name, disease_name, disease_name, dialogue_string)

    success_flag, failed_time = False, 0
    thought = 'none'
    response_content = 'none'
    while not success_flag:
        try:
            response = get_response(prompt, call_llm, client_id, question_type)
            thought = parse_thought(response, language)
            if language == 'chn' and '<问题>' in response:
                response_content = parse_question(response, language)
            elif language == 'chn' and '<诊断>' in response:
                response_content = parse_diagnosis(response, language)
            elif language == 'eng' and '<Question>' in response:
                response_content = parse_question(response, language)
            else:
                assert language == 'end' and '<Diagnosis>' in response
                response_content = parse_question(response, language)
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info(f'client: {client_id}, sufficiency analysis failed')
                break
    if '你确诊了{}'.format(disease_name) in response_content:
        complete_flag = 1
        confirm_flag = 1
    elif '你没有得{}'.format(disease_name) in response_content:
        complete_flag = 1
        confirm_flag = 0
    else:
        complete_flag = 0
        confirm_flag = 0
    if question_num + 1 >= maximum_question_per_differential_diagnosis_disease:
        complete_flag = 1
    return thought, response_content, complete_flag, confirm_flag, question_num+1


def state_init_or_reload(content, diagnosis_target, phase):
    if len(content) < 1:
        if phase == 'DIAGNOSIS':
            screen_flag = 0
            candidate_disease_list = [
                diagnosis_target
            ]
        else:
            assert phase == 'SCREEN' or phase == 'ALL'
            screen_flag = 1
            candidate_disease_list = []
        state = {
            'screen_flag': screen_flag,
            'end_flag': 0,
            'candidate_disease_list': candidate_disease_list
        }
    else:
        # 如果并非第一轮对话，则基于上一轮对话的state进行初始化
        start_idx = (content[-2]['full_response'].find('<AFFILIATED-INFO>') +
                     len('<AFFILIATED-INFO>'))
        end_idx = content[-2]['full_response'].find('</AFFILIATED-INFO>')
        state = json.loads(content[-2]['full_response'][start_idx: end_idx])
    return state


def dialogue_organize(content, language):
    dialogue_list, dialogue_string = [], ''
    assert len(content) % 2 == 0
    for i in range(len(content)):
        flag_idx = content[i]['full_response'].find('<RESPONSE>')
        end_idx = content[i]['full_response'].find('</RESPONSE>')

        start_idx = flag_idx + len('<RESPONSE>')
        turn_content = content[i]['full_response'][start_idx: end_idx]
        assert end_idx > 0 and flag_idx >= 0

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

    turn_num = len(content) // 2
    return dialogue_string, dialogue_list, turn_num


