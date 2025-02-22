import copy
import logging
from .doctor_config import args
from .ecdai_doctor_util import procedure_dict
from .ecdai_doctor_prompt import ecdai_prompt_dict
from .util import parse_patient_response_intention

logger = logging.getLogger('evaluation_logger')

SUB_PROCEDURE_TO_START = 'SUB_PROCEDURE_TO_START'
SUB_PROCEDURE_ONGOING = 'SUB_PROCEDURE_ONGOING'
SUB_PROCEDURE_CONFIRM = 'SUB_PROCEDURE_CONFIRM'
SUB_PROCEDURE_EXCLUDE = 'SUB_PROCEDURE_EXCLUDE'

PROCEDURE_ILLEGAL = 'PROCEDURE_ILLEGAL'
PROCEDURE_TO_START = 'PROCEDURE_TO_START'
PROCEDURE_ONGOING = 'PROCEDURE_ONGOING'
PROCEDURE_CONFIRM = 'PROCEDURE_CONFIRM'
PROCEDURE_EXCLUDE = 'PROCEDURE_EXCLUDE'
PROCEDURE_CONFIRM_END = 'PROCEDURE_END_CONFIRM'
PROCEDURE_EXCLUDE_END = 'PROCEDURE_END_EXCLUDE'
EPISODE_END = 'EPISODE_END'

REQUEST_PROCEDURE_START = "REQUEST_PROCEDURE_START"
REQUEST_INFORMATION = "REQUEST_INFORMATION"

call_llm_retry_time = args['call_llm_retry_time']
code_missing_strategy = args['ecdai_code_missing_strategy']
diagnosis_internal_forward_flag = args['ecdai_diagnosis_internal_forward_flag']
assert diagnosis_internal_forward_flag == 1 or diagnosis_internal_forward_flag == 0
diagnosis_internal_forward_flag = True if diagnosis_internal_forward_flag == 1 else False


def request_parse(dialogue_list, call_llm, language):
    success_flag = False
    failed_time = 0
    know_flag = False
    while not success_flag:
        try:
            prompt_type = 'diagnosis_patient_agreement_parse_template'
            prompt = ecdai_prompt_dict[prompt_type][language].format(dialogue_list[-2], dialogue_list[-1])
            response = call_llm(prompt)
            know_flag = parse_patient_response_intention(response, target_length=1, language=language)[0]
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info('parse failed')
                break
    return know_flag


def internal_forward_parse(question, context, call_llm, language):
    success_flag = False
    failed_time = 0
    know_flag, confirm_flag = False, False
    while not success_flag:
        try:
            prompt_type = 'diagnosis_internal_forward_template'
            response_prompt = ecdai_prompt_dict[prompt_type][language].format(question, context)
            response = call_llm(response_prompt)
            know_flag, confirm_flag = parse_patient_response_intention(response, target_length=2, language=language)
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info('parse failed')
                break
    return know_flag, confirm_flag


def answer_parse(dialogue_list, call_llm, language):
    success_flag = False
    failed_time = 0
    know_flag, confirm_flag = False, False
    while not success_flag:
        try:
            prompt_type = 'diagnosis_patient_answer_parse_template'
            response_prompt = ecdai_prompt_dict[prompt_type][language].format(dialogue_list[-2], dialogue_list[-1])
            response = call_llm(response_prompt)
            know_flag, confirm_flag = parse_patient_response_intention(response, target_length=2, language=language)
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info('parse failed')
                break
    return know_flag, confirm_flag


def re_initialize_state(diagnosis_state, candidate_disease_list, top_n):
    # 由于我们目前的设计是精确到ICD编码第4位，因此诊断流程可能会出现一对多的情况。因此这里的Procedure分更为具体的SUB_PROCEDURE
    # 一开始都置PROCEDURE_STATE_TO_START，SUB_PROCEDURE_TO_START状态。后续等问到了后进入ONGOING状态
    # 诊断原则是，如果下属所有疾病都可以排除，则置PROCEDURE_EXCLUDE，只要有一个确诊，就置PROCEDURE_CONFIRM。
    diagnosis_state = copy.deepcopy(diagnosis_state)
    for i in range(top_n):
        code = candidate_disease_list[i][2]
        if code in procedure_dict:
            if code not in diagnosis_state:
                procedure_state = {
                    'state': PROCEDURE_TO_START,
                    'disease_info': {}
                }
                for idx, item in enumerate(procedure_dict[code]):
                    # 因为这里后面要序列化，根据json规定，key必须是字符串
                    procedure_state['disease_info'][str(idx)] = {
                        'state': SUB_PROCEDURE_TO_START,
                        'first_question': int(item[3]),
                        'current_question': int(item[3]),
                    }
                diagnosis_state[code] = procedure_state
        else:
            # 技术上初筛阶段如果是skip，鉴别诊断阶段不应该能进入这个分支。
            assert code_missing_strategy == 'end'

            # 对于那些在初筛清单中，但不mapping鉴别诊断的疾病，给定两种策略。
            # 如果是skip策略，就直接跳过，当做完全不存在
            # 如果是end策略，就直接终止对话
            if code_missing_strategy == 'skip':
                diagnosis_state[code] = {
                    'state': PROCEDURE_EXCLUDE_END,
                    'disease_info': {}
                }
            else:
                assert code_missing_strategy == 'end'
                diagnosis_state[code] = {'state': SUB_PROCEDURE_TO_START}
    return diagnosis_state


def last_question_answer_state_update(last_action, diagnosis_question_flag, disease_code, question_type,
                                      sub_procedure_index, sub_procedure_question_index, diagnosis_state,
                                      dialogue_list, call_llm, client_id, language):
    # disease_code代表icd编码前4位。对应的question type有两种。PROCEDURE_START_REQUEST，代表上一轮的对话是提问是否要开展某个疾病的诊断
    # 另一个是PROCEDURE_STATE_ONGOING。代表正在某个疾病的诊断过程中
    # 注意，PROCEDURE_STATE 还会有其它状态，但是这些状态不会在previous action清单里出现。
    end_flag = 0
    assert diagnosis_question_flag == 'D'
    if question_type == REQUEST_PROCEDURE_START:
        # 正常情况下，PROCEDURE_START_REQUEST不对应具体的诊断问题，因此相应的问题一定是None
        assert sub_procedure_index == 'None' and sub_procedure_question_index == 'None'
        # 当是第一个问题时，只要没有明确表示同意，就直接退出。
        approve_flag = request_parse(dialogue_list, call_llm, language)
        if approve_flag:
            # 当明确表示同意时，将相关疾病的状态置为ongoing
            # 并且自动将下属的第一个子疾病状态置为ongoing
            diagnosis_state[disease_code]['state'] = PROCEDURE_ONGOING
            diagnosis_state[disease_code]['disease_info']['0']['state'] = SUB_PROCEDURE_ONGOING
            logger.info(f'client: {client_id}, last request: {last_action}, approved')
        else:
            logger.info(f'client: {client_id}, last request: {last_action}, refused')
            end_flag = 1
    else:
        assert question_type == PROCEDURE_ONGOING
        # 然后更新状态，事实上我们只要关注上一轮的问题即可。在明确回答的情况下，回答有三种情况。
        # 要不患者回答承认了，或者否认了所提问的是非题，要不患者说他不知道。
        # know_flag代表患者的回答表示患者不知道所提问的信息，answer_flag代表（在询问信息知晓的情况下），患者是否承认了
        know_flag, confirm_flag = answer_parse(dialogue_list, call_llm, language)
        logger.info(f'client: {client_id}, last request: {last_action}, know flag: {know_flag}, '
                    f'confirm flag: {confirm_flag}')

        if know_flag:
            diagnosis_state = diagnose_procedure_step_forward(
                disease_code, sub_procedure_index, sub_procedure_question_index, diagnosis_state, confirm_flag,
                client_id)
        else:
            end_flag = 1

    if is_all_exclude(diagnosis_state, disease_code):
        diagnosis_state[disease_code]['state'] = PROCEDURE_EXCLUDE
        logger.info(f'client: {client_id}, code {disease_code}, exclude')
    diagnosis_state = copy.deepcopy(diagnosis_state)
    return diagnosis_state, end_flag


def diagnose_procedure_step_forward(disease_code, sub_procedure_index, sub_procedure_question_index, diagnosis_state,
                                    confirm_flag, client_id):
    condition = procedure_dict[disease_code][int(sub_procedure_index)][2][int(sub_procedure_question_index)]
    if confirm_flag:
        logger.info(f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                    f'question idx: {sub_procedure_question_index}, confirm')
        if condition.yes[1] == 'NAVIGATE':
            next_question = int(condition.yes[0])
            diagnosis_state[disease_code]['disease_info'][str(sub_procedure_index)]['current_question'] = (
                next_question)
            logger.info(
                f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                f'question idx: {sub_procedure_question_index}, next question id: {next_question}')
        elif condition.yes[1] == 'DIAGNOSIS CONFIRM':
            diagnosis_state[disease_code]['state'] = PROCEDURE_CONFIRM
            diagnosis_state[disease_code]['disease_info'][str(sub_procedure_index)]['state'] = (
                SUB_PROCEDURE_CONFIRM)
            logger.info(
                f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                f'question idx: {sub_procedure_question_index}, sub procedure confirm')
        else:
            assert condition.yes[1] == 'DIAGNOSIS EXCLUDE'
            diagnosis_state[disease_code]['disease_info'][str(sub_procedure_index)]['state'] = SUB_PROCEDURE_EXCLUDE
            logger.info(
                f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                f'question idx: {sub_procedure_question_index}, sub procedure exclude')

            # 当排除时，自动置下一个为ongoing
            if str(int(sub_procedure_index) + 1) in diagnosis_state[disease_code]['disease_info']:
                diagnosis_state[disease_code]['disease_info'][str(int(sub_procedure_index) + 1)]['state'] = (
                    SUB_PROCEDURE_ONGOING)
                logger.info(
                    f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                    f'question idx: {sub_procedure_question_index}, next sub procedure: '
                    f'{str(int(sub_procedure_index) + 1)}')
    else:
        logger.info(f'internal forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                    f'question idx: {sub_procedure_question_index}, deny')
        if condition.no[1] == 'NAVIGATE':
            next_question = int(condition.no[0])
            diagnosis_state[disease_code]['disease_info'][str(sub_procedure_index)]['current_question'] = (
                next_question)
            logger.info(
                f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                f'question idx: {sub_procedure_question_index}, next question id: {next_question}')
        elif condition.no[1] == 'DIAGNOSIS CONFIRM':
            diagnosis_state[disease_code]['state'] = PROCEDURE_CONFIRM
            diagnosis_state[disease_code]['disease_info'][str(sub_procedure_index)]['state'] = (
                SUB_PROCEDURE_CONFIRM)
            logger.info(
                f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                f'question idx: {sub_procedure_question_index}, sub procedure confirm')
        else:
            diagnosis_state[disease_code]['disease_info'][str(sub_procedure_index)]['state'] = (
                SUB_PROCEDURE_EXCLUDE)
            logger.info(
                f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                f'question idx: {sub_procedure_question_index}, sub procedure exclude')
            assert condition.no[1] == 'DIAGNOSIS EXCLUDE'
            # 当排除时，自动置下一个为ongoing
            next_sub_procedure_index = str(int(sub_procedure_index) + 1)
            if next_sub_procedure_index in diagnosis_state[disease_code]['disease_info']:
                diagnosis_state[disease_code]['disease_info'][next_sub_procedure_index]['state'] = (
                    SUB_PROCEDURE_ONGOING)
                logger.info(
                    f'client: {client_id}, forward, code: {disease_code}, sub_procedure_idx: {sub_procedure_index}, '
                    f'question idx: {sub_procedure_question_index}, next sub procedure: '
                    f'{next_sub_procedure_index}')
    return copy.deepcopy(diagnosis_state)


def internal_forward(diagnosis_state, disease_code, dialogue_string, call_llm, client_id, language):
    # 内部推进仅对当前disease coding对应的流程开展（技术上来说，如果不断进行初筛和诊断的跳转，造成最高风险疾病变化，
    # 可能会造成多个diagnosis同时ongoing），因此要做限制
    # 本函数用于避免在对话中重复询问已经知道的信息。具体地说，他会遍历所有当前ongoing的疾病（并从目前的进度开始）
    # 一步一步在内部确认这个信息在之前的对话中是否出现过。如果有就直接自更新，如果没有就报stop跳出，在当前中止
    # 注意，这个内部推进算法只会推进当前已经在ongoing的诊断树
    if diagnosis_state[disease_code]['state'] != PROCEDURE_ONGOING:
        return diagnosis_state, True
    if disease_code not in procedure_dict:
        logger.info(f'client: {client_id}, code: {disease_code} not in procedure_dict')
        return diagnosis_state, True

    diagnosis_procedure = procedure_dict[disease_code]

    # get next question
    sub_procedure_idx, question_idx = None, None
    for key in diagnosis_state[disease_code]['disease_info']:
        if diagnosis_state[disease_code]['disease_info'][key]['state'] != SUB_PROCEDURE_ONGOING:
            continue
        sub_procedure_idx = int(key)
        question_idx = diagnosis_state[disease_code]['disease_info'][key]['current_question']
        break
    # 按照道理，如果procedure在ongoing，则一定会有相应的问题存在。如果不存在说明问题已经问完了，则说明状态不应该是ongoing
    assert sub_procedure_idx is not None and question_idx is not None
    condition = diagnosis_procedure[sub_procedure_idx][2][question_idx]
    question_str = condition.condition
    context = dialogue_string
    know_flag, confirm_flag = internal_forward_parse(question_str, context, call_llm, language)

    if not know_flag:
        logger.info(f'client: {client_id}, code: {disease_code}, sub_procedure_idx: {sub_procedure_idx}, '
                    f'question_idx: {question_idx}, internal forward failed')
        return diagnosis_state, True
    else:
        logger.info(f'client: {client_id}, code: {disease_code}, sub_procedure_idx: {sub_procedure_idx}, '
                    f'question_idx: {question_idx}, internal forward success')
        diagnosis_state = diagnose_procedure_step_forward(
            disease_code, sub_procedure_idx, question_idx, diagnosis_state, confirm_flag, client_id)
        diagnosis_state = copy.deepcopy(diagnosis_state)

        return diagnosis_state, False


def update_diagnosis_state(diagnosis_state, candidate_disease_list, previous_questions, dialogue_string,
                           dialogue_list, direct_answer, call_llm, top_n, client_id, language):
    # 对没有初始化过的diagnosis做初始化，如果初始化过了就不做处理。均只处理风险最高的n个疾病
    diagnosis_state = re_initialize_state(diagnosis_state, candidate_disease_list, top_n)

    # 此处有三种情况。phase为diagnosis，此时diagnosis state刚被初始化，没有获得任何信息
    # 注意，如果是正常的初筛进入鉴别诊断，应该上一轮直接是一个和鉴别诊断相关的问题
    assert len(previous_questions) > 0 and previous_questions[0] == 'FIRST_QUESTION-0'
    if len(previous_questions) == 1:
        logger.info(f'client: {client_id}, enter differential diagnosis when no screening question is answered')
        return diagnosis_state, 0

    # 2. 上一轮是鉴别诊断的提问，但没有直接回答
    # 注意，说不知道也是一种直接回答。因此
    # 如果direct answer是True，那么最后一个answered question就是上一轮问过的问题，直接回答了，但是还没有解析过的状态
    # 如果direct answer是False，那么最后一个answered question应该是上一次已经成功解析过的诊断提问
    # 因此，如果direct answer是True，且最后一个状态是D打头的，那么就一定是需要更新的状态，否则Diagnosis state无需更新
    # 在上一轮有明确的回答，且end_flag为0导致可以走到这一步的情况下，是不可能导致对话结束的，因此此处的end_flag默认为0
    # 这里直接返回diagnosis_state，如果没有重新解析，则正常情况会重新触发提问
    last_action = previous_questions[-1]
    if not ('D' == last_action[0] and direct_answer):
        return diagnosis_state, 0

    diagnosis_question_flag, disease_code, question_type, sub_procedure_index, sub_procedure_question_index = (
        last_action.split('-'))

    # 3. 上一轮是直接回答了鉴别诊断信息。基于上一轮的问题做出解析，进行update
    # 3.1 如果用户说不知道或拒绝回答，直接终止对话。注意，按照之前的设计，direct answer会被判断为直接回答了对话
    # 3.2 如果用户正常回复了，正常update和内部internal forward
    # 技术上可能出现last action对应的state不在当前top n的情况，但是按照道理来讲这不会造成问题，因为action对应的state应该被初始化过
    diagnosis_state, end_flag = (
        last_question_answer_state_update(
            last_action, diagnosis_question_flag, disease_code, question_type, sub_procedure_index,
            sub_procedure_question_index, diagnosis_state, dialogue_list, call_llm, client_id, language
        )
    )

    assert end_flag == 1 or end_flag == 0

    stop_flag = False
    logger.info(f'client: {client_id}, code: {disease_code}, start internal forward')
    while (not stop_flag) and end_flag == 0 and diagnosis_internal_forward_flag:
        # 内部推进不会涉及end_flag。并且只在上一轮问的问题中推进。
        diagnosis_state, stop_flag = (
            internal_forward(diagnosis_state, disease_code, dialogue_string, call_llm, client_id, language)
        )
    return diagnosis_state, end_flag


def is_all_exclude(diagnosis_state, disease_code):
    all_exclude_flag = True
    for idx in diagnosis_state[disease_code]['disease_info']:
        if diagnosis_state[disease_code]['disease_info'][idx]['state'] != SUB_PROCEDURE_EXCLUDE:
            all_exclude_flag = False
    return all_exclude_flag


def information_parse(diagnosis_procedure_info, call_llm, language):
    diagnosis_text_list = []
    for disease_info in diagnosis_procedure_info:
        text = disease_info[1]
        diagnosis_text_list.append(text)

    diagnosis_text = '\n\n'.join(diagnosis_text_list)
    prompt_type = 'diagnosis_lab_test_exam_summary_template'
    response_prompt = ecdai_prompt_dict[prompt_type][language].format(diagnosis_text)

    success_flag, failed_time, response = False, 0, ''
    while not success_flag:
        try:
            response = call_llm(response_prompt)
            success_flag = True
        except:
            failed_time += 1
            if failed_time > call_llm_retry_time:
                logger.info('parse failed')
                break
    logger.info('response: {}'.format(response))
    return response


def generate_procedure_action(diagnosis_procedure_info, diagnosis_state):
    assert diagnosis_state['state'] == PROCEDURE_ONGOING
    question, idx, question_idx = '', 0, 0
    for idx in [str(i) for i in range(1000)]:
        if idx not in diagnosis_state['disease_info']:
            continue
        state = diagnosis_state['disease_info'][idx]['state']
        # 按照当前的设计（有一个SUB CONFIRM就判定PROCEDURE CONFIRM，并在最开始的状态解析确认的设计）
        # 是不可能在这里出现CONFIRM的。遇到SUB_PROCEDURE_EXCLUDE是可能的，但是可以直接跳过
        assert state != SUB_PROCEDURE_CONFIRM
        if state == SUB_PROCEDURE_EXCLUDE:
            continue
        else:
            assert state == SUB_PROCEDURE_ONGOING
            question_idx = diagnosis_state['disease_info'][idx]['current_question']
            question = diagnosis_procedure_info[int(idx)][2][question_idx].condition
            break
    assert question != ''
    return question, int(idx), int(question_idx)


def check_legal_status(state):
    # 任意时刻，至多只有一个状态在Exclude或者Confirm
    count = 0
    for code in state:
        procedure_state = state[code]['state']
        if procedure_state == PROCEDURE_CONFIRM:
            count += 1
        if procedure_state == PROCEDURE_EXCLUDE:
            count += 1
    return count < 2


def diagnosis_diagnosis_prompt_generate(state, high_risk_diseases, top_n, language):
    # 根据已经更新过的state逐个进行判断
    # 注意，按照目前的设计，至多只存在一种疾病完成了CONFIRM或者EXCLUDE。因此只要做一次判断即可，如果没match就输出None
    assert check_legal_status(state)

    disease_list = high_risk_diseases[: top_n]
    response_prompt, response_key, confirm_flag = None, None, False
    for disease_info in disease_list:
        code = disease_info[2]
        disease_diagnosis_state = state[code]
        # 如果code在诊断表里，则按照诊断表决策
        if code in procedure_dict:
            disease_name = disease_info[1]
            procedure_state = disease_diagnosis_state['state']
            if procedure_state == PROCEDURE_CONFIRM:
                key = 'diagnosis_confirm_template'
                response_prompt = ecdai_prompt_dict[key][language].format(disease_name)
                response_key = '-'.join(['D', code, PROCEDURE_CONFIRM, 'None', 'None'])
                disease_diagnosis_state['state'] = PROCEDURE_CONFIRM_END
                confirm_flag = True
                break
            elif procedure_state == PROCEDURE_EXCLUDE:
                key = 'diagnosis_exclude_template'
                response_prompt = ecdai_prompt_dict[key][language].format(disease_name)
                response_key = '-'.join(['D', code, PROCEDURE_EXCLUDE, 'None', 'None'])
                disease_diagnosis_state['state'] = PROCEDURE_EXCLUDE_END
                break

    # 确认同一时刻在更新后是没有PROCEDURE_CONFIRM和EXCLUDE这两个暂态状态
    for disease_info in disease_list:
        code = disease_info[2]
        procedure_state = state[code]['state']
        assert procedure_state != PROCEDURE_CONFIRM and procedure_state != PROCEDURE_EXCLUDE
    return response_prompt, response_key, confirm_flag


def diagnosis_inquiry_prompt_generate(previous_state, high_risk_diseases, call_llm, top_n, language, diagnosis_mode):
    # 根据已经更新过的state逐个进行判断
    disease_list = high_risk_diseases[: top_n]
    response_prompt, response_key, end_flag = None, None, 0
    for disease_info in disease_list:
        code = disease_info[2]
        disease_diagnosis_state = previous_state[code]
        # 如果code在诊断表里，则按照诊断表决策
        if code in procedure_dict:
            diagnosis_procedure_info = procedure_dict[code]
            disease_name = disease_info[1]
            procedure_state = disease_diagnosis_state['state']

            if procedure_state == PROCEDURE_TO_START:
                prompt_type = 'diagnosis_procedure_start_inquiry_template'
                essential_info_str = information_parse(diagnosis_procedure_info, call_llm, language)
                response_prompt = ecdai_prompt_dict[prompt_type][language].format(disease_name, essential_info_str)
                response_key = '-'.join(['D', code, REQUEST_PROCEDURE_START, 'None', "None"])
                logger.info(f'Procedure Start Request: {response_key}')
            elif procedure_state == PROCEDURE_ONGOING:
                question, idx, question_idx = generate_procedure_action(
                    diagnosis_procedure_info, disease_diagnosis_state)
                prompt_type = 'diagnosis_procedure_proceeding_template'
                response_prompt = ecdai_prompt_dict[prompt_type][language].format(question)
                response_key = '-'.join(['D', code, PROCEDURE_ONGOING, str(idx), str(question_idx)])
                logger.info(f'Procedure Ongoing: {response_key}')
            else:
                # 如果是这三个状态，本来就是不做处理
                assert procedure_state in [PROCEDURE_CONFIRM, PROCEDURE_CONFIRM_END, PROCEDURE_EXCLUDE,
                                           PROCEDURE_EXCLUDE_END]
            if response_prompt is not None and response_key is not None:
                break
        else:
            #
            assert code_missing_strategy == 'end'
            prompt_type = 'diagnosis_failed_diagnosis_procedure_missing_template'
            response_prompt = ecdai_prompt_dict[prompt_type][language].format(disease_info[1])
            response_key = '-'.join(['D', code, PROCEDURE_ILLEGAL, 'None', "None"])
            logger.info(f'Procedure illegal: {response_key}')
            end_flag = 1
            break

    # 如果所有的问题都问完了，response key应该是None，否则应该会有一个问题
    if response_prompt is None:
        if diagnosis_mode == 'TOP_HIT':
            prompt_type = 'diagnosis_failed_not_hit'
        else:
            assert diagnosis_mode == 'ALL'
            prompt_type = 'diagnosis_accomplish_end'
        response_prompt = ecdai_prompt_dict[prompt_type][language]
        response_key = '-'.join(['D', 'None', EPISODE_END, 'None', "None"])
        end_flag = 1
    return response_prompt, response_key, end_flag
