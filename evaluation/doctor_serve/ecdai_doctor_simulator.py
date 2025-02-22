import asyncio
import json
import logging
from .ecdai_doctor_prompt import ecdai_prompt_dict
from .ecdai_screening import screen_prompt_generation
from .ecdai_diagnosis import (
    update_diagnosis_state, diagnosis_diagnosis_prompt_generate, diagnosis_inquiry_prompt_generate
)
from .util import (non_streaming_call_llm, streaming_call_llm, generate_non_streaming_response, set_fake_response,
                   parse_return_data)
from .ecdai_doctor_util import content_reorganize, ecdai_parse_previous_interact

logger = logging.getLogger('evaluation_logger')


# 注意，这里的部分代码片段和patient serve是重合的，这么设计是为了确保patient 和 doctor simulator都可以直接单独作为后端应用的支撑模块使用
def doctor_agent_behavior(messages, client_id, model_name, phase, screening_maximum_question,
                          top_diagnosis_disease_num, diagnosis_target, diagnosis_mode, language,
                          maximum_question_per_differential_diagnosis_disease):
    """
    本函数用于实现和生成doctor agent的行为prompt。用于后续的输出
    本函数中调用的llm不涉及对外输出，所以不使用流式
    """
    assert phase == 'ALL' or phase == 'SCREEN' or phase == 'DIAGNOSIS'
    if diagnosis_target is not None:
        assert phase == 'DIAGNOSIS'
    if phase == 'DIAGNOSIS':
        assert diagnosis_target is not None

    dialogue_string, dialogue_list, state, turn_num, answered_questions = content_reorganize(messages, phase, language)

    call_llm = lambda input_prompt: non_streaming_call_llm(model_name, input_prompt)
    logger.info('current turn num: {}'.format(turn_num + 1))
    if len(messages) == 0:
        prompt_type = 'first_question'
        prompt_list = [[prompt_type, ecdai_prompt_dict[prompt_type][language], '']]
        state['question_key'] = 'FIRST_QUESTION-0'
        # 如果是诊断阶段，则设置疾病
        # 注意，diagnosis target应该是一个符合本研究格式要求的icd code，能够在诊断决策流程字典中匹配到
        if phase == 'DIAGNOSIS':
            state['candidate_disease_list'] = [[None, None, diagnosis_target, None]]
        state_str = json.dumps(state)
        return prompt_list, state_str

    # 如果是诊断阶段，则可选诊断疾病列表长度必须为1
    if phase == 'DIAGNOSIS':
        assert len(state['candidate_disease_list']) == 1

    # 走到这里代表一定已经开始了正常的对话轮次
    assert len(messages) > 0 and len(messages) % 2 == 0
    direct_answer, medical_question_flag, non_medical_flag, demographic_flag, history_flag, present_flag = (
        ecdai_parse_previous_interact(dialogue_string, call_llm, client_id, language))

    # 重载diagnosis list之类的信息，默认从上个state中继承。如果本轮还满足初筛信息的解析轮次，则重新解析更新
    # 如果是第一轮对话，则所有feature都会置None，然后再后期更新，同时在第一轮结束时所有的feature都会被更新为非None
    previous_turn_end_flag = state['end_flag']
    screen_flag = state['screen_flag']
    candidate_disease_list = state['candidate_disease_list']
    question_key = state['question_key']
    high_risk_diagnoses_str = state['high_risk_diagnoses_str']
    screen_question_str = state['screen_question_str']
    screen_question_candidate_str = state['screen_question_candidate_str']
    diagnosis_state = state['diagnosis_state']

    # 我们在一次对话中可能需要表达多个意图，这里的多个意图会使用不同的prompt多次call llm，逐个实现，每个prompt只表达一个意图。
    # 这里的prompt list用于存储不同意图的指令
    prompt_list = []
    end_flag = 0
    if previous_turn_end_flag == 1:
        prompt_type = 'ending'
        content = ecdai_prompt_dict[prompt_type][language]
        # 如果上一轮已经判定终止，则本轮即什么都不做。可以让患者回退来重新进行对话
        prompt_list.append([prompt_type, content, ''])
        end_flag = 1
    else:
        # 解析上一轮的问题是否被回答。注意，即使是第一轮对话，也是有问题的。
        if question_key is not None:
            if direct_answer:
                answered_questions = answered_questions + [question_key]
        else:
            assert turn_num == 1

        answered_screen_question_num = calculate_answered_screening_questions(answered_questions)

        # 处理上一轮中是否有询问额外的医学问题和非医学问题，这些问题的回答是瞬时性的，不影响本轮的状态流
        last_response = messages[-1]['full_response']
        if medical_question_flag:
            prompt_type = 'response_medical_question_template'
            prompt = ecdai_prompt_dict[prompt_type][language].format(last_response, dialogue_string)
            prompt_list.append(["response_medical_question", prompt, ''])
        if non_medical_flag:
            prompt_type = 'response_non_medical_question_template'
            prompt = ecdai_prompt_dict[prompt_type][language].format(last_response)
            prompt_list.append(['response_non_medical_question', prompt, ''])

        # diagnosis_flag指的是本轮是否要进入鉴别诊断模块，"S"是疾病初筛标志符
        # 如果上一轮中问的问题是鉴别诊断问题。则有两种处理方式
        # 1. 如果phase是diagnosis，则直接取True
        # 1. 如果本轮没有触发screen模块，则默认继续进入diagnosis模块
        # 2. 如果本轮触发了screen模块，则根据screen的判定结果判断，如果screen判定信息收集充分，则继续进入鉴别诊断模块
        #    反之回退会初筛模块进行重新初筛询问
        if phase == 'DIAGNOSIS':
            differential_diagnosis_procedure_flag = True
        elif len(answered_questions) == 0:
            differential_diagnosis_procedure_flag = False
        else:
            differential_diagnosis_procedure_flag = 'D-' in answered_questions[-1]

        # 是否要进行新一轮的疾病初筛
        # 疾病初筛模块会执行症状结构化等一系列操作。最后输出一个动作和当前高风险疾病清单
        # 动作包含两类，第一类是具体的问题，第二类是screen终止的判断
        # 模块的启动条件是：AI发起（得到回答的）初筛提问数量小于阈值。在这个条件下，即便之前已经进入鉴别诊断阶段了，依旧重做初筛。
        if answered_screen_question_num <= screening_maximum_question and phase != 'DIAGNOSIS':
            # 如果进入了初筛，则由当前历史对话解析症状，决定到底要问什么（或者完成初筛）
            # 注意，鉴于初筛的特点，direct answer对这个模型的行为的影响在previous actions（协助其屏蔽部分问题）
            (proactive_inquiry_prompt, question_key, screen_question_str, screen_question_candidate_str,
             high_risk_diagnoses_str, candidate_disease_list, screening_decision_flag) = (
                screen_prompt_generation(
                    dialogue_string, demographic_flag, history_flag, present_flag, answered_questions, call_llm,
                    client_id, screening_maximum_question, top_diagnosis_disease_num, language)
            )
            if screening_decision_flag:
                # screening_decision_flag指代模型要求继续问问题
                # 此时proactive_inquiry_prompt事实上是一个诊断
                # 但是因为初筛诊断后，整体系统的状态可能在后续回退到初筛询问，这种反复跳转甚至可能多次发生，高风险疾病清单
                # 也会随之变化，造成诊断疾病的变化。
                # 如果每次做出诊断都随之告知，要是输出看上去会很奇怪。如果只告知一次，又会显得后面可能出现不同步的问题
                # 因此只在phase设定为初筛（screen）时，疾病初筛结果做显式输出。随之把end flag置1
                # 如果是DIAGNOSIS或者ALL，这一步不做显式输出
                if phase == 'SCREEN':
                    prompt_list.append(['初筛诊断prompt', proactive_inquiry_prompt])
                    end_flag = 1
                else:
                    differential_diagnosis_procedure_flag = True
            else:
                # 此时的screen prompt对应的是一个主动地问题
                prompt_list.append(['screening_prompt', proactive_inquiry_prompt, question_key])
                differential_diagnosis_procedure_flag = False
                # 强制退出
                if turn_num > 5:
                    end_flag = 1
                    logger.info('too long exit')
        else:
            # 这里的分支是为了处理一种特殊的情况
            # 当耗尽所有screen_maximum_question的限额，但是screen model依旧在问问题。
            # 此时，如果phase处于screen，代表本session应当结束
            # 如果phase是all，代表应当无条件进入diagnosis
            if answered_screen_question_num >= screening_maximum_question:
                if phase == 'SCREEN':
                    prompt_list.append(['初筛诊断prompt', 'end'])
                    end_flag = 1
                    differential_diagnosis_procedure_flag = False
                elif phase == 'ALL':
                    differential_diagnosis_procedure_flag = True
                else:
                    assert phase == 'DIAGNOSIS'
                    assert differential_diagnosis_procedure_flag

        if end_flag == 0 and differential_diagnosis_procedure_flag and phase != 'SCREEN':
            # 此处进入鉴别诊断阶段的条件有三个
            # 1. end_flag != 0，按照当前设计，只有在phase设为screen时，且做出了screen decision决策或超长截断时会不满足
            # 2. differential_diagnosis_procedure_flag。有三种情况
            #    第一，phase是diagnosis，应当直接取True，且因为phase是diagnosis，不可能触发screen中的重置机制
            #    第二，上一轮的问题就是诊断，且没有因为screen（或者screen已经跑完了）重置flag
            #    第三，screen过程正常跑出了决策，然后随之flag置True

            # 如果是ALL，则根据预先设定的范围重新设置诊断空间；如果是针对某种特定疾病的Diagnosis，则是必须只有一个
            if phase == 'ALL':
                top_n = top_diagnosis_disease_num
            else:
                assert phase == 'DIAGNOSIS'
                top_n = 1
            diagnosis_state, end_flag = update_diagnosis_state(
                diagnosis_state, candidate_disease_list, answered_questions, dialogue_string, dialogue_list,
                direct_answer, call_llm, top_n, client_id, language
            )

            if end_flag == 1:
                # 此处end_flag触发终止只有一种情况，就是在update state时，因为说了不同意或者不知道触发了end flag
                prompt_type = 'response_insufficient_information'
                prompt = ecdai_prompt_dict[prompt_type][language]
                prompt_list.append([prompt_type, prompt, ''])
            else:
                # 当这里意味着疾病确诊时，diagnosis_diagnosis_prompt_generate函数会自动把confirm置True
                response_prompt, response_key, disease_confirm_flag = (
                    diagnosis_diagnosis_prompt_generate(
                        diagnosis_state, candidate_disease_list, top_diagnosis_disease_num, language)
                )
                # 这是更新状态后如果有诊断的输出，如果无诊断会直接无输出。
                if response_prompt is not None:
                    prompt_type = 'differential_diagnosis_decision'
                    prompt_list.append([prompt_type, response_prompt, response_key])

                # 如果有疾病被确认且diagnosis mode为top hit，则直接终止对话
                if disease_confirm_flag and diagnosis_mode == 'TOP_HIT':
                    end_flag = 1
                    prompt_type = 'diagnosis_confirm_end'
                    response_prompt = ecdai_prompt_dict[prompt_type][language]
                    prompt_list.append(['confirm_end', response_prompt, ''])
                else:
                    # 此处的end flag如果触发，就是的确走完了完整的诊断流程
                    proactive_inquiry_prompt, question_key, end_flag = (
                        diagnosis_inquiry_prompt_generate(
                            diagnosis_state, candidate_disease_list, call_llm, top_diagnosis_disease_num, language,
                            diagnosis_mode)
                    )
                    prompt_type = 'differential_diagnosis_inquiry'
                    screen_flag = 0
                    prompt_list.append([prompt_type, proactive_inquiry_prompt, question_key])
        else:
            if end_flag == 1:
                assert phase == 'SCREEN'
            assert len(prompt_list) > 0

    # 状态存储
    assert len(prompt_list) > 0
    assert question_key is not None
    new_state = generate_state(answered_questions, question_key, candidate_disease_list, high_risk_diagnoses_str,
                               screen_question_str, screen_question_candidate_str, diagnosis_state, screen_flag,
                               end_flag)
    new_state_str = json.dumps(new_state)
    return prompt_list, new_state_str


async def doctor_response_generation(prompt_list, client_id, model_name, streaming, new_state_str):
    if streaming:
        yield json.dumps(set_fake_response('<AFFILIATED-INFO>'
                                           + new_state_str
                                           + '</AFFILIATED-INFO>'
                                           + '<RESPONSE>')) + '\n\n$$$\n\n'
        for prompt_info in prompt_list:
            content = prompt_info[2].replace('\n', ' ')
            logger.info(f'client: {client_id}, question type: {prompt_info[0]}, content: {prompt_info[1]}, '
                        f'key: {content}')
            async for result in streaming_call_llm(model_name, prompt_info[1], streaming):
                yield result
        yield json.dumps(set_fake_response('</RESPONSE>')) + '\n\n$$$\n\n'
    else:
        yield generate_non_streaming_response(
            prompt_list, client_id, model_name, streaming, new_state_str
        )


def generate_state(answered_questions, question_key, candidate_disease_list, high_risk_diagnoses_str,
                   screen_question_str, screen_question_candidate_str, diagnosis_state, screen_flag,
                   end_flag):
    assert end_flag == 1 or end_flag == 0
    assert screen_flag == 1 or screen_flag == 0

    state = dict()
    # 如果本轮没有更新，则直接使用上一轮的
    assert answered_questions is not None
    assert question_key is not None
    assert diagnosis_state is not None
    assert end_flag is not None
    assert candidate_disease_list is not None
    assert high_risk_diagnoses_str is not None
    assert screen_question_str is not None
    assert screen_question_candidate_str is not None
    assert diagnosis_state is not None

    state['answered_questions'] = answered_questions
    state['question_key'] = question_key
    state['screen_flag'] = screen_flag
    state['diagnosis_state'] = diagnosis_state
    state['end_flag'] = end_flag
    state['candidate_disease_list'] = candidate_disease_list
    state['high_risk_diagnoses_str'] = high_risk_diagnoses_str
    state['screen_question_str'] = screen_question_str
    state['screen_question_candidate_str'] = screen_question_candidate_str
    state['diagnosis_state'] = diagnosis_state
    return state


def calculate_answered_screening_questions(previous_questions):
    # 注意，这里的screen questions只计算初筛症状问题
    count = 0
    for item in previous_questions:
        question_type, question_value = item.split('-')[:2]
        if question_type == 'S' and question_value.isdecimal():
            count += 1
    return count


async def collect_responses(generator):
    result_list = []
    async for res in generator:
        result_list.append(res)
    return result_list


def doctor_behavior_wrapper(messages, client_id, model_name, phase, screening_maximum_question,
                            top_diagnosis_disease_num, diagnosis_target, diagnosis_mode, environment_language,
                            maximum_question_per_differential_diagnosis_disease):
    streaming = False
    prompt_list, new_state_str = doctor_agent_behavior(
        messages, client_id, model_name, phase, screening_maximum_question, top_diagnosis_disease_num, diagnosis_target,
        diagnosis_mode, environment_language, maximum_question_per_differential_diagnosis_disease)
    response_generator = doctor_response_generation(prompt_list, client_id, model_name, streaming, new_state_str)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        llm_response = loop.run_until_complete(collect_responses(response_generator))
    finally:
        loop.close()

    response = parse_return_data(llm_response, streaming=streaming)
    return response
