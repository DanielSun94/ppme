import json
from .react_doctor_util import (dialogue_organize, screen_information_sufficiency_analyze, screen_question_generation,
                                generate_high_risk_diseases, state_init_or_reload, diagnosis_procedure)
from .util import (non_streaming_call_llm)


def input_check(phase, screening_maximum_question, top_diagnosis_disease_num, diagnosis_target, diagnosis_mode):
    assert phase == 'ALL' or phase == 'SCREEN' or phase == 'DIAGNOSIS'
    if phase == 'SCREEN':
        assert screening_maximum_question > 0
    elif phase == 'DIAGNOSIS':
        # assert top_diagnosis_disease_num == 0
        assert diagnosis_target is not None
        assert diagnosis_mode is not None
    else:
        assert phase == 'ALL'
        assert screening_maximum_question > 0
        assert top_diagnosis_disease_num > 0
        assert diagnosis_target is None
        assert diagnosis_mode is not None


def doctor_agent_behavior(messages,
                          client_id,
                          model_name,
                          phase,
                          screening_maximum_question,
                          top_diagnosis_disease_num,
                          diagnosis_target,
                          diagnosis_mode,
                          environment_language,
                          maximum_question_per_differential_diagnosis_disease
                          ):
    """
    注意，react doctor simulator和llm doctor的整体决策链是完全一致的。区别仅仅在于涉及到的所有prompt加一个反思模块
    这里在react doctor util处有些函数是复用的，我没有把他们合并
    """
    input_check(phase, screening_maximum_question, top_diagnosis_disease_num, diagnosis_target, diagnosis_mode)

    state = state_init_or_reload(messages, diagnosis_target, phase)
    dialogue_string, dialogue_list, turn_num = dialogue_organize(messages, environment_language)
    call_llm = lambda input_prompt: non_streaming_call_llm(model_name, input_prompt, max_token_num=512)

    # 走到这里代表一定已经开始了正常的对话轮次
    # LLM based simulator默认会回答所有问题。因此，此处不需要做意图判断
    # 只需要判断信息是否完整即可
    # 重载diagnosis list之类的信息，默认从上个state中继承。如果本轮还满足初筛信息的解析轮次，则重新解析更新
    # 如果是第一轮对话，则所有feature都会置None，然后再后期更新，同时在第一轮结束时所有的feature都会被更新为非None
    previous_end_flag = state['end_flag']
    screen_flag = state['screen_flag']
    candidate_disease_list = state['candidate_disease_list']
    response, question_type, end_flag = None, None, 0

    turn_thought_dict = {}
    if previous_end_flag == 1 or turn_num > 15:
        # 如果上一轮已经判定终止，则本轮即什么都不做。可以让患者回退来重新进行对话
        # question_type = 'ending'
        response = 'ending'
        end_flag = 1
    else:
        # 强制的二段式分析
        if screen_flag == 1:
            # sufficient_flag指代已经获取的信息是否已经充分，以进入下一阶段。
            # 有两种触发条件，一个是大模型自行判断充分，另一个是长度到了预设值
            if len(dialogue_string) > 0:
                sufficient_thought, sufficient_flag = (
                    screen_information_sufficiency_analyze(dialogue_string, call_llm, client_id, environment_language))
                turn_thought_dict['sufficient_thought'] = sufficient_thought
            else:
                sufficient_flag = 0

            if turn_num >= screening_maximum_question:
                sufficient_flag = 1

            # 如果进入了初筛，则由当前历史对话解析症状，决定到底要问什么（或者完成初筛）
            # 注意，鉴于初筛的特点，direct answer对这个模型的行为的影响在previous actions（协助其屏蔽部分问题）
            if sufficient_flag == 0:
                screen_question_thought, response, question_type = (
                    screen_question_generation(dialogue_string, call_llm, client_id, environment_language))
                turn_thought_dict['screen_question_thought'] = screen_question_thought
            else:
                assert sufficient_flag == 1
                _, screen_diagnosis_thought, candidate_disease_list, question_type = (
                    generate_high_risk_diseases(dialogue_string, call_llm, client_id, environment_language))
                turn_thought_dict['screen_diagnosis_thought'] = screen_diagnosis_thought
                screen_flag = 0
                if phase == 'SCREEN':
                    end_flag = 1

        if phase != 'SCREEN' and screen_flag == 0:
            assert len(candidate_disease_list) > 0
            for i in range(top_diagnosis_disease_num):
                idx, disease_code, disease_name, complete_flag, confirm_flag, question_num = candidate_disease_list[i]
                if complete_flag == 0:
                    differential_diagnosis_thought, response, complete_flag, confirm_flag, question_num = (
                        diagnosis_procedure(dialogue_string, call_llm, client_id, disease_name, question_num,
                                            maximum_question_per_differential_diagnosis_disease,
                                            environment_language))
                    candidate_disease_list[i] = (
                        idx, disease_code, disease_name, complete_flag, confirm_flag, question_num)
                    turn_thought_dict['differential_diagnosis_thought'] = differential_diagnosis_thought
                    break

            complete_procedure_num, confirm_procedure_num = 0, 0
            for i in range(top_diagnosis_disease_num):
                idx, disease_code, disease_name, complete_flag, confirm_flag, question_num = candidate_disease_list[i]
                if complete_flag == 1:
                    complete_procedure_num += 1
                    if confirm_flag == 1:
                        confirm_procedure_num += 1
            if diagnosis_mode == "TOP_HIT" and confirm_procedure_num > 0:
                end_flag = 1
            elif diagnosis_mode == 'ALL' and complete_procedure_num == top_diagnosis_disease_num:
                end_flag = 1
            elif diagnosis_mode == "TOP_HIT" and complete_procedure_num == top_diagnosis_disease_num:
                end_flag = 1

    new_state = {'end_flag': end_flag, 'screen_flag': screen_flag, 'candidate_disease_list': candidate_disease_list,
                 'thought_dict': turn_thought_dict}
    new_state_str = json.dumps(new_state)
    response = f'<AFFILIATED-INFO>{new_state_str}</AFFILIATED-INFO><RESPONSE>{response}</RESPONSE>'
    return response


def doctor_behavior_wrapper(messages, client_id, model_name, phase, screening_maximum_question,
                            top_diagnosis_disease_num, diagnosis_target, diagnosis_mode, environment_language,
                            maximum_question_per_differential_diagnosis_disease):
    # 对llm based agent，这里其实什么都不做，写这个只是为了对仗。llm agent设计之初也只是作为benchmark使用，不会投入应用。
    # 之所以llm based agent不能写成prompt + llm_call的形式，是因为pure llm agent的输出不可控。他做初筛输出和诊断输出时
    # 输出有可能是不合法的，因此我们需要先输出看一看是否合法。而ecdai的意图已经由外部控制器决定了，它基本没这个问题
    return doctor_agent_behavior(messages, client_id, model_name, phase, screening_maximum_question,
                                 top_diagnosis_disease_num, diagnosis_target, diagnosis_mode, environment_language,
                                 maximum_question_per_differential_diagnosis_disease)
