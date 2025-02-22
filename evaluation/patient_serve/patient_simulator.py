# patient simulator其实非常简单，也不分什么阶段，就是一个角色扮演机制，问什么就答什么
import logging
import os
import json
import asyncio
import openai

logger = logging.getLogger('evaluation_logger')

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_info_json = os.path.join(root, 'resource', 'model_info.json')
endpoint_info = json.load(open(model_info_json, 'r', encoding='utf-8-sig'))

prompt_screen_template = {
    'chn': '请假装你是一名病人，正在进行医疗咨询。你的医生正在询问问题，他的问题内容是：{}。'
           '\n\n你们的既往对话历史是：\n{}\n\n请根据医生的问题和你掌握的临床信息（下附）回答接下来的问题。'
           '注意，请简要并且清晰地回答问题，如果这个问题可以用"是"或"否"回答，则优先用是或否回答。'
           '如果医生问你是否要开展某个疾病的鉴别诊断程序，回答是。不要回答医生没有问的问题，不要向医生提任何诉求。'
           '如果你无法从临床信息中找到回答问题的信息，请给予否定性回答。例如，如果问你有没有什么症状，回答没有，问你是不是，回答不是。'
           '注意，如果医生问了超过一个问题，请分别回答。'
           '如果医生询问你的现病史信息，你不可以回复已有的检查检验的结果，也不可以直接告诉医生你现在可能得了什么疾病。'
           '只可以回复主诉、症状、体征、现病史和既往史等信息。请注意可能出现的临床缩写。\n'
           '注意：在描述自己的病情时，尽量不要使用专业医疗术语，请表现得像一个一般的，没有接受过专业医疗教育的民众\n'
           '\n临床信息：\n{}\n\n',
    'eng': 'Pretend you are a patient seeking medical consultation. Your doctor is asking a question: {}. \n\n'
           'Your previous conversation history is:\n{}\n\nPlease respond to the next question based on the doctor\'s '
           'inquiry and the clinical information provided below. Note: Keep your answers brief and clear. '
           'If the question can be answered with "yes" or "no," prioritize using "yes" or "no". '
           'If the doctor asks whether to proceed with a differential diagnosis for a certain disease, answer "yes". '
           'Do not answer questions the doctor has not asked or make any requests of the doctor. '
           'If you cannot find information in the clinical details to answer the question, '
           'provide a negative response. For example, if asked whether you have certain symptoms, answer "no", '
           'If asked "are you," answer "not." Note that if the doctor asks more than one question, '
           'respond to each one separately. '
           'If the doctor inquires about your present illness, do not mention test results or disease diagnoses for '
           'the current condition. Instead, respond only with chief complaints, symptoms, '
           'signs, current illness, and past medical history. Be mindful of possible clinical abbreviations. \n'
           'NOTE: When describing your medical condition, try to avoid using professional medical terminology and '
           'present yourself as a general individual without formal medical education.'
           '\nClinical information:\n{}\n\n'
}

prompt_diagnosis_template = {
    'chn': '请假装你是一名病人，正在进行医疗咨询。你的医生正在询问问题，他的问题内容是：{}。'
           '\n\n你们的既往对话历史是：\n{}\n\n请根据医生的问题和你掌握的临床信息（下附）回答接下来的问题。'
           '注意，请简要并且清晰地回答问题，如果这个问题可以用"是"或"否"回答，则优先用是或否回答。'
           '如果医生问你是否要开展某个疾病的鉴别诊断程序，回答是。不要回答医生没有问的问题，不要向医生提任何诉求。'
           '如果你无法从临床信息中找到回答问题的信息，请给予否定性回答。例如，如果问你有没有什么症状，回答没有，问你是不是，回答不是。'
           '注意，如果医生问了超过一个问题，请分别回答。'
           '如果医生询问你的现病史信息，你不可以回复其它医院对当前疾病做的相关的诊断。'
           '请注意可能出现的临床缩写。\n临床信息：\n{}\n\n',
    'eng': 'Pretend you are a patient undergoing a medical consultation. Your doctor is asking a question: {}.\n\n'
           'Your prior conversation history is:\n{}\n\nBased on the doctor\'s question and the clinical information '
           'you have (provided below), answer the following question. '
           'Note: Respond briefly and clearly. If the question can be answered with "yes" or "no", '
           'prioritize using "yes" or "no." If the doctor asks whether you want to conduct a differential diagnosis '
           'for a certain disease, answer "yes." Do not answer questions the doctor has not asked, and do not make '
           'any requests to the doctor. If you cannot find information in the clinical data to answer the question, '
           'provide a negative response. For example, if asked whether you have any symptoms, reply "no"; '
           'if asked whether something is true, reply "no". If the doctor asks more than one question, '
           'respond to each question separately. If the doctor inquires about your current medical history, '
           'you may not reference diagnoses made by other hospitals regarding the current illness. '
           'Be aware of potential clinical abbreviations.\nClinical Information:\n{}\n\n'
}


def content_reorganize(content, language):
    dialogue_list, dialogue_string = [], ''
    assert len(content) % 2 == 0
    for i in range(len(content)):
        flag_idx = content[i]['full_response'].find('<RESPONSE>')
        end_idx = content[i]['full_response'].find('</RESPONSE>')

        assert end_idx > 0 and flag_idx >= 0
        start_idx = flag_idx + len('<RESPONSE>')
        turn_content = content[i]['full_response'][start_idx: end_idx]

        if i % 2 == 0:
            assert content[i]['role'] == 'doctor'
            if language == 'chn':
                turn_dialogue_str = f'第{i // 2 + 1}轮，医生说：\n {turn_content} \n'
            else:
                assert language == 'eng'
                turn_dialogue_str = f'Turn #{i // 2 + 1}, Doctor Said: \n {turn_content} \n'
            dialogue_string = dialogue_string + turn_dialogue_str
        else:
            assert content[i]['role'] == 'patient'
            if language == 'chn':
                turn_dialogue_str = f'第{i // 2 + 1}轮，病人说：\n {turn_content} \n\n'
            else:
                assert language == 'eng'
                turn_dialogue_str = f'Turn #{i // 2 + 1}, Patient Said: \n {turn_content} \n'
            dialogue_string = dialogue_string + turn_dialogue_str
        dialogue_list.append(turn_dialogue_str)
    return dialogue_string, dialogue_list


def background_information_organize(info, screen_flag, language):
    no_sign = '无' if language == 'chn' else 'none record'
    key_1, value_1 = '门诊记录' if language == 'chn' else 'outpatient record', info['outpatient_record']
    key_2, value_2 = '入院记录' if language == 'chn' else 'admission record', info['admission_record']
    key_3, value_3 = '大病史' if language == 'chn' else 'comprehensive history', info['comprehensive_history']
    key_4, value_4 = '出院记录' if language == 'chn' else 'discharge record', info['discharge_record']
    key_5, value_5 = '补充信息' if language == 'chn' else 'affiliated_info', info['affiliated_info']
    assert language == 'chn' or language == 'eng'

    if screen_flag == 1:
        context = (f'{key_1}:\n {value_1 if len(value_1) > 4 else no_sign}\n\n' +
                   f'{key_2}:\n {value_2 if len(value_2) > 4 else no_sign}\n\n')
        return context
    context = (f'{key_1}:\n {value_1 if len(value_1) > 4 else no_sign}\n\n' +
               f'{key_2}:\n {value_2 if len(value_2) > 4 else no_sign}\n\n' +
               f'{key_3}:\n {value_3 if len(value_3) > 4 else no_sign}\n\n' +
               f'{key_4}:\n {value_4 if len(value_4) > 4 else no_sign}\n\n' +
               f'{key_5}:\n {value_5 if len(value_5) > 4 else no_sign}\n\n')
    return context


def patient_behavior(client_id, background_information, history, doctor_full_utterance, language, llm_name, streaming):
    # 重新组织对话
    assert len(history) % 2 == 0
    dialogue_string, _ = content_reorganize(history, language)
    empty = '无对话历史' if language == 'chn' else 'NO DIALOGUE  HISTORY'
    dialogue_string = dialogue_string if len(dialogue_string) > 0 else empty
    state_start_idx = doctor_full_utterance.find('<AFFILIATED-INFO>') + len('<AFFILIATED-INFO>')
    state_end_idx = doctor_full_utterance.find('</AFFILIATED-INFO>')
    screen_flag = json.loads(doctor_full_utterance[state_start_idx:state_end_idx])['screen_flag']
    assert screen_flag == 1 or screen_flag == 0

    background_information_available = background_information_organize(background_information, screen_flag, language)
    flag_idx = doctor_full_utterance.find('<RESPONSE>')
    end_idx = doctor_full_utterance.find('</RESPONSE>')
    assert end_idx > 0 and flag_idx >= 0
    start_idx = flag_idx + len('<RESPONSE>')
    show_utterance = doctor_full_utterance[start_idx: end_idx]

    if screen_flag == 1:
        prompt = prompt_screen_template[language].format(show_utterance, dialogue_string,
                                                         background_information_available)
    else:
        prompt = prompt_diagnosis_template[language].format(show_utterance, dialogue_string,
                                                            background_information_available)
    # logger.info(f'client_id: {client_id}, patient response prompt: {prompt}'.replace('\n', ''))

    patient_response_generator = patient_response_generation(
        [prompt], client_id, llm_name, streaming
    )
    return patient_response_generator


async def patient_response_generation(prompt_list, client_id, model_name, streaming):
    if streaming:
        yield generate_streaming_response(prompt_list, client_id, model_name, streaming)
    else:
        yield generate_non_streaming_response(prompt_list, client_id, model_name, streaming)


def generate_non_streaming_response(prompt_list, client_id, model_name, streaming):
    assert not streaming
    assert len(prompt_list) == 1
    logger.info(f'client: {client_id}, question content: {prompt_list[0]}'.replace('\n', ''))
    response_list = []
    for prompt_info in prompt_list:
        response = non_streaming_call_llm(model_name, prompt_info, max_token_num=512)
        response_list.append(response)

    response_str = '<RESPONSE>'+response_list[0]+'</RESPONSE>'
    return json.dumps(set_fake_response(response_str)) + '\n\n$$$\n\n'


async def generate_streaming_response(prompt_list, client_id, model_name, streaming):
    assert len(prompt_list) == 1
    logger.info(f'client: {client_id}, question content: {prompt_list[0]}')
    yield json.dumps(set_fake_response('<RESPONSE>')) + '\n\n$$$\n\n'
    for prompt_info in prompt_list:
        async for result in streaming_call_llm(client_id, model_name, prompt_info):
            yield result
    yield json.dumps(set_fake_response('</RESPONSE>')) + '\n\n$$$\n\n'


def set_fake_response(content):
    data = {
        'choices': [
            {
                'delta': {
                    'content': content
                }
            }
        ]
    }
    return data


def non_streaming_call_llm(model_name, content, max_token_num):
    assert 'o1' not in model_name and 'r1' not in model_name
    model_key = endpoint_info[model_name]['api_key']
    model_url = endpoint_info[model_name]['url']
    model_id = endpoint_info[model_name]['model_name']
    client = openai.OpenAI(
        api_key=model_key,
        base_url=model_url,
    )
    if model_name == 'ultra_medical_llm':
        stop_token = ["<|eot_id|>"]
    else:
        stop_token = None

    completion = client.chat.completions.create(
        model=model_id,
        max_tokens=max_token_num,
        stop=stop_token,
        messages=[{'role': 'assistant', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': content}],
    )
    message = completion.choices[0].message.content
    return message


async def streaming_call_llm(model_name, prompt, stream):
    assert 'o1' not in model_name and 'r1' not in model_name
    assert stream
    model_key = endpoint_info[model_name]['api_key']
    model_url = endpoint_info[model_name]['url']
    client = openai.OpenAI(
        api_key=model_key,
        base_url=model_url,
    )
    if model_name == 'ultra_medical_llm':
        stop_token = ["<|eot_id|>"]
    else:
        stop_token = None
    completion = client.chat.completions.create(
        model=model_name,
        stop=stop_token,
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': prompt}],
        stream=stream,
        temperature=0.0,
        stream_options={"include_usage": True}
    )
    for chunk in completion:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is None:
            continue
        response = chunk.model_dump_json() + '\n\n$$$\n\n'
        yield response


def patient_behavior_wrapper(client_id, context, language, history, doctor_response, llm_name, streaming):
    # 按照原有的设计，patient behavior的输出是兼容open ai api的流式输出
    # 这里要做同步化
    response_generator = patient_behavior(
        client_id=client_id,
        background_information=context,
        history=history,
        doctor_full_utterance=doctor_response,
        language=language,
        llm_name=llm_name,
        streaming=streaming
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        patient_call_llm_results = loop.run_until_complete(collect_responses(response_generator))
    finally:
        loop.close()
    patient_response = parse_return_data(patient_call_llm_results, streaming)
    return patient_response


async def collect_responses(generator):
    result_list = []
    async for res in generator:
        result_list.append(res)
    return result_list


def parse_return_data(response_list, streaming, split_key='\n\n$$$\n\n'):
    content = ''
    if not streaming:
        assert len(response_list) == 1
        data = response_list[0]
        data_list = data.split(split_key)
        for item in data_list:
            if len(item) == 0:
                continue
            data = json.loads(item)
            content += data['choices'][0]['delta']['content']
    else:
        for data in response_list:
            data_list = data.split(split_key)
            for item in data_list:
                if len(item) == 0:
                    continue
                data = json.loads(item)
                if len(data['choices']) > 0:
                    assert len(data['choices']) == 1
                    content += data['choices'][0]['delta']['content']
    return content
