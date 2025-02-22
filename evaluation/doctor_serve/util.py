import csv
import openai
import os
import logging
from itertools import islice
import json
from .doctor_config import args, model_info_json

logger = logging.getLogger('evaluation_logger')
max_new_token_llm = args['max_new_token_llm']


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


def generate_non_streaming_response(prompt_list, client_id, model_name, streaming, new_state_str):
    assert not streaming

    response_list = []
    for prompt_info in prompt_list:
        # logger.info(f'client: {client_id}, question type: {prompt_info[0]}, content: {prompt_info[1]}, '
        #             f'key: {prompt_info[2]}'.replace('\n', ' '))
        response = non_streaming_call_llm(model_name, prompt_info[1])
        response_list.append(response)

    response_str = ('<AFFILIATED-INFO>' + new_state_str +
                    '</AFFILIATED-INFO><RESPONSE>'+'。'.join(response_list)+'</RESPONSE>')
    return json.dumps(set_fake_response(response_str)) + '\n\n$$$\n\n'


def parse_tag_content(response, start_tag, end_tag):
    result_str = response.strip()
    start_index = result_str.find(start_tag) + len(start_tag)
    end_index = result_str.find(end_tag)
    if start_index == -1 or end_index == -1:
        logger.info('format illegal')
        raise ValueError('Invalid result')

    tag_content = result_str[start_index:end_index]
    return tag_content


def parse_question(response, language):
    if language == 'chn':
        start_tag = '<问题>'
        end_tag = '</问题>'
    else:
        assert language == 'eng'
        start_tag = '<Question>'
        end_tag = '</Question>'

    thought_content = parse_tag_content(response, start_tag, end_tag)
    return thought_content


def parse_diagnosis(response, language):
    if language == 'chn':
        start_tag = '<诊断>'
        end_tag = '</诊断>'
    else:
        assert language == 'eng'
        start_tag = '<Diagnosis>'
        end_tag = '</Diagnosis>'

    thought_content = parse_tag_content(response, start_tag, end_tag)
    return thought_content


def parse_thought(response, language):
    if language == 'chn':
        start_tag = '<思考>'
        end_tag = '</思考>'
    else:
        assert language == 'eng'
        start_tag = '<Thought>'
        end_tag = '</Thought>'

    thought_content = parse_tag_content(response, start_tag, end_tag)
    return thought_content


def parse_patient_response_intention(response, target_length, language):
    if language == 'chn':
        question_template = '#问题 {}#'
    else:
        assert language == 'eng'
        question_template = '#Question {}#'
    result_str = response.strip()
    start_index = result_str.find(question_template.format(1))
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
        flag_1 = question_template.format(i + 1) in result
        flag_2 = ('NO' in result and 'YES' not in result) or ('NO' not in result and 'YES' in result)
        if not (flag_1 and flag_2):
            if not flag_1:
                logger.info('count illegal')
            if not flag_2:
                logger.info('format illegal')
            raise ValueError('Invalid result')
        result_list.append(False if 'NO' in result and 'YES' not in result else True)
    return result_list


def non_streaming_call_llm(model_name, content):
    client = openai.OpenAI(
        api_key=endpoint_info[model_name]['api_key'],
        base_url=endpoint_info[model_name]['url'],
    )
    if model_name == 'ultra_medical_llm':
        stop_token = ["<|eot_id|>"]
    else:
        stop_token = None
    completion = client.chat.completions.create(
        model=endpoint_info[model_name]['model_name'],
        max_tokens=max_new_token_llm,
        messages=[{'role': 'assistant', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': content}],
        stop=stop_token
    )
    message = completion.choices[0].message.content

    if 'huatuogpt_o1' in model_name:
        logger.info(f'huatuo o1 original response: {message}')
        if '## Final Response' in message:
            start_index = message.find('## Final Response') + len('## Final Response')
        else:
            start_index = 0
        message = message[start_index:].strip()
    if 'deepseek_r1' in model_name:
        logger.info(f'deepseek original response: {message}')
        if '</think>' in message:
            start_index = message.find('</think>') + len('</think>')
        else:
            start_index = 0
        message = message[start_index:].strip()
    return message


async def streaming_call_llm(model_name, prompt, stream):
    assert stream is True
    client = openai.OpenAI(
        api_key=endpoint_info[model_name]['api_key'],
        base_url=endpoint_info[model_name]['url'],
    )
    if model_name == 'ultra_medical_llm':
        stop_token = ["<|eot_id|>"]
    else:
        stop_token = None

    completion = client.chat.completions.create(
        model=endpoint_info[model_name]['model_name'],
        messages=[{'role': 'assistant', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': prompt}],
        stream=stream,
        stream_options={"include_usage": True},
        stop=stop_token
    )
    for chunk in completion:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content is None:
            continue
        response = chunk.model_dump_json() + '\n\n$$$\n\n'
        yield response


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


def eng_icd_name_mapping(file_path, index_diagnosis_code_map):
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


endpoint_info = json.load(open(model_info_json, 'r', encoding='utf-8-sig'))
