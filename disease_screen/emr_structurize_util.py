import os
from disease_screen_logger import logger
import tiktoken
from openai import OpenAI
from transformers import AutoTokenizer

openai = OpenAI(
    api_key="7tiHQkxe2iWHpHZLHfXWde5jRgNEjcU9",
    base_url="https://api.deepinfra.com/v1/openai",
)


def call_open_ai_embedding(input_text):
    client = OpenAI(
        api_key="TBD"
    )
    embedding = client.embeddings.create(input=input_text, model='text-embedding-3-large').data[0].embedding
    return embedding


def call_open_ai(prompt, model_name):
    # os.environ['http_proxy'] = 'http://127.0.0.1:7890'
    # os.environ['https_proxy'] = 'http://127.0.0.1:7890'
    if 'gpt_4o_mini' in model_name:
        model_id = 'gpt-4o-mini'
    elif 'gpt_4o' in model_name:
        model_id = 'gpt-4o'
    else:
        raise ValueError('')
    success_flag, response = False, None
    while not success_flag:
        try:
            client = OpenAI(
                api_key="TBD"
            )
            chat_completion = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            response = chat_completion.choices[0].message.content
            success_flag = True
        except Exception as err:
            logger.info('Error info: {}'.format(err))
    # if 'http_proxy' in os.environ:
    #     del os.environ['http_proxy']
    # if 'https_proxy' in os.environ:
    #     del os.environ['https_proxy']
    return response


def call_deepinfra(prompt, model_id):
    assert model_id == 'meta-llama/Meta-Llama-3.1-70B-Instruct'
    assert isinstance(prompt, str) or (isinstance(prompt, list) and len(prompt) == 1)
    if isinstance(prompt, list):
        prompt = prompt[0]
    success_flag, response = False, None
    while not success_flag:
        try:
            chat_completion = openai.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
            )
            response = chat_completion.choices[0].message.content
            success_flag = True
        except Exception as err:
            logger.info('Error info: {}'.format(err))
    return [response]


def call_remote_llm(llm_name, prompt):
    if 'gpt' in llm_name:
        response = call_open_ai(prompt, llm_name)
    elif 'llama-3.1-70b' in llm_name.lower():
        response = call_deepinfra(prompt, llm_name)
    else:
        raise ValueError('')
    return response


def call_remote_tokenizer_tokenize(llm_name, prompt):
    llm_name = llm_name.lower()
    if 'gpt' in llm_name:
        enc = tiktoken.encoding_for_model("gpt-4o")
        token_list = enc.encode(prompt)
    else:
        assert 'llama-3.1-70b' in llm_name.lower()
        llama_31_tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/disk_1/llm_cache/LLM-Research/Meta-Llama-3___1-70B-Instruct-GPTQ-INT4',
            use_fast=False,
            padding_side='left',
            trust_remote_code=True,
        )
        token_list = llama_31_tokenizer(prompt)['input_ids']
    return token_list


def call_remote_tokenizer_reverse_tokenize(llm_name, token_list):
    llm_name = llm_name.lower()
    if 'gpt' in llm_name:
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt = enc.decode(token_list)
    else:
        assert 'llama-3.1-70b' in llm_name.lower()
        llama_31_tokenizer = AutoTokenizer.from_pretrained(
            '/mnt/disk_1/llm_cache/LLM-Research/Meta-Llama-3___1-70B-Instruct-GPTQ-INT4',
            use_fast=False,
            padding_side='left',
            trust_remote_code=True,
        )
        prompt = llama_31_tokenizer.decode(token_list)
    return prompt


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
            level_2_list_dict[key] = construct_level_two_question_list(
                key, secondary_questions_dict, language, maximum_questions)
        else:
            level_2_list_dict[key] = None
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
