llm_prompt_dict = {
    'opening_question': {
        'chn': "请你扮演一名医生，复述如下的句子，注意，只要回复你需要复述的句子即可，不要回复其它内容。\n复述句子：您好，请问您哪儿不舒服？",
        'eng': 'Please play the role of a doctor and repeat the following sentence. '
               'Note, only reply with the sentence you need to repeat, and do not reply with any additional content.\n'
               'Paraphrase the sentence: Hello, may I ask where you are feeling uncomfortable?'
    },
    'ending': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户用户已经终止，不会再就任何信息进行回复。"
               "注意：回复不要太长，不要透露你在扮演一名医生，直接告诉用户这个信息即可，不要回复其他任何内容",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user.\n"
               "Inform the user that the session has been terminated and no further information will be provided. "
               "Note: Keep the response concise, do not disclose that you are playing the role of a doctor, "
               "and simply convey this information without providing any additional content."
    },
    'response_insufficient_information': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户由于无法获取所需信息，对话已经终止，你不会再就任何信息进行回复。"
               "注意：回复不要太长，不要透露你在扮演一名医生，直接告诉用户这个信息即可，不要回复其他任何内容",
        'eng': "Please assume you are a doctor conducting a diagnostic dialogue with a user.\n"
               "Inform the user that the conversation has been terminated due to the inability to obtain the required "
               "information, and you will no longer respond to any inquiries. Note: Keep the response concise, "
               "do not disclose that you are acting as a doctor, and simply convey this information without "
               "providing any additional content."
    },
    'screening_sufficiency_analysis_template': {
        'chn': "请假设你是一名医生，假设你正在和一名病人进行医疗咨询，以推测病人可能患有的疾病。\n\n历史对话是：{}\n\n"
               "现在，你需要判断既往的对话信息是否已经足够充足来支持你做出疾病初筛判断。如果你觉得信息足够充分，无需继续收集信息，"
               "请回复YES；如果你觉得询问更多的问题的收益不可忽视，则回答NO。\n你需要用YES或NO回答以下问题，"
               "但我们不允许同时回答YES/NO，你只能选一个回复：\n"
               "#问题 1#：你是否认为现有信息已经充分？\n,请根据以下格式作答:\n"
               "#问题 1#：YES/NO",
        'eng': "Please assume you are a doctor, engaging in a medical consultation with a patient to infer possible "
               "diseases.\n\nThe historical conversation is: {}\n\nNow, you need to determine whether the previous "
               "dialogue information is sufficient to support an initial disease screening judgment. "
               "If you believe the information is sufficient and there is no need to gather more, reply YES; "
               "if you think the benefit of asking additional questions is significant, reply NO.\n"
               "You must answer the following question with either YES or NO. "
               "Simultaneous responses such as YES/NO are not allowed; "
               "you can only choose one response:\n"
               "#Question 1#: Do you think collected information is sufficient?\n"
               "Please respond in the following format:\n"
               "#Question 1#: YES/NO"
    },
    'screening_asking_template': {
        'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据对话历史，询问新的有效问题，"
               "以推测病人可能患有的疾病。你每次只可以询问一个简要的问题，不可以同时询问很多个问题。"
               "问题范围局限于询问病人的主诉、现病史、既往史、个人史、家族史，问题范围不包括病人实验室检查和ECG、EEG检查和影像学"
               "(CT、MRI、超声)检查结果。\n\n历史对话是：{}\n\n"
               "请用自然的语言陈述。请只询问你想问的问题，不要生成其他内容，注意，重申，你每次只可以询问一个简要的问题。请只回复提问的内容，"
               "不要生成任何与问题无关的内容，特别是建议患者去就医之类的建议。",
        'eng': "Please assume you are a doctor conducting a medical consultation with a patient. "
               "Based on the conversation history, you need to ask new, relevant questions to infer the possible "
               "diseases the patient may have. You can only ask one concise question at a time and are not allowed "
               "to ask multiple questions simultaneously. The scope of the questions is limited to inquiries about "
               "the patient's chief complaint, history of present illness, past medical history, personal history, "
               "and family history. The questions must not involve the patient's laboratory test results, ECG, EEG, "
               "or imaging (CT, MRI, echo) findings.\n\nThe conversation history is: {}\n\n"
               "Please phrase your question in natural "
               "language. Only ask the question you want to ask, without generating any additional content. "
               "Remember, you are limited to asking one concise question at a time. Do not generate any content "
               "unrelated to the question, especially suggestions such as advising the patient to seek medical "
               "attention."
    },
    'screening_decision_template': {
        'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据和病人的对话历史，推测他有什么疾病。你们的对话历史是：\n{}\n\n。"
               "你需要基于这一堆话，列出造成病人本次入院的最可能的5种疾病，注意，越高风险的疾病应当排在越前面，"
               "你不能输出重复的疾病，疾病的粒度应当尽可能细致。\n你的输出格式应当遵循：\n"
               "#1#: 疾病名称\n#2#: 疾病名称\n#3#: 疾病名称\n#4#: 疾病名称\n#5#: 疾病名称\n",
        'eng': "Please assume you are a doctor conducting a medical consultation with a patient. "
               "Based on the conversation history with the patient, you need to infer their possible diseases. "
               "The conversation history is:\n{}\n\n. Based on this dialogue, you are required to list the five most "
               "likely diseases causing the patient's current hospitalization. Note that higher-risk diseases should "
               "be listed first. You must not output duplicate diseases, and the granularity of the diseases should "
               "be as detailed as possible.\nYour output format should follow:\n#1#: Disease Name\n"
               "#2#: Disease Name\n#3#: Disease Name\n#4#: Disease Name\n#5#: Disease Name\n"
    },
    'diagnosis_conversation_template': {
        'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询，你想要知道病人是否患有{}。你需要根据和病人的对话历史，提出问题或做出诊断。"
               "你的问题范围没有限制，你可以询问任意问题以协助你做出判断，"
               "提问范围包括病人的主诉、现病史、既往史、个人史、家族史，实验室检查和ECG、"
               "EEG检查和影像学检查结果。但是，你每一轮只可以询问一个问题，不可以同时询问多个问题。\n"
               "如果你认为需要继续提问，请生成问题。你只需要生成问题即可，不要回复其它内容。\n"
               "如果你认为信息已经充分，请回复诊断意见。如果你认为他有这一疾病，请回复：#你确诊了{}#，如果你认为他没有得这一疾病"
               "请回复：#你没有得{}#。\n\n你们的对话历史是：\n{}\n\n。",
        'eng': "Please assume you are a doctor consulting with a patient, and you want to determine whether the "
               "patient has {}. Based on the conversation history with the patient, you need to ask questions or "
               "make a diagnosis. There are no limits to the scope of your questions, and you can inquire about "
               "anything to assist in your judgment. The scope of questions includes the patient’s chief complaint, "
               "history of present illness, past medical history, personal history, family history, laboratory tests,"
               " ECG, EEG results, and imaging findings. However, you can only ask one question per round and cannot "
               "ask multiple questions simultaneously.\nIf you think further questions are needed, please generate a "
               "question. You should only generate the question and avoid providing any other response.\n"
               "If you believe the information is sufficient, please respond with a diagnosis. If you think the "
               "patient has the disease, reply with: #You have diagnosed {}#, and if you think the patient does not "
               "have the disease, reply with: #You do not have {}#.\n\nThe conversation history is as follows:\n{}\n\n."
    }
}
