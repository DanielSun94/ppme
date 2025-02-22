llm_prompt_dict = {
    'opening_question': {
        'chn': "请你扮演一名医生，复述如下的句子，注意，只要回复你需要复述的句子即可，不要回复其它内容。\n"
               "复述句子：您好，请问您哪儿不舒服？",
        'eng': 'Please act as a doctor and repeat the following sentence. Note, only reply with the sentence you '
               'need to repeat; do not reply with anything else.\nRepeat the sentence: Hello, may I ask where you '
               'are feeling uncomfortable?'
    },
    'screening_sufficiency_analysis_template': {
        'chn': "请假设你是一名医生，假设你正在和一名病人进行医疗咨询，以推测病人可能患有的疾病。\n\n历史对话是：{}\n\n"
               '现在，你需要判断既往的对话信息是否已经足够充足来支持你做出疾病初筛判断。你的回复中应当包含"思考"和对问题 1的回答。'
               '"思考"部分中，你需要回顾已经完成的对话，并对已经收集的信息是否完全进行分析。'
               '对问题 1的回答中，你需要明确输出当前已收集的信息是否足够支撑你做出疾病筛查。如果你觉得信息已经充分，请回复YES；'
               "如果你觉得询问更多的问题的收益不可忽视，则回答NO。\n你需要用YES或NO回答以下问题，"
               "但我们不允许同时回答YES/NO，你只能选一个回复：\n"
               "#问题 1#：你认为现有的信息是否充分？\n,"
               "请根据以下格式作答:\n"
               "<思考>思考内容</思考>\n"
               "#问题 1#：YES/NO\n",
        'eng': "Please assume you are a doctor conducting a medical consultation with a patient to infer potential "
               "diseases.\n\nThe historical conversation is: {}\n\nNow, you need to determine whether the "
               "information from the previous conversation is sufficient to support an initial disease screening "
               "judgment. Your response should include a \"Thought\" section and an answer to Question 1. "
               "In the \"Thought\" section, you need to review the completed conversation and analyze whether the "
               "collected information is complete. For Question 1, you must explicitly state whether the "
               "currently collected information is sufficient to support disease screening. If you believe the "
               "information is sufficient, reply YES; if you think asking more questions could provide significant "
               "benefits, reply NO.\n\nYou need to answer the following question with YES or NO, but you cannot "
               "answer with both YES/NO. You must choose only one response:\n\n#Question 1#: Do you think collected "
               "information is sufficient?\n\nPlease respond in the following format:\n"
               "<Thought>Thought content</Thought>\n#Question 1#: YES/NO"
    },
    'screening_asking_template': {
        'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据对话历史，询问新的有效问题，"
               '你的回复中应当包含"思考"和"问题"两部分。'
               '"思考"部分中，你需要回顾已经完成的对话，分析并判断问什么问题可能是最有帮助的。在"问题"，你需要明确输出一个问题。\n'
               "注意：你每次只可以询问一个简要的问题，不可以同时询问很多个问题。"
               "问题范围局限于询问病人的主诉、现病史、既往史、个人史、家族史，问题范围不包括病人实验室检查和ECG、EEG检查和影像学"
               "(CT, MRI, 超声)检查结果。"
               "\n\n历史对话是：{}\n\n"
               "请用自然的语言陈述。请只询问你想问的问题，不要生成其他内容，注意，重申，你每次只可以询问一个明确的问题。"
               "你的输出应当遵循以下的类标签语言格式进行："
               "<思考>思考内容</思考>\n"
               "<问题>问题内容</问题>\n",
        'eng': "Please assume you are a doctor consulting with a patient. Based on the dialogue history, "
               "you need to ask a new, effective question. Your response should include two sections: \"Thought\" "
               "and \"Question\". In the \"Thought\" section, review the completed dialogue, analyze it, "
               "and determine what question might be most helpful to ask next. In the \"Question\" section, "
               "clearly output one specific question. \nNote: You may only ask one brief question at a time "
               "and cannot ask multiple questions simultaneously. The scope of your questions is limited to the "
               "patient's chief complaint, present illness, past medical history, personal history, or family history. "
               "Questions outside these areas, such as those regarding laboratory test results, ECG, EEG, or imaging "
               "examination (CT, MRI, echo) results, are not allowed.\n\nThe historical dialogue is: {}\n\n"
               "Please phrase your "
               "question in natural language. Only ask the question you want to ask without generating any additional "
               "content. Note again, you are allowed to ask only one clear question at a time. Your output should "
               "follow the format below:\n <Thought>Content of your thoughts</Thought>\n "
               "<Question>Content of your question</Question>\n"
    },
    'screening_decision_template': {
        'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据和病人的对话历史，推测他有什么疾病。你们的对话历史是：\n{}\n\n。"
               "你需要基于以下疾病清单表，列出造成病人本次入院的最可能的5种疾病。注意：你的回复中应当包含\"思考\"和\"诊断\"两部分。"
               "\"思考\"部分中，你需要回顾对话，分析病人身体状况，推测可能存在的问题。在\"诊断\"中，你需要基于这一堆话，"
               "列出造成病人本次入院的最可能的5种疾病，注意，越高风险的疾病应当排在越前面。"
               "你不能输出重复的疾病，疾病的粒度应当尽可能细致。\n你的输出格式应当遵循：\n"
               "<思考>思考内容</思考>\n"
               "<诊断>\n#1#: 疾病名称\n#2#: 疾病名称\n#3#: 疾病名称\n#4#: 疾病名称\n#5#: 疾病名称\n</诊断>",
        'eng': "Please assume you are a doctor, conducting a medical consultation with a patient. "
               "Based on the patient's dialogue history, you need to infer their possible illnesses. "
               "The dialogue history is as follows:\n{}\n\n. Based on the provided list of diseases, "
               "you need to identify the top 5 most likely diseases causing this patient's current hospitalization. "
               "Note: Your response should include two parts: \"Thought\" and \"Diagnosis\". "
               "In the \"Thought\" section, you need to review the conversation, analyze the patient's physical "
               "condition, and hypothesize potential problems. In the \"Diagnosis\" section, you should list the "
               "top 5 most likely diseases causing the patient's current hospitalization based on the conversation. "
               "Higher-risk diseases should be listed first. Avoid listing duplicate diseases, and ensure the disease "
               "granularity is as specific as possible.\nYour output format should follow:\n"
               "<Thought>Thinking content</Thought>\n<Diagnosis>\n#1#: Disease name\n#2#: Disease name\n"
               "#3#: Disease name\n#4#: Disease name\n#5#: Disease name\n</Diagnosis>"
    },
    'diagnosis_conversation_template': {
        'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询，你想要知道病人是否患有{}。你需要根据和病人的对话历史，提出问题或做出诊断。"
               "你的问题范围没有限制，你可以询问任意问题以协助你做出判断，"
               "提问范围包括病人的主诉、现病史、既往史、个人史、家族史，实验室检查和ECG、"
               "EEG检查和影像学检查结果。但是，你每一轮只可以询问一个问题，不可以同时询问多个问题。\n"
               "如果你认为需要继续提问，请生成问题。你只需要生成问题即可，不要回复其它内容。\n"
               "如果你认为信息已经充分，请回复诊断意见。如果你认为他有这一疾病，请回复诊断意见：#你确诊了{}#，如果你认为他没有得这一疾病"
               "请回复诊断意见：#你没有得{}#。\n\n你们的对话历史是：\n{}\n\n。"
               "你的输出应当遵循以下的类标签语言格式进行："
               "<思考>思考内容</思考>\n"
               "<问题>问题内容</问题>\n"
               "或者:\n"
               "<思考>思考内容</思考>\n"
               "<诊断>诊断意见</诊断>\n",
        'eng': "Please assume you are a doctor consulting with a patient. Based on the conversation history with "
               "the patient, you need to infer what diseases they might have. The conversation history is: \n{}\n\n. "
               "Using the following list of diseases, list the 5 most likely diseases causing the patient’s current "
               "hospitalization. Note: Your response should include two sections: \"Thought\" and \"Diagnosis\". "
               "In the \"Thought\" section, you need to review the conversation, analyze the patient's physical "
               "condition, and infer potential issues. In the \"Diagnosis\" section, you need to list the 5 most "
               "likely diseases causing the patient's current hospitalization based on this information. "
               "Higher-risk diseases should be ranked higher. You must not output duplicate diseases, "
               "and the diseases should be as detailed as possible. Your output format should follow:"
               "<Thought>Thinking content</Thought><Diagnosis>#1#: Disease Name\n#2#: Disease Name\n"
               "#3#: Disease Name\n#4#: Disease Name\n#5#: Disease Name\n</Diagnosis>"
    }
}
