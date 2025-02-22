ecdai_prompt_dict = {

    'first_question': {
        'chn': "请扮演一名医生，复述如下的句子，注意，只要回复你需要复述的句子即可，不要回复其它内容。\n"
               "复述句子：您好，请问您哪儿不舒服？",
        'eng': 'Please act as a doctor and repeat the following sentence. Note, only reply with the sentence you need '
               'to repeat, and do not include any other content.\nRepeated sentence: '
               'Hello, may I ask where you are feeling uncomfortable?'
    },
    'ending': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户用户已经终止，不会再就任何信息进行回复。"
               "注意：回复不要太长，不要透露你在扮演一名医生，直接告诉用户这个信息即可，不要回复其他任何内容",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user.\n"
               "Inform the user that the user has been terminated and will no longer respond to any information. "
               "Note: Keep the response brief, do not reveal that you are acting as a doctor, and simply convey "
               "this information without providing any additional content."
    },
    'response_medical_question_template': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。在上一轮对话中，用户的回复是：{}。\n既往的完整对话是：\n{}\n。"
               "请你仔细思考，并简明扼要的回答用户在上一轮中提出的医学问题。回复不要太长，直接回答问题即可。请不要反问用户问题，直接给出回答。"
               "若用户的问题很难精准回答，给出宽泛的回答即可。针对患者提出的非医学问题和诉求，请不要做任何回复。如果用户的问题是要求你诊断疾病，"
               "请不要回答，回复：请回答诊断系统的问题，我们会适时给出诊断建议。",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user. "
               "In the previous round, the user's response was: {}.\nThe full prior conversation is:\n{}\n."
               " Please think carefully and provide a concise answer to the medical question the user raised in the "
               "previous round. Keep your response brief and direct, without asking the user any questions. "
               "If the user's question is difficult to answer precisely, provide a broad response. Do not respond "
               "to non-medical questions or requests from the user. If the user's question asks you to diagnose a "
               "disease, reply: Please answer the diagnostic system's questions, "
               "and we will provide diagnostic suggestions at the appropriate time."
    },
    'response_non_medical_question_template': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户其诉求中和医学无关的问题不会被回答。回复不要太长。",
        'eng': 'Please assume you are a doctor conducting a diagnostic conversation with a user.\n'
               'Inform the user that questions unrelated to medical issues will not be answered. '
               'Keep the response brief.'
    },
    'response_insufficient_information': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。\n请告诉用户由于无法获取所需信息，对话已经终止，你不会再就任何信息进行回复。"
               "注意：回复不要太长，不要透露你在扮演一名医生，直接告诉用户这个信息即可，不要回复其他任何内容",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user.\n"
               "Inform the user that the conversation has been terminated due to the inability to obtain "
               "the required information, and you will no longer respond to any information. "
               "Note: Keep the response brief, do not reveal that you are acting as a doctor, "
               "and simply convey this information without providing any additional content."
    },
    'last_interaction_parse_template': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。你们的对话记录是：\n{}\n你需要用YES或NO回答以下问题，"
               "但我们不允许同时回答YES/NO，你只能选一个回复：\n"
               "问题 1：用户是否回答了对话中的最后一个问题？。注意，用户如果说他不知道，应当视为回答了问题，回复YES。"
               "但如果用户对问题的内容不理解而进行提问，应当视为没有回答，回复NO。\n"
               "问题 2：在最后一轮对话中，用户是否询问了与医学相关的问题？如果用户的回复中包含与医学相关的任何问题，"
               "包括但不限于药物的服用方式，疾病的症状等信息，对既往问题中的内容不理解而提问，请判定为YES，"
               "如果患者没有提及任何医学相关的问题，则判定为NO。)\n"
               "问题 3：在最后一轮对话中，用户是否询问或发起了与医学无关的问题或指令？如果用户的发起了与医学无关的问题，请返回YES。"
               "如果用户发起了指令让模型做一些与医学无关的任务，请返回YES。\n"
               "问题 4：用户是否已经在这些对话中介绍了自己的性别和年龄？\n"
               "问题 5：用户是否已经在这些对话中介绍了自己的既往病史信息？\n"
               "问题 6：用户是否已经在这些对话中介绍了自己当前的症状（任何现有症状或者体征）？\n"
               "请返回NO,若用户的回复不涉及与医学无关的问题，也不涉及与医学无关的指令，请返回NO。请根据以下格式作答:\n"
               "#问题 1#：YES/NO\n#问题 2#：YES/NO\n#问题 3#：YES/NO\n#问题 4#：YES/NO\n#问题 5#：YES/NO\n#问题 6#：YES/NO\n",
        'eng': "Please assume you are a doctor engaged in a diagnostic conversation with a user. "
               "The conversation transcript is as follows:\n{}\nYou are required to answer the following "
               "questions with YES or NO. However, you are not allowed to answer both YES/NO; you must choose "
               "one response for each question:\n\nQuestion 1: Did the user answer the last question in the "
               "conversation? Note that if the user says they do not know, this should be considered as having "
               "answered the question, and you should respond with YES. However, if the user asked a clarifying "
               "question because they did not understand the content of the question, it should be considered as "
               "not having answered, and you should respond with NO.\n\nQuestion 2: In the last round of "
               "conversation, did the user ask any medically related questions? If the user's response contains "
               "any medically related questions, including but not limited to information about medication usage, "
               "symptoms of diseases, or clarification questions about prior medical issues, you should answer YES. "
               "If the user did not mention any medically related questions, respond with NO.\n\nQuestion 3: In the "
               "last round of conversation, did the user ask or initiate any non-medical questions or commands? If "
               "the user raised any non-medical questions, please respond with YES. If the user initiated a command "
               "asking the model to perform tasks unrelated to medicine, also respond with YES. If the user's reply "
               "does not involve non-medical questions or non-medical commands, please respond with NO.\n\n"
               "Question 4: Has the user introduced their gender and age in this conversation?\n\nQuestion 5: "
               "Has the user introduced any information about their medical history in this conversation?\n\n"
               "Question 6: Has the user introduced their current symptoms (any existing symptoms or signs) in "
               "this conversation?\n\nPlease answer in the following format:\n#Question 1#: YES/NO\n"
               "#Question 2#: YES/NO\n#Question 3#: YES/NO\n#Question 4#: YES/NO\n#Question 5#: YES/NO\n"
               "#Question 6#: YES/NO\n"
    },
    'screening_asking_template': {
        'chn': "请假设你是一名医生，你需要询问病人这个问题，请用自然的语言陈述。请只回复问题，不要回复其他信息。\n内容:\n: {}",
        'eng': "Please assume you are a doctor, and you need to ask the patient this question. "
               "Please state it in natural language. Only respond with the question, "
               "without providing any additional information.\nContent:\n: {}"
    },
    'screening_ask_demographic_medical_history': {
        'chn': "请假设你是一名医生，请询问病人的性别年龄和既往史。请简要询问。请不要回复除了问句以外的内容。",
        'eng': "Please assume you are a doctor. Ask the patient about their gender, age, and medical history. "
               "Keep the questions brief. Do not respond with anything other than questions."
    },
    'screening_ask_demographic': {
        'chn': "请假设你是一名医生，请询问病人的性别年龄。请简要询问。请不要回复除了问句以外的内容",
        'eng': "Please assume you are a doctor and ask the patient about their gender and age. "
               "Keep the question concise. Do not respond with anything other than the question."
    },
    'screening_ask_medical_history': {
        'chn': "请假设你是一名医生，请询问病人的既往史。请简要询问。请不要回复除了问句以外的内容",
        'eng': "Please assume you are a doctor and ask the patient about their medical history. "
               "Keep the questions brief. Do not respond with anything other than questions."
    },
    'screening_ask_chief_complaint': {
        'chn': "请假设你是一名医生，请询问病人的当前症状。请简要询问。请不要回复除了问句以外的内容。",
        'eng': "Please assume you are a doctor and ask the patient about their current symptoms. "
               "Keep the question brief. Do not respond with anything other than the question."
    },
    'screening_decision_template': {
        'chn': '请你扮演一名医生讲话，但是不要在对话中说你正在扮演一名医生，直接仿照医生的口吻说话即可。'
               '你现在需要告知病人他的高风险疾病。但是注意，不要说你做了任何检查检验，只说根据目前掌握的信息，推测患者可能有XX疾病。'
               '请你以较为自然的语言告知，特别是对在对疾病名进行描述时，有些疾病的名字比较拗口（例如“非特指的XX病”，“未特指的XX病”，'
               '“其它XX病”，“XX病伴有/不伴有XX”）请你修改成更为自然的表述。注意，只回复疾病即可，不要回复其它任何内容，例如生活和干预建议。'
               '请使用较为简洁的语言表述。患者最高风险的疾病可能是一个大类，可以细分为更为精确的疾病，这些细分疾病会被在下文枚举。'
               '如果存在，请简要的挑一两个提一下，无需全部提及。\n患者风险最高的疾病是：{}。',
        'eng': "Please speak as if you were a doctor, but do not mention in the conversation that you are playing "
               "the role of a doctor. Simply speak in a doctor-like tone. You now need to inform the patient "
               "about their high-risk disease. However, note that you should not mention conducting any tests "
               "or examinations; just state that, based on the information currently available, it is speculated "
               "that the patient may have XX disease. Use more natural language to communicate this, particularly"
               " when describing disease names, as some names can sound cumbersome (e.g., \"nonspecific XX disease,\" "
               "\"unspecified XX disease,\" \"other XX diseases,\" \"XX disease with/without XX\"). "
               "Modify such terms to make them sound more natural. Only provide the disease name; "
               "do not include any other content, such as lifestyle or intervention advice. Use concise language. "
               "The patient's highest-risk disease may be a broad category that can be subdivided into more "
               "specific diseases. If so, briefly mention one or two examples without listing all of them.\n"
               "The patient’s highest-risk disease is: {}."
    },
    'embedding_input_generation_template': {
        'eng': (
            "Please act as a patient and answer the following four questions based on the given DIALOGUE HISTORY "
            "record (attached). Respond in natural sentences and keep your answers concise; "
            "do not make them overly lengthy. Please respond as if you were a real patient."
            "Do not provide any information other than the answers to the questions. "
            "Your answer should be written in English, regardless the input language\n"
            "Question 1: Please tell me your gender and age. If this information is not recorded in the "
            "electronic medical record, please respond with 'I don't know.'\n"
            "Question 2: Please tell me your past medical history (previous illnesses) and current medical condition "
            "(current illnesses). If this information is not recorded in the dialogue history, please "
            "respond with 'I don't know.' Do not provide information that is not present in the dialogue history. "
            "Please do not reply symptoms or illness that causes your current admission.\n"
            "Question 3: Do you know any illnesses that your family members have had? If this information is not "
            "recorded in the dialogue history, please respond with 'I don't know.'\n"
            'Question 4: Where are you feeling unwell?\n'
            "Please respond in the following format:\n"
            "#Start#\n"
            "#1#: (answer)\n#2#: (answer)\n#3#: (answer)\n#4#: (answer)\n#End#\n"
            "The DIALOGUE HISTORY data is as follows:\n"
        ),
    'chn': (
        '请你扮演一名病人，根据给定的对话历史(下附)回答如下四个问题。请用自然的句子简要回复，不要回复的过于冗长，请回复的像一个真实的患者一样。'
        '除了问题的回答，不要回答其它任何信息。\n'
        '无论输入是何种语言，你的输出应当是中文。\n'
        '问题1：请告诉我你的性别，年龄信息。如果对话中未记录相关信息，请回答我不知道。\n'
        '问题2：请告诉我你的既往史（以前得过什么病）和现病史（现在正在得什么病）？如果对话中未记录相关信息，请回答我不知道，'
        '请不要回答对话中不存在的信息。你不可以在这个问题的回答中提到你本次入院的主诉、症状和诊断疾病\n'
        '问题3：请问你家里人之前得过什么病？如果对话中未记录相关信息，请回答我不知道。\n'
        '问题4：请问你最近哪儿不舒服？如果对话中未记录相关信息，请回答我不知道。\n\n'
        '请按照如下格式回复:\n'
        '#回答开始#\n\n'
        '#1#: (answer)\n'
        '#2#: (answer)\n'
        '#3#: (answer)\n'
        '#4#: (answer)\n'
        '#回答结束#\n'
        "对话数据下附：\n"
        )
    },
    'diagnosis_confirm_template': {
        'chn': '请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n你确诊了{}',
        'eng': "Please play the role of a patient and answer the following three questions based on the given "
               "conversation (attached below). Please respond concisely in natural sentences, avoiding overly "
               "lengthy replies, and answer as if you were a real patient. Do not provide any information other "
               "than the answers to the questions.\n\nQuestion 1: Please tell me your gender and age. If this "
               "information is not recorded in the conversation, please answer \"I don't know.\"\n\nQuestion 2: "
               "Please tell me your medical history (diseases you had in the past or chronic conditions, not your "
               "current illness). If this information is not recorded in the conversation, please answer "
               "\"I don't know,\" and do not include information not present in the conversation.\n\n"
               "Question 3: Do you know what illnesses your family members have had? If this information "
               "is not recorded in the conversation, please answer \"I don't know.\"\n\n"
               "Please respond using the following format:\n"
               "#Response Start#\n#1#: (answer)\n#2#: (answer)\n#3#: (answer)\n#Response End#\n\n"
               "The conversation data is as follows:\n{}\n",
    },
    'diagnosis_exclude_template': {
        'chn': '请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n你没有得{}',
        'eng': "Please act as a doctor and say the following in fluent sentences. "
               "Without altering the original meaning, strictly restate it as accurately as possible without "
               "adding anything extra.\nYou do not have {}"
    },
    'diagnosis_procedure_start_inquiry_template': {
        'chn': '请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n我们还怀疑患有{}的风险较高。'
               '我们希望开展鉴别诊断，为了完成鉴别诊断，你可能需要{}的检查结果（但不一定是全部需要）。请问你是否要开始鉴别诊断流程？',
        'eng': "Please act as a doctor and say the following in a fluent manner. "
               "Strictly paraphrase without altering the original meaning, and do not add any additional content.\n"
               "We also suspect a higher risk of {}. We would like to proceed with a differential diagnosis. "
               "To complete this process, you may need the results of {} tests (though not necessarily all of them). "
               "Would you like to begin the differential diagnosis process?"
    },
    'diagnosis_procedure_proceeding_template': {
        'chn': '请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n{}',
        'eng': 'Please act as a doctor and fluently say the following statement. '
               'Please strictly repeat it without altering the original meaning or adding any personal input.\n{}'
    },
    'diagnosis_internal_forward_template': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。你现在想要问的问题是：\n{}\n你手头已经有了一份之前和用户的对话，"
               "内容是：\n{}\n\n请根据这个对话内容，用YES或NO回答以下问题，但我们不允许同时回答YES/NO，你只能选一个回复："
               "#问题 1#：之前的对话内容中是否已经包含了你现在想要问的问题的答案？如果包含请回复YES，不包含请回复NO，注意"
               "是你现在要问的问题，不是之前对话中已经问过的问题。\n"
               "#问题 2#：请问，之前的对话中是承认了你现在询问的问题，还是否认了你现在询问的问题？如果承认了，回答YES；"
               "如果否认了，回答NO。如果用户如果问题1的答案是NO，则直接返回NO。如果用户的回复无法用“承认”或“否认”形容，统一回复NO。\n"
               "你需要按照下面的格式进行回复。\n#问题 1#：YES/NO\n#问题 2#：YES/NO\n",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user. "
               "The question you currently want to ask is:\n{}\nYou already have a record of a previous conversation "
               "with the user, and the content is:\n{}\n\nBased on this conversation, answer the following questions "
               "using YES or NO, but you are not allowed to answer both YES and NO simultaneously. "
               "You must choose only one response:\n#Question 1#: Does the previous conversation already contain "
               "the answer to the question you currently want to ask? If it does, reply with YES; if it does not, "
               "reply with NO. Note that this pertains to the question you currently want to ask, not the questions "
               "already asked in the previous conversation.\n#Question 2#: Did the previous conversation affirm or "
               "deny the question you are currently asking? If it affirmed, reply YES; if it denied, reply NO. "
               "If the answer to Question 1 is NO, directly reply with NO. If the user's response cannot be "
               "categorized as \"affirm\" or \"deny,\" reply with NO.\n"
               "You need to respond in the following format:\n#Question 1#: YES/NO\n#Question 2#: YES/NO\n"
    },
    'diagnosis_patient_agreement_parse_template': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。上一轮你问：\n{}\n用户回答：{}\n\n你需要用YES或NO回答以下问题，"
               "但我们不允许同时回答YES/NO，你只能选一个回复：\n#问题 1#：在对话中，你询问用户是否要开始鉴别诊断，请问用户的想要开展吗？"
               "如果想要开始，回复YES，不想要开始，回复NO\n你需要按照下面的格式进行回复。\n#问题 1#：YES/NO\n",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user. "
               "In the previous round, you asked:\n{}\nThe user responded:\n{}\n\nYou need to answer the "
               "following question with either YES or NO, but you are not allowed to answer both YES/NO at "
               "the same time—you must choose only one response:\n#Question 1#: In the conversation, you asked "
               "the user whether they want to start a differential diagnosis. Does the user want to proceed? "
               "If they want to start, reply YES; if they do not want to start, reply NO.\nYou need to respond "
               "using the format below.\n#Question 1#: YES/NO\n"
    },
    'diagnosis_patient_answer_parse_template': {
        'chn': "请假设你是一位医生，正在对一名用户进行诊断对话。上一轮你问：\n{}\n用户回答：{}\n\n你需要用YES或NO回答以下问题，"
               "但我们不允许同时回答YES/NO，你只能选一个回复：\n"
               "#问题 1#：用户是否知道问题问的内容的结果？如果用户的回复可以推测出他知道这一结果，请回复YES。反之请回复NO。\n"
               "#问题 2#：用户承认了最后一个问题中的内容存在，还是否认了最后一个问题中问的内容存在？"
               "如果用户承认了，回答YES，如果否认了，回答NO。如果问题1的答案是NO，则直接返回NO。"
               "如果用户的回复无法用“承认”或“否认”形容，统一回复NO。\n"
               "你需要按照下面的格式进行回复。\n#问题 1#：YES/NO\n#问题 2#：YES/NO\n",
        'eng': "Please assume you are a doctor conducting a diagnostic conversation with a user. "
               "In the previous round, you asked:\n{}\nThe user responded:\n{}\n\nYou need to answer the "
               "following questions with YES or NO, but you are not allowed to respond with both "
               "YES/NO simultaneously; you can only choose one answer:\n#Question 1#: Does the user know the result "
               "of what the question is asking? If the user's response suggests they know the result, reply YES. "
               "Otherwise, reply NO.\n#Question 2#: Did the user acknowledge the existence of the content in the "
               "last question, or did they deny the existence of the content in the last question? If the user "
               "acknowledged it, reply YES. If they denied it, reply NO. If the answer to Question 1 is NO, "
               "directly return NO. If the user's response cannot be described as \"acknowledge\" or \"deny,\" "
               "reply NO as well.\nYou need to reply in the following format:\n"
               "#Question 1#: YES/NO\n#Question 2#: YES/NO\n"
    },
    'diagnosis_failed_diagnosis_procedure_missing_template': {
        'chn': "请你扮演一名初级医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n内容：我怀疑你有{}，"
               "但是由于我没有鉴别诊断他的能力，因此建议你去专业医生处寻求帮助。",
        'eng': "Please act as a junior doctor and phrase the following statement fluently. "
               "Ensure you strictly restate it without altering the original meaning or adding any personal "
               "interpretation.\nContent: I suspect you have {}, but since I lack the ability to make a "
               "differential diagnosis, I recommend seeking help from a specialist doctor."
    },
    'diagnosis_failed_not_hit': {
        'chn': "请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n"
               "内容：我们已经就所有怀疑的疾病开展了问询，本对话已经结束，我们很遗憾没有确诊任何一种疾病，请您赴专业机构进行咨询",
        'eng': "Please act as a doctor and rephrase the following statement fluently. "
               "Ensure that the original meaning remains unchanged, and refrain from making any modifications or "
               "improvisations.\nContent: We have inquired about all suspected diseases. This conversation has "
               "now concluded. We regret that no specific diagnosis was made. Please consult a professional "
               "institution for further evaluation."
    },
    'diagnosis_lab_test_exam_summary_template': {
        'chn': "你现在有若干疾病的诊断流程（下面列的）。你需要仔细阅览这些流程，然后返回确诊这些疾病涉及的实验室检查和影像学检查项目。"
               "你只需要合并相同的检查项目，并且返回项目名称即可，请不要回复除了项目名称外的任何信息。诊断流程列表：\n{}",
        'eng': "You now have several diagnostic workflows for diseases (listed below). "
               "You need to carefully review these workflows and then return the laboratory tests and "
               "imaging studies involved in diagnosing these diseases. You only need to merge duplicate "
               "test items and return the names of the items. Please do not respond with anything other "
               "than the item names. Diagnostic workflow list:\n{}"
    },
    'diagnosis_accomplish_end': {
        'chn': '请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述，不要自行发挥。\n'
               '内容：我们已经就所有怀疑的疾病开展了问询，本对话已经结束。',
        'eng': "Please act as a doctor and deliver the following message in a fluent manner. "
               "Strictly repeat the content as closely as possible without altering the meaning or adding anything "
               "extra.\n Content: We have inquired about all suspected diseases. This conversation has now concluded."
    },
    'diagnosis_confirm_end': {
        'chn': '请你扮演一名医生，请语句通顺的说下面的话。请在不改变原意的情况下，尽可能严格复述。\n'
               '内容：我们已经确诊了您的疾病，本对话已经结束',
        'eng': 'Please act as a doctor and rephrase the following statement fluently. Ensure you strictly restate '
               'it without altering its original meaning as much as possible.\nContent: We have confirmed your '
               'diagnosis, and this conversation has now concluded.'
    }
}
