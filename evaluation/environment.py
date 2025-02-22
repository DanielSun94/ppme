import json
from evaluation_logger import logger
from evaluation_config import args

logger.info('start evaluation environment')
language_ = args['language']
phase_ = args['phase']
screening_maximum_question_ = args['screening_maximum_question']
top_n_differential_diagnosis_disease_ = args['top_n_differential_diagnosis_disease']
diagnosis_mode_ = args['diagnosis_mode']
doctor_type = args['doctor_type']
differential_diagnosis_icd = args['differential_diagnosis_icd']
maximum_question_per_differential_diagnosis_disease = args['maximum_question_per_differential_diagnosis_disease']


class Environment(object):
    def __init__(self,
                 patient_behavior_func,
                 doctor_behavior_func,
                 patient_info_context,
                 phase,
                 data_id,
                 doctor_llm_name,
                 patient_llm_name,
                 diagnosis_mode,
                 language,
                 screening_maximum_question,
                 top_n_differential_diagnosis_disease,
                 diagnosis_target
                 ):
        """
        根据目前的设计，patient_simulator和doctor_simulator本身都不保存状态（但不是完全没有状态，只是状态会在外面保存）
        :param patient_behavior_func:
        :param doctor_behavior_func:
        """
        assert phase == 'ALL' or phase == 'SCREEN' or phase == 'DIAGNOSIS'

        self.patient_behavior = patient_behavior_func
        self.doctor_behavior = doctor_behavior_func
        self.phase = phase
        self.patient_info_context = patient_info_context
        self.data_id = data_id
        self.doctor_llm_name = doctor_llm_name
        self.patient_llm_name = patient_llm_name
        self.language = language
        self.diagnosis_mode = diagnosis_mode
        self.screening_maximum_question = screening_maximum_question
        self.top_n_differential_diagnosis_disease = top_n_differential_diagnosis_disease
        self.diagnosis_target = diagnosis_target
        self.streaming = False

        self.end_flag = False
        self.dialogue_history = []

    def step(self):
        # 定义一个协程函数来收集异步生成器的结果
        history = self.dialogue_history
        context = self.patient_info_context

        # doctor info generation
        doctor_response = self.doctor_behavior(
            messages=history,
            client_id=self.data_id,
            model_name=self.doctor_llm_name,
            phase=self.phase,
            screening_maximum_question=self.screening_maximum_question,
            top_diagnosis_disease_num=self.top_n_differential_diagnosis_disease,
            diagnosis_target=self.diagnosis_target,
            diagnosis_mode=self.diagnosis_mode,
            environment_language=self.language,
            maximum_question_per_differential_diagnosis_disease=maximum_question_per_differential_diagnosis_disease
        )
        start_idx = doctor_response.find('<RESPONSE>') + len('<RESPONSE>')
        end_idx = doctor_response.find('</RESPONSE>')
        doctor_show_response = doctor_response[start_idx:end_idx]
        start_idx = doctor_response.find('<AFFILIATED-INFO>') + len('<AFFILIATED-INFO>')
        end_idx = doctor_response.find('</AFFILIATED-INFO>')
        end_flag = json.loads(doctor_response[start_idx:end_idx])['end_flag']
        assert end_flag == 1 or end_flag == 0

        if end_flag == 0:
            # patient info generation
            patient_response = self.patient_behavior(
                client_id='local',
                context=context,
                history=history,
                doctor_response=doctor_response,
                language=self.language,
                llm_name=self.patient_llm_name,
                streaming=False
            )

            start_idx = patient_response.find('<RESPONSE>') + len('<RESPONSE>')
            end_idx = patient_response.find('</RESPONSE>')
            patient_show_response = patient_response[start_idx:end_idx]
        else:
            patient_response = '<RESPONSE>END</RESPONSE>'
            patient_show_response = 'END'

        logger.info(f'doctor show response: {doctor_show_response}')
        logger.info(f'patient show response: {patient_show_response}')
        history.append({
            'role': 'doctor',
            'full_response': doctor_response,
            'show_response': doctor_show_response
        })
        history.append({
            'role': 'patient',
            'full_response': patient_response,
            'show_response': patient_show_response
        })
        return end_flag

