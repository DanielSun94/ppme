import os
import argparse
import logging

logger = logging.getLogger('evaluation_logger')

call_llm_retry_time = 5
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

full_info_cache_folder = os.path.join(project_root, 'resource', 'final_joint_data')
full_diagnosis_file = os.path.join(full_info_cache_folder, 'use_code_map_False_digit_4_fraction_0.95.csv')
full_diagnosis_sample_file = os.path.join(full_info_cache_folder, 'sample_use_code_map_False_digit_4_fraction_0.95.csv')


structured_symptom_folder = os.path.join(project_root, 'resource', 'disease_screen', 'structured_symptom')
evaluation_result_cache_folder = os.path.join(project_root, 'resource', 'evaluation', 'simulation_result')
external_evaluation_result_cache_folder = os.path.join(project_root, 'resource', 'evaluation',
                                                       'external_evaluation_result')
symptom_recover_path_template = os.path.join(project_root, 'resource', 'evaluation', 'symptom_recover_{}.pkl')
evaluation_rerank_cache_folder = os.path.join(project_root, 'resource', 'evaluation', 'rerank')
evaluation_cross_eval_folder = os.path.join(project_root, 'resource', 'evaluation', 'cross_eval')
evaluation_final_diagnosis_eval = os.path.join(project_root, 'resource', 'evaluation', 'final_diagnosis_eval')
final_diagnosis_eval_category_folder = os.path.join(project_root, 'resource', 'evaluation',
                                                    'final_diagnosis_eval_category')
evaluation_cross_icd_parsing_accuracy_file = os.path.join(project_root, 'resource', 'evaluation', 'icd_accuracy.json')
full_info_eval_folder = os.path.join(project_root, 'resource', 'evaluation', 'full_info_eval')
eval_result_cache = os.path.join(project_root, 'resource', 'disease_screen', 'eval_cache.pkl')
symptom_num_path = os.path.join(project_root, 'resource', 'disease_screen', 'symptom_num_{}.pkl')
model_info_json = os.path.join(project_root, 'resource', 'model_info.json')
disease_screen_folder = os.path.join(project_root, 'resource', 'disease_screen')
diagnosis_cache_template = os.path.join(disease_screen_folder, 'diagnosis_cache_{}_{}_{}_{}_{}_{}.pkl')
history_text_folder = os.path.join(project_root, 'resource', 'disease_screen', 'fixed_question_answer')
chn_icd_mapping_path = os.path.join(project_root, 'resource', 'icd_list_debug.csv')
eng_icd_mapping_file = os.path.join(project_root, 'resource', 'd_icd_diagnoses.csv')
srrsh_severe_path = os.path.join(project_root, 'resource', 'icu_patient_idx.json')

os.makedirs(final_diagnosis_eval_category_folder, exist_ok=True)
os.makedirs(evaluation_result_cache_folder, exist_ok=True)

phase = "SCREEN"
diagnosis_mode = 'TOP_HIT'
top_n_differential_diagnosis_disease = 3
screening_maximum_question = 10
language = 'eng'
# gpt_4o_openai, local_qwen_2__5_72b_int4 deepseek_r1_70b local_qwen_2__5_72b_int4 deepseek_r1_remote
# qwen_2__5_72b_it_deepinfra local_qwen_2__5_72b_int4_2 huatuogpt_o1_72b
doctor_llm_name = 'gpt_4o_openai'
patient_llm_name = 'local_qwen_2__5_72b_int4_2'
doctor_type = 'ecdai'
start_index = 300
end_index = 400
maximum_question_per_differential_diagnosis_disease = 8
# srrsh-outpatient srrsh-hospitalization mimic_iv mimic_iii srrsh-hospitalization-severe
filter_criteria = 'mimic'
validation_type = 'internal'
icd_type = 'DISCARD_OL'

# 只对differential diagnosis阶段有效
differential_diagnosis_icd = None


parser = argparse.ArgumentParser(description="evaluation_parser")
parser.add_argument('--language', help='',
                    default=language, type=str)
parser.add_argument('--phase', help='',
                    default=phase, type=str)
parser.add_argument('--top_n_differential_diagnosis_disease', help='',
                    default=top_n_differential_diagnosis_disease, type=int)
parser.add_argument('--maximum_question_per_differential_diagnosis_disease', help='',
                    default=maximum_question_per_differential_diagnosis_disease, type=int)
parser.add_argument('--screening_maximum_question', help='',
                    default=screening_maximum_question, type=float)
parser.add_argument('--diagnosis_mode', help='',
                    default=diagnosis_mode, type=str)
parser.add_argument('--doctor_llm_name', help='',
                    default=doctor_llm_name, type=str)
parser.add_argument('--patient_llm_name', help='',
                    default=patient_llm_name, type=str)
parser.add_argument('--doctor_type', help='',
                    default=doctor_type, type=str)
parser.add_argument('--differential_diagnosis_icd', help='',
                    default=differential_diagnosis_icd, type=str)
parser.add_argument('--start_index', help='',
                    default=start_index, type=int)
parser.add_argument('--end_index', help='',
                    default=end_index, type=int)
parser.add_argument('--filter_criteria', help='',
                    default=filter_criteria, type=str)
parser.add_argument('--validation_type', help='',
                    default=validation_type, type=str)
parser.add_argument('--icd_type', help='',
                    default=icd_type, type=str)
args = vars(parser.parse_args())


args_list = []
for key in args:
    args_list.append([key, args[key]])
args_list = sorted(args_list, key=lambda x: x[0])
for item in args_list:
    logger.info('{}: {}'.format(item[0], item[1]))
