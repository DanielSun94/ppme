import os
import argparse
import logging

os.environ["QIANFAN_ACCESS_KEY"] = "231c3fdd197b472fb59363cec29005b8"
os.environ["QIANFAN_SECRET_KEY"] = "345aa1c80714488aa1aab99b27d5c2fd"


logger = logging.getLogger('evaluation_logger')

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
resource_folder = os.path.join(root, 'resource', 'evaluation')
model_info_json = os.path.join(root, 'resource', 'model_info.json')


disease_screening_model_path = os.path.join(
    resource_folder,
    'model_mimic_20250211103028_DISCARD_OL.pth'
    # 'model_srrsh_20250211110434_DISCARD_OL.pth'
    # 基于重症数据训练的srrsh model
    # 'model_srrsh-hospitalization-severe_20250206010424.pth'
)
planner_file_path = os.path.join(
    resource_folder,
    # 'model_mimic_iv_icd_3_embedding_type_2_eng_10_256000_20241226050252_policy.pth',
    'model_mimic_eng_10_256000_20250212094250_policy.pth'
    # 'model_srrsh-hospitalization_icd_4_embedding_type_2_chn_10_256000_20241216070926_policy.pth'
)


icd_skip_path = os.path.join(resource_folder, 'icd_skip.csv')
symptom_file_path = os.path.join(resource_folder, 'symptom_{}_20240801.csv')
question_file_path = os.path.join(resource_folder, 'symptom_20240801_question_list.csv')
differential_diagnosis_procedure_file_path = os.path.join(resource_folder, 'fused_diagnosis_procedure.json')
chn_icd_mapping_path = os.path.join(root, 'resource', 'icd_list_debug.csv')
eng_icd_mapping_file = os.path.join(root, 'resource', 'd_icd_diagnoses.csv')
bge_model_file_path = '/home/sunzhoujian/llm_cache/baai/bge-m3'


call_llm_retry_time = 5
language = 'eng'
embedding_size = 1024
mask_history_action_flag = 1
diagnosis_internal_forward_flag = 1
maximum_structurize_items = 30
ecdai_planner_device = 'cpu'
embedding_device = 'cpu'
embedding_type = 2
max_new_token_llm = 1024
ecdai_disease_filter_flag = 0

# code missing strategy用于处理初筛列入高风险疾病在鉴别诊断阶段没有办法匹配的情况
# 有两个合法值，skip代表直接忽略，假装这些疾病完全不存在。end_hit指代如果出现了直接终止对话
code_missing_strategy = 'skip'

# 注意，此处的所有config参数，除了call_llm_retry_time是三个llm都需要的
# 其余只有ecdai需要指定参数，llm和react本质上都没有需要指定的参数（其实是我把参数全部写成函数调用的形参，在simulation时注入了）
# ecdai这里有些参数因为是设计导致必须设定的，实在不适合写成形参（会导致simulation时接口不统一，因此需要单独设置）
parser = argparse.ArgumentParser(description="doctor_parser")
parser.add_argument('--call_llm_retry_time', help='',
                    default=call_llm_retry_time, type=int)
parser.add_argument('--ecdai_embedding_size', help='',
                    default=embedding_size, type=int)
parser.add_argument('--ecdai_mask_history_action_flag', help='',
                    default=mask_history_action_flag, type=int)
parser.add_argument('--ecdai_diagnosis_internal_forward_flag', help='',
                    default=diagnosis_internal_forward_flag, type=int)
parser.add_argument('--ecdai_maximum_structurize_items', help='',
                    default=maximum_structurize_items, type=int)
parser.add_argument('--max_new_token_llm', help='',
                    default=max_new_token_llm, type=int)
parser.add_argument('--ecdai_code_missing_strategy', help='',
                    default=code_missing_strategy, type=str)
parser.add_argument('--ecdai_planner_device', help='',
                    default=ecdai_planner_device, type=str)
parser.add_argument('--ecdai_embedding_device', help='',
                    default=embedding_device, type=str)
parser.add_argument('--ecdai_embedding_type', help='',
                    default=embedding_type, type=int)
parser.add_argument('--ecdai_language', help='',
                    default=language, type=str)
parser.add_argument('--ecdai_disease_filter_flag', help='',
                    default=ecdai_disease_filter_flag, type=int)
parser.add_argument('--planner_file_path', help='',
                    default=planner_file_path, type=str)
parser.add_argument('--disease_screening_model_path', help='',
                    default=disease_screening_model_path, type=str)
args = vars(parser.parse_args())


args_list = []
for key in args:
    args_list.append([key, args[key]])
args_list = sorted(args_list, key=lambda x: x[0])
for item in args_list:
    logger.info('{}: {}'.format(item[0], item[1]))





