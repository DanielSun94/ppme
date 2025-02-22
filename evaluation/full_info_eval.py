import os
import json
import pickle
import argparse
import traceback
from evaluation_logger import logger
from util import (non_streaming_call_llm, parse_result_disease_think, parse_rank, rerank_prompt_dict,
                  eng_icd_mapping_dict, eval_performance)
from read_data import load_data, filter_data
from evaluation_config import symptom_num_path, full_diagnosis_file, full_info_eval_folder

call_llm_retry_time = 5
filter_criteria_ = 'srrsh'
start_index_ = 0
end_index_ = 2500
# openbiollm llama_3_3_70b ultra_medical_llm local_qwen_2__5_72b_int4_2 deepseek_r1_70b
# qwen_2__5_72b_it_deepinfra deepseek_r1_remote
doctor_llm_name_ = 'deepseek_r1_remote'
rank_llm_name_ = 'deepseek_v3_remote'
language_ = 'chn'

max_len = 3072
parser = argparse.ArgumentParser(description="evaluation_parser")
parser.add_argument('--language', help='', default=language_, type=str)
parser.add_argument('--doctor_llm_name', help='', default=doctor_llm_name_, type=str)
parser.add_argument('--rank_llm_name', default=rank_llm_name_, type=str)
parser.add_argument('--start_index', default=start_index_, type=int)
parser.add_argument('--end_index', default=end_index_, type=int)
parser.add_argument('--filter_criteria',  default=filter_criteria_, type=str)
args_list = []
args = vars(parser.parse_args())
for arg in args:
    args_list.append([arg, args[arg]])
args_list = sorted(args_list, key=lambda x: x[0])
for arg in args_list:
    logger.info('{}: {}'.format(arg[0], arg[1]))


prompt_dict = {
    'chn': "请假设你是一名医生，你正在和一名病人进行医疗咨询。你需要根据以下电子病历，推测他有什么疾病。"
           "你需要列出造成病人本次入院的最可能的5种疾病，注意，越高风险的疾病应当排在越前面，"
           "你不能输出重复的疾病，疾病的粒度应当尽可能细致。\n"
           "输出格式应当遵循如下格式（#ANSWER START#是开始标记，用于定位真正的回答开始位置，请严格输出#ANSWER START#）"
           "#ANSWER START#\n#1#: 疾病名称\n#2#: 疾病名称\n#3#: 疾病名称\n#4#: 疾病名称\n#5#: 疾病名称\n"
           "以下是电子病历的内容\n",
    'eng': "Please assume you are a doctor conducting a medical consultation with a patient. "
           "Based on the EMR with the patient, you need to infer their possible diseases. "
           "You are required to list the five most "
           "likely diseases causing the patient's current hospitalization. Note that higher-risk diseases should "
           "be listed first. You must not output duplicate diseases, and the granularity of the diseases should "
           "be as detailed as possible.\nYour output format should follow (#ANSWER START# is the start token of the "
           "answer, it is used to navigate the real start of the answer, please strictly output #ANSWER START#):\n"
           "#ANSWER START#\n#1#: Disease Name\n#2#: Disease Name\n#3#: Disease Name\n#4#: Disease Name\n"
           "#5#: Disease Name\n"
           "The EMR content is:\n"
}


def main():
    def call_llm(prompt_, llm_name):
        return non_streaming_call_llm(llm_name, prompt_)

    save_folder = os.path.join(full_info_eval_folder, args['doctor_llm_name'], args['filter_criteria'])
    os.makedirs(save_folder, exist_ok=True)
    eval_performance(save_folder, max_size=2500)
    # 默认使用3位有效数字，这里其实不影响，因为只是选择数据集
    key = '_'.join(["3", args['filter_criteria'], 'ALL'])
    logger.info('start loading data')
    symptom_num_dict = pickle.load(open(symptom_num_path.format(key), 'rb'))
    data_dict = load_data(full_diagnosis_file)
    data_list = filter_data(data_dict, symptom_num_dict,
                            args['filter_criteria'],
                            args['start_index'],
                            args['end_index'])
    logger.info(f'simulation data filtered, data length: {len(data_list)}')

    parse_index = 0
    for item in data_list:
        key, selected_data, _ = item
        save_file_name = f"{key}.json"
        save_path = str(os.path.join(save_folder, save_file_name))

        if os.path.exists(save_path):
            logger.info('Parsed already: {}'.format(save_path))
            continue
        selected_data = data_dict[key]
        failed_time = 0
        parse_index += 1
        success_flag = False
        while not success_flag:
            try:
                if 'outpatient' in key:
                    emr = selected_data['outpatient_record']
                else:
                    emr = selected_data['admission_record'][:max_len]
                full_prompt = prompt_dict[args['language']] + emr
                response = call_llm(full_prompt, args['doctor_llm_name'])
                result = parse_result_disease_think(response)
                candidate_str = ''
                for idx, candidate in enumerate(result):
                    candidate_str += f'#{idx + 1}#: {candidate}\n'

                if 'srrsh' in key:
                    oracle_diagnosis_str = selected_data['table_diagnosis'].strip().lower()
                    diagnosis_str = oracle_diagnosis_str.split('$$$$$')[0]
                else:
                    assert 'mimic' in key
                    icd_dict = eng_icd_mapping_dict
                    oracle_diagnosis = selected_data['oracle_diagnosis'].strip().lower()
                    first_diagnosis = oracle_diagnosis.split('$$$$$')[0]
                    if first_diagnosis[:3] not in icd_dict or first_diagnosis[:4] not in icd_dict:
                        diagnosis_str = 'NO DISEASE'
                    else:
                        diagnosis_str = icd_dict[first_diagnosis[:3]]
                prompt = rerank_prompt_dict[args['language']].format(candidate_str, diagnosis_str, diagnosis_str)
                name_rank = parse_rank(call_llm, prompt, args['rank_llm_name'])
                if 'srrsh' in key:
                    rank_dict = {'name': name_rank}
                else:
                    assert 'mimic' in key
                    rank_dict = {'3': name_rank}

                data_to_save = {
                    "rank_dict": rank_dict,
                    'diagnosis': result,
                    'data': selected_data
                }
                json.dump(data_to_save, open(save_path, 'w', encoding='utf-8-sig'))
                logger.info(f'success, key: {key}')
                success_flag = True

                if parse_index % 10 == 0:
                    eval_performance(save_folder)
            except:
                failed_time += 1
                logger.info(u"Error Trace {}".format(traceback.format_exc()))
                if failed_time > call_llm_retry_time:
                    logger.info(f'client: {key}, parse previous interact failed')
                    break


if __name__ == "__main__":
    main()
