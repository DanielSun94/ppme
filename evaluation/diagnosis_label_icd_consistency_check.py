import os
import json
import pickle
import argparse
import traceback
from evaluation_logger import logger
from util import (non_streaming_call_llm, read_english_icd, read_chinese_icd)
from read_data import load_data, filter_data
from evaluation_config import symptom_num_path, full_diagnosis_file, evaluation_cross_icd_parsing_accuracy_file

call_llm_retry_time = 5
filter_criteria_ = 'srrsh-hospitalization'
start_index_ = 0
end_index_ = 5000
rank_llm_name_ = 'deepseek_v3_remote'
language_ = 'chn'

max_len = 3072
parser = argparse.ArgumentParser(description="evaluation_parser")
parser.add_argument('--language', help='', default=language_, type=str)
parser.add_argument('--llm_name', default=rank_llm_name_, type=str)
parser.add_argument('--start_index', default=0, type=int)
parser.add_argument('--end_index', default=2000, type=int)
parser.add_argument('--filter_criteria',  default=filter_criteria_, type=str)
args_list = []
args = vars(parser.parse_args())
for arg in args:
    args_list.append([arg, args[arg]])
args_list = sorted(args_list, key=lambda x: x[0])
for arg in args_list:
    logger.info('{}: {}'.format(arg[0], arg[1]))


prompt_dict = {
    'chn': "请假设你是一名医生，你需要判断下面的两种疾病的关系。1.{}。 2: {}."
           "如果疾病1和疾病2是同一种疾病，或疾病2可以宽泛地视为疾病1的子型（或者是一部分），返回YES。"
           "如果疾病1和疾病2明显不是同一种疾病，也没有从属关系，返回NO。\n"
           "注意，你只需输出YES或者NO即可，无需返回其它任何信息",
    'eng': "Please assume you are a doctor and need to determine the relationship between the following two diseases.\n"
           "1.{}, 2. {}. If Disease 1 and Disease 2 are the same disease, or if Disease 2 is a more specific subtype "
           "of Disease 1, return YES; otherwise, return NO. Note: You should only output YES or NO, "
           "without returning any other information."
}


eng_icd_mapping_dict = read_english_icd()
chn_icd_mapping_dict = read_chinese_icd()

def main():
    def call_llm(prompt_, llm_name):
        return non_streaming_call_llm(llm_name, prompt_)

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

    if args['language'] == 'chn':
        icd_mapping = chn_icd_mapping_dict
    else:
        assert args['language'] == 'eng'
        icd_mapping = eng_icd_mapping_dict

    if os.path.exists(evaluation_cross_icd_parsing_accuracy_file):
        data_dict = json.load(open(evaluation_cross_icd_parsing_accuracy_file, 'r', encoding='utf-8-sig'))
        eval_performance(data_dict)
    else:
        data_dict = dict()

    for i, item in enumerate(data_list):
        key = item[0]
        if key in data_dict:
            logger.info(f'{key} parsing success')
            continue
        oracle_diagnosis_str = item[1]['table_diagnosis'].strip().lower()
        first_diagnosis_str = oracle_diagnosis_str.split('$$$$$')[0]
        oracle_diagnosis_icd = item[1]['oracle_diagnosis'].strip().lower()
        first_diagnosis_icd = oracle_diagnosis_icd.split('$$$$$')[0][:3]

        if len(first_diagnosis_str) == 0 or len(first_diagnosis_icd) == 0 or first_diagnosis_icd not in icd_mapping:
            continue
        oracle_diagnosis_str = icd_mapping[first_diagnosis_icd]

        failure_time = 0
        success_flag = False
        result = 0
        while not success_flag:
            try:
                prompt = prompt_dict[args['language']].format(oracle_diagnosis_str, first_diagnosis_str)
                response = call_llm(prompt, args['llm_name'])
                response = response.upper()
                if 'YES' in response and "NO" in response:
                    raise ValueError('TWO HIT ERROR')
                elif 'YES' not in response and "NO" not in response:
                    raise ValueError('None HIT ERROR')
                elif 'YES' in response:
                    result = 1
                else:
                    assert "NO" in response
                    result = 0
                logger.info(f'key: {key}, oracle diagnosis: {oracle_diagnosis_str}, '
                            f'first diagnosis: {first_diagnosis_str}, result: {result}')
                success_flag = True
            except Exception:
                failure_time += 1
                logger.info(u"Error Trance {}".format(traceback.format_exc()))
                if failure_time >= 5:
                    success_flag = True
        data_dict[item[0]] = result, oracle_diagnosis_str, first_diagnosis_str
        logger.info(f'{key} parsing success')

        if len(data_dict) > 0 and len(data_dict) % 100 == 0:
            json.dump(data_dict, open(evaluation_cross_icd_parsing_accuracy_file, 'w', encoding='utf-8-sig'))
            eval_performance(data_dict)
    json.dump(data_dict, open(evaluation_cross_icd_parsing_accuracy_file, 'w', encoding='utf-8-sig'))
    print_false_result(data_dict)


def print_false_result(data_dict):
    for key in data_dict:
        result, oracle_diagnosis_str, first_diagnosis_str = data_dict[key]
        if result == 0:
            logger.info(f'false, key: {key}, oracle_diagnosis_str: {oracle_diagnosis_str}, '
                        f'first_diagnosis_str {first_diagnosis_str}')


def eval_performance(data_dict):
    hit_count = 0
    for key in data_dict:
        hit_count += data_dict[key][0]
    logger.info(f'hit_count: {hit_count}, full_size: {len(data_dict)}, hit ratio {hit_count / len(data_dict)}')


if __name__ == "__main__":
    main()
