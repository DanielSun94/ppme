import json
import os
from util import recall_ci_eval_dict, parse_disease_idx
from evaluation_logger import logger
from evaluation_config import evaluation_result_cache_folder, external_evaluation_result_cache_folder


def sample_filter(top_n, model, turn_num, valid_type, icd_type, filter_failed, *filter_key_list):
    if valid_type == 'internal':
        target_folder = evaluation_result_cache_folder
    else:
        assert valid_type == 'external'
        target_folder = external_evaluation_result_cache_folder
    folder = os.path.join(target_folder, model)
    file_list = os.listdir(folder)
    reserved_list = []
    success_count, failed_count = 0, 0
    candidate_failed = 0
    for file_name in file_list:
        reserve_flag = True
        file_path = os.path.join(folder, file_name)
        for key in filter_key_list:
            if key not in file_name:
                reserve_flag = False
        if file_name[0:len(turn_num)] != turn_num:
            reserve_flag = False

        if reserve_flag:
            if icd_type == 'BeforeO':
                data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
                icd_first = data['data']['oracle_diagnosis'][0]
                if icd_first.upper() >= 'O':
                    reserve_flag = False
                    failed_count += 1
                else:
                    reserve_flag = True
                    success_count += 1
            else:
                assert icd_type == 'ALL'

        if filter_failed and reserve_flag:
            data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
            last_utterance = data['dialogue'][-2]['full_response']
            start_index = last_utterance.find('<AFFILIATED-INFO>') + len("<AFFILIATED-INFO>")
            end_index = last_utterance.find('</AFFILIATED-INFO>')
            candidate = json.loads(last_utterance[start_index:end_index])['candidate_disease_list']
            if len(candidate) == 0:
                reserve_flag = False
                candidate_failed += 1
        if reserve_flag:
            reserved_list.append(file_path)

    if top_n > 0:
        reserved_list = reserved_list[:top_n]

    logger.info(f'icd_type filter, success: {success_count}, failed: {failed_count}, '
                f'candidate_failed: {candidate_failed}')
    info_list = []
    for file_path in reserved_list:
        time_stamp = os.path.getctime(file_path)
        info_list.append((file_path, time_stamp))
    info_list = sorted(info_list, key=lambda x: x[1])
    reserved_list = [item[0] for item in info_list]
    return reserved_list


def screen_eval(reserved_list, *eval_criteria):
    rank_result_list_dict = dict()
    for idx, file_path in enumerate(reserved_list):
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        rank_dict = data['rank']
        for key in rank_dict.keys():
            if key not in eval_criteria:
                continue
            rank = rank_dict[key]
            if key not in rank_result_list_dict:
                rank_result_list_dict[key] = []
            rank_result_list_dict[key].append(rank)

    key_list = sorted(rank_result_list_dict.keys())
    recall_ci_eval_dict(key_list, rank_result_list_dict)


def diagnosis_eval(reserved_list, top_n, max_size):
    failed_count, success_count, wrong_count = 0, 0, 0
    for idx, file_path in enumerate(reserved_list):
        if max_size > 0 and idx >= max_size:
            break

        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        true_icd = data['data']['icd_code'].replace('.', '').lower()[:4]
        assert len(true_icd) > 0
        last_doctor_utterance = data['dialogue'][-2]['full_response']
        start_idx = last_doctor_utterance.find('<AFFILIATED-INFO>') + len('<AFFILIATED-INFO>')
        end_idx = last_doctor_utterance.find('</AFFILIATED-INFO>')
        candidate_list = json.loads(last_doctor_utterance[start_idx:end_idx])['candidate_disease_list']

        hit_flag = False
        for i in range(top_n):
            icd_code = candidate_list[i][1]
            confirm_flag = candidate_list[i][4]
            if confirm_flag == 1 and icd_code == true_icd:
                success_count += 1
                hit_flag = True
            elif confirm_flag == 1 and icd_code != true_icd:
                wrong_count += 1
                hit_flag = True
        if not hit_flag:
            failed_count += 1

    print(f'data len: {len(reserved_list)}, success: {success_count}, wrong: {wrong_count}, failure: {failed_count}')



def main():
    # visit_type = 'outpatient'  # hospitalization outpatient
    # assert key in {"SCREEN", 'DIAGNOSIS', "ALL"}
    valid_type = 'internal'
    complete_screen_file_num = 0
    icd_type = 'BeforeO'
    key = 'SCREEN'
    filter_failed = True
    filter_list = [
        # ['srrsh', 'ecdai', 'local_qwen_2__5_72b_int4_2', '0', '3'],  # larger than 2500
        # ['mimic', 'ecdai', 'local_qwen_2__5_72b_int4', '0', '3'],  # larger than 2500
        # ['srrsh', 'llm', 'deepseek_r1_70b', '10', 'name'],   # larger than 2500, 2517
        ['mimic', 'llm', 'deepseek_r1_70b', '10', '3'],   # larger than 2500, 2765
        ['mimic', 'llm', 'deepseek_r1_remote', '20', '3'],
        # ['srrsh', 'llm', 'local_qwen_2__5_72b_int4_2', '10', 'name'],  # larger than 2500, 3988
        # ['mimic', 'llm', 'local_qwen_2__5_72b_int4_2', '10', '3'],  # larger than 1500, 2342
        # ['srrsh', 'llm', 'local_qwen_2__5_72b', '20', 'name'],  # larger than 2500, 3758
        # ['mimic', 'llm', 'local_qwen_2__5_72b', '20', '3'],  # larger than 1500, 1767 (意义不大，因为只有不到10%)
        # ['srrsh', 'llm', 'llama_3_3_70b', '10', 'name'],  # larger than 2500, 3140
        # ['mimic', 'llm', 'llama_3_3_70b', '10', '3'],  # larger than 2500, 2898
        # ['srrsh', 'llm', 'llama_3_3_70b', '20', 'name'],  # larger than 2500, 3140
        # ['mimic', 'llm', 'llama_3_3_70b', '20', '3'],  # larger than 2500, 2898
        # ['srrsh', 'llm', 'ultra_medical_llm', '10', 'name'],  # larger than 2500, 4949
        # ['mimic', 'llm', 'ultra_medical_llm', '10', '3'],   # larger than 2500, 2785
        # ['srrsh', 'llm', 'openbiollm', '10', 'name'],  # larger than 2500, 2829
        # ['mimic', 'llm', 'openbiollm', '10', '3'],  # larger than 2500, 2743
        # ['srrsh', 'llm', 'huatuogpt_o1_72b', '10', 'name'],  # larger than 1500, 2500
        # ['mimic', 'llm', 'huatuogpt_o1_72b', '10', '3'],
        # ['srrsh', 'llm', 'huatuogpt_o1_72b', '20', 'name'],
        # ['mimic', 'llm', 'huatuogpt_o1_72b', '20', '3'],
    ]

    for item in filter_list:
        data_source, llm_type, model, turn_num, eval_criteria = item
        reserved_list = sample_filter(2500, model, turn_num, valid_type, icd_type, filter_failed,
                                      key, llm_type, data_source)
        complete_screen_file_num += len(reserved_list)
        logger.info(f'screen eval, model: {model}, llm type: {llm_type}, data_source: {data_source}, '
                    f'turn_num: {turn_num}')
        # screen_eval(reserved_list, eval_criteria)
        screen_eval(reserved_list, eval_criteria)

    logger.info(f'complete file count: {complete_screen_file_num}')
    # if key != 'SCREEN':
    #     diagnosis_eval(reserved_list, top_n_differential_diagnosis_disease, max_size)
    # logger.info(f'start eval, llm type: {llm_type}, data_source: {data_source}, visit_type: {visit_type}, key: {key}')


if __name__ == '__main__':
    main()
