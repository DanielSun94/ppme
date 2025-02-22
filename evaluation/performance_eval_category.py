from evaluation_logger import logger
from performance_eval import sample_filter
from util import ci_calculate
import json
import numpy as np
from util import parse_disease_idx

category_list = [
    [0, '传染病和寄生虫病 A00-B99', 'A', 00, 'B', 99],  # Certain infectious and parasitic diseases
    [1, '肿瘤 C00-D49', 'C', 00, 'D', 49],  # Neoplasms
    [2, '血液病 D50-D89', 'D', 50, 'D', 89],  # Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism
    [3, '内分泌疾病 E00-E99', 'E', 00, 'E', 99],  # Endocrine, nutritional and metabolic diseases
    [4, '精神和行为障碍 F00-F99', 'F', 00, 'F', 99],  # Mental and behavioural disorders
    [5, '神经系统疾病 G00-G99', 'G', 00, 'G', 99],  # Diseases of the nervous system
    [6, '眼科疾病 H00-H59', 'H', 00, 'H', 59],  # Diseases of the eye and adnexa
    [7, '耳部及乳突疾病 H60-H99', 'H', 60, 'H', 99],  # Diseases of the ear and mastoid process
    [8, '循环系统疾病 I00-I99', 'I', 00, 'I', 99],  # Diseases of the circulatory system
    [9, '呼吸系统疾病 J00-J99', 'J', 00, 'J', 99],  # Diseases of the respiratory system
    [10, '消化系统疾病 K00-K99', 'K', 00, 'K', 99],  # Diseases of the digestive system
    [11, '皮肤和皮下组织疾病 L00-L99', 'L', 00, 'L', 99],  # Diseases of the skin and subcutaneous tissue
    [12, '肌肉骨骼和结缔组织病 M00-M99', 'M', 00, 'M', 99],  # Diseases of the musculoskeletal system and connective tissue
    [13, '泌尿生殖系统疾病 N00-N99', 'N', 00, 'N', 99],  # Diseases of the genitourinary system
]
#
# # category_list = [
# #     [0, '真实疾病 A00-O99', 'A', 00, 'Q', 99],  # Certain infectious and parasitic diseases
# # ]
#
# # category_list = [
# #     [0, '所有疾病 A00-Z99', 'A', 00, 'Z', 99],  # Certain infectious and parasitic diseases
# # ]


def screen_eval(reserved_list, *eval_criteria):

    rank_result_list_dict = dict()
    for idx, file_path in enumerate(reserved_list):
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        rank_dict = data['rank']
        diagnosis = data['data']['oracle_diagnosis']
        first_diagnosis = diagnosis.split('$$$$$')[0]
        idx = parse_disease_idx(first_diagnosis, category_list)
        for key in rank_dict.keys():
            if key not in eval_criteria:
                continue

            rank = rank_dict[key]
            if key not in rank_result_list_dict:
                rank_result_list_dict[key] = {}
            if idx not in rank_result_list_dict[key]:
                rank_result_list_dict[key][idx] = []
            rank_result_list_dict[key][idx].append(rank)

    key_list = sorted(rank_result_list_dict.keys())
    for key in key_list:
        for idx in range(len(category_list)):
            if idx in rank_result_list_dict[key] and len(rank_result_list_dict[key][idx]) > 0:
                logger.info(f'key_name: {key}, length: {len(rank_result_list_dict[key][idx])}, '
                            f'idx: {idx}, category: {category_list[idx][1]}')
                rank_list = np.array(rank_result_list_dict[key][idx])
                top_1 = np.sum(rank_list < 2) / len(rank_list)
                top_2 = np.sum(rank_list < 3) / len(rank_list)
                top_3 = np.sum(rank_list < 4) / len(rank_list)
                top_4 = np.sum(rank_list < 5) / len(rank_list)
                top_5 = np.sum(rank_list < 6) / len(rank_list)

                top_1_ci = ci_calculate(rank_list < 2)
                top_2_ci = ci_calculate(rank_list < 3)
                top_3_ci = ci_calculate(rank_list < 4)
                top_4_ci = ci_calculate(rank_list < 5)
                top_5_ci = ci_calculate(rank_list < 6)
            else:
                top_1, top_2, top_3, top_4, top_5 = 0, 0, 0, 0, 0
                top_1_ci, top_2_ci, top_3_ci, top_4_ci, top_5_ci = 0, 0, 0, 0, 0
            logger.info('top 1: {:.4f}, 95% CI: {:.4f}'.format(top_1, top_1_ci))
            logger.info('top 2: {:.4f}, 95% CI: {:.4f}'.format(top_2, top_2_ci))
            logger.info('top 3: {:.4f}, 95% CI: {:.4f}'.format(top_3, top_3_ci))
            logger.info('top 4: {:.4f}, 95% CI: {:.4f}'.format(top_4, top_4_ci))
            logger.info('top 5: {:.4f}, 95% CI: {:.4f}'.format(top_5, top_5_ci))
            logger.info('')


def main():
    # visit_type = 'outpatient'  # hospitalization outpatient
    # assert key in {"SCREEN", 'DIAGNOSIS', "ALL"}
    complete_screen_file_num = 0
    valid_type = 'internal'
    icd_type = 'BeforeO'
    key = 'SCREEN'
    filter_list = [
        # ['srrsh', 'llm', 'deepseek_r1_70b', '10', 'name'],
        # ['mimic', 'llm', 'deepseek_r1_70b', '10', '3'],
        # ['mimic', 'llm', 'deepseek_r1_remote', '20', '3'],
        # ['srrsh', 'ecdai', 'local_qwen_2__5_72b_int4_2', '0', '3'],  #
        ['mimic', 'ecdai', 'local_qwen_2__5_72b_int4', '0', '3'],  #
        # ['srrsh', 'llm', 'local_qwen_2__5_72b_int4_2', '10', 'name'],
        # ['mimic', 'llm', 'local_qwen_2__5_72b_int4_2', '10', '3'],
        # ['srrsh', 'llm', 'local_qwen_2__5_72b', '20', 'name'],
        # ['mimic', 'llm', 'local_qwen_2__5_72b', '20', '3'],
        # ['srrsh', 'llm', 'llama_3_3_70b', '10', 'name'],
        ['mimic', 'llm', 'llama_3_3_70b', '10', '3'],
        # ['srrsh', 'llm', 'ultra_medical_llm', '10', 'name'],
        # ['mimic', 'llm', 'ultra_medical_llm', '10', '3'],
        # ['srrsh', 'llm', 'openbiollm', '10', 'name'],
        # ['mimic', 'llm', 'openbiollm', '10', '3'],
    ]


    for item in filter_list:
        data_source, llm_type, model, turn_num, eval_criteria = item
        reserved_list = sample_filter(5000, model, turn_num, valid_type, icd_type,
                                      key, llm_type, data_source)
        complete_screen_file_num += len(reserved_list)
        logger.info(f'screen eval, llm type: {llm_type}, data_source: {data_source}')
        screen_eval(reserved_list, eval_criteria)

    logger.info(f'complete file count: {complete_screen_file_num}')
    # if key != 'SCREEN':
    #     diagnosis_eval(reserved_list, top_n_differential_diagnosis_disease, max_size)
    # logger.info(f'start eval, llm type: {llm_type}, data_source: {data_source}, visit_type: {visit_type}, key: {key}')


if __name__ == '__main__':
    main()
