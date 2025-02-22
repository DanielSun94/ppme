import json
import pickle
import os.path
import traceback
from evaluation_logger import logger
from read_data import load_data, filter_data
from util import moderator_judge_request, non_streaming_call_llm
from evaluation_config import (args, full_diagnosis_file, symptom_num_path)
from evaluation_config import evaluation_result_cache_folder, external_evaluation_result_cache_folder
from patient_serve.patient_simulator import patient_behavior_wrapper
from environment import Environment


language = args['language']
phase_ = args['phase']
screening_maximum_question_ = args['screening_maximum_question']
top_n_differential_diagnosis_disease_ = args['top_n_differential_diagnosis_disease']
diagnosis_mode_ = args['diagnosis_mode']
doctor_llm_name = args['doctor_llm_name']
patient_llm_name = args['patient_llm_name']
doctor_type = args['doctor_type']
differential_diagnosis_icd = args['differential_diagnosis_icd']
start_index = args['start_index']
end_index = args['end_index']
filter_criteria = args['filter_criteria']
icd_type = args['icd_type']
logger.info('start simulation')

max_size = 2500

if args['validation_type'] == 'internal':
    target_folder = evaluation_result_cache_folder
else:
    assert args['validation_type'] == 'external'
    target_folder = external_evaluation_result_cache_folder


def current_parsed_info_check(prefix, filter_criteria_, folder):
    hit_count = 0
    file_list = os.listdir(folder)
    for file_name in file_list:
        if prefix + f'_{filter_criteria_}' in file_name:
            hit_count += 1
    return hit_count


def main():
    def call_llm(prompt, llm_name):
        return non_streaming_call_llm(llm_name, prompt)

    assert doctor_type == 'llm' or doctor_type == 'react' or 'ecdai' in doctor_type
    if doctor_type == 'llm':
        from doctor_serve.llm_doctor_simulator import doctor_behavior_wrapper as llm_behavior
        doctor_behavior = llm_behavior
    elif doctor_type == 'react':
        from doctor_serve.react_doctor_simulator import doctor_behavior_wrapper as react_behavior
        doctor_behavior = react_behavior
    else:
        assert 'ecdai' in doctor_type
        from doctor_serve.ecdai_doctor_simulator import doctor_behavior_wrapper as ecdai_behavior
        doctor_behavior = ecdai_behavior

    # 默认使用3位有效数字，这里其实不影响，因为只是选择数据集
    logger.info('start loading data')

    if filter_criteria == 'srrsh-hospitalization-severe':
        key = '_'.join(["3", 'srrsh', icd_type])
    else:
        key = '_'.join(["3", filter_criteria, icd_type])
    symptom_num_dict = pickle.load(open(symptom_num_path.format(key), 'rb'))
    data_dict = load_data(full_diagnosis_file)
    data_list = filter_data(data_dict, symptom_num_dict, filter_criteria, start_index, end_index)
    logger.info(f'simulation data filtered, data length: {len(data_list)}')
    for item in data_list:
        key, selected_data, _ = item
        prefix = f'{screening_maximum_question_}_{doctor_type}_{phase_}'
        save_file_name = f"{prefix}_{key}.json"
        save_folder = os.path.join(target_folder, doctor_llm_name)
        os.makedirs(save_folder, exist_ok=True)
        save_path = str(os.path.join(save_folder, save_file_name))

        hit_count = current_parsed_info_check(prefix, filter_criteria, save_folder)
        if hit_count > max_size:
            logger.info('exceed max size, return')
            break

        if os.path.exists(save_path):
            logger.info('Parsed already: {}'.format(save_path))
            continue
        selected_data = data_dict[key]
        data_id = key
        environment = Environment(
            patient_behavior_func=patient_behavior_wrapper,
            doctor_behavior_func=doctor_behavior,
            patient_info_context=selected_data,
            phase=phase_,
            doctor_llm_name=doctor_llm_name,
            patient_llm_name=patient_llm_name,
            screening_maximum_question=screening_maximum_question_,
            top_n_differential_diagnosis_disease=top_n_differential_diagnosis_disease_,
            language=language,
            data_id=data_id,
            diagnosis_mode=diagnosis_mode_,
            diagnosis_target=differential_diagnosis_icd
        )

        try:
            end_flag = False
            while not end_flag:
                end_flag = environment.step()
            history = environment.dialogue_history
            save_data = {
                'dialogue': history,
                'data': selected_data,
                'llm': f'doctor: {doctor_llm_name}, patient: {patient_llm_name}',
                'rank': moderator_judge_request(
                    history,
                    selected_data,
                    call_llm,
                    doctor_type,
                    doctor_llm_name
                )
            }
            json.dump(save_data, open(save_path, 'w', encoding='utf-8-sig'))
            logger.info(f'key: {key} success')
        except:
            logger.info(u"Error Trance {}".format(traceback.format_exc()))
            logger.info(f'key: {key} simulated dialogue failed')


if __name__ == "__main__":
    main()
