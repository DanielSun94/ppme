import os


llm_cache_folder = '/mnt/disk_1/llm_cache'
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
resource_folder = os.path.join(parent_dir, 'resource')
final_data_folder = os.path.join(resource_folder, 'final_joint_data')
disease_screen_folder = os.path.join(resource_folder, 'disease_screen')
# data_file_path = os.path.join(resource_folder, 'final_joint_data', 'use_code_map_False_digit_4_fraction_0.95.csv')
data_file_path = os.path.join(resource_folder, 'final_joint_data', 'use_code_map_False_digit_4_fraction_1.0.csv')
data_file_code_true_path = os.path.join(resource_folder, 'final_joint_data',
                                        'use_code_map_True_digit_4_fraction_0.95.csv')
structurize_symptom_cache_template = os.path.join(disease_screen_folder, 'structurize_symptom_cache_{}.pkl')
fixed_question_answer_folder = os.path.join(disease_screen_folder, 'fixed_question_answer')
diagnosis_cache_template = os.path.join(disease_screen_folder, 'diagnosis_cache_{}_{}_{}_{}_{}_{}.pkl')
symptom_diagnosis_cache = os.path.join(disease_screen_folder, 'symptom_diagnosis_cache.pkl')
symptom_diagnosis_sample = os.path.join(disease_screen_folder, 'symptom_diagnosis_sample.csv')
symptom_file_path_template = os.path.join(disease_screen_folder, 'symptom_{}_20240801.csv')
question_file_path = os.path.join(disease_screen_folder, 'symptom_20240801_question_list.csv')
screen_model_save_folder = os.path.join(disease_screen_folder, 'screen_model')
history_text_embedding_folder = os.path.join(disease_screen_folder, 'history_text_embedding')
symptom_num_path = os.path.join(disease_screen_folder, 'symptom_num_{}.pkl')
symptom_folder_path = os.path.join(disease_screen_folder, 'structured_symptom')
icu_patient_idx_file = os.path.join(resource_folder, 'icu_patient_idx.json')
os.makedirs(screen_model_save_folder, exist_ok=True)



symptom_info_dict = {
    'all':
        [
            [
                os.path.join(symptom_folder_path, 'Qwen2-72B-Instruct-GPTQ-Int4', 'srrsh', 'hospitalization'),
                'chn',
                'srrsh-hospitalization'
            ],
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'srrsh', 'outpatient'),
                'chn',
                'srrsh-outpatient'
            ],
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'mimic_iv', 'hospitalization'),
                'eng',
                'mimic_iv-hospitalization'
            ],
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'mimic_iii', 'hospitalization'),
                'eng',
                'mimic_iii-hospitalization'
            ]
        ],
    'mimic': [
        [
            os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'mimic_iii', 'hospitalization'),
            'eng',
            'mimic_iii-hospitalization'
        ],
        [
            os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'mimic_iv', 'hospitalization'),
            'eng',
            'mimic_iv-hospitalization'
        ]
    ],
    'mimic_iii':
        [
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'mimic_iii', 'hospitalization'),
                'eng',
                'mimic_iii-hospitalization'
            ]
        ],
    'mimic_iv':
        [
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'mimic_iv', 'hospitalization'),
                'eng',
                'mimic_iv-hospitalization'
            ]
        ],
    'srrsh':
        [
            [
                os.path.join(symptom_folder_path, 'Qwen2-72B-Instruct-GPTQ-Int4', 'srrsh', 'hospitalization'),
                'chn',
                'srrsh-hospitalization'
            ],
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'srrsh', 'outpatient'),
                'chn',
                'srrsh-outpatient'
            ],
        ],
    'srrsh-hospitalization':
        [
            [
                os.path.join(symptom_folder_path, 'Qwen2-72B-Instruct-GPTQ-Int4', 'srrsh', 'hospitalization'),
                'chn',
                'srrsh-hospitalization'
            ]
        ],
    'srrsh-outpatient':
        [
            [
                os.path.join(symptom_folder_path, 'Qwen2___5-72B-Instruct-GPTQ-Int4', 'srrsh', 'outpatient'),
                'chn',
                'srrsh-outpatient'
            ],
        ],
}