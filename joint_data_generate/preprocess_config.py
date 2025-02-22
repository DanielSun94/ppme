import os


fused_folder = os.path.abspath('/mnt/disk_2/sunzhoujian/ecdai_resource/preprocessed_data')
original_extraction = os.path.join(fused_folder, 'original_extraction')
full_diagnosis_info_srrsh_dataset = os.path.join(fused_folder, 'full_patient_info_cache', 'full_diagnosis_info.csv')

resource_folder = os.path.abspath('../resource')
final_joint_data_folder = os.path.join(resource_folder, 'final_joint_data')
renmin_mapping_file = os.path.join(resource_folder, 'diagnosis_tree', 'data_cache.json')
standard_mapping_file = os.path.join(resource_folder, 'icd_list.csv')
label_mapping_save_path = os.path.join(resource_folder, 'joint_dataset_diagnosis_mapping.csv')
disease_knowledge_path = os.path.join(resource_folder, 'disease_knowledge.csv')
diagnosis_tree_folder = os.path.join(resource_folder, 'diagnosis_tree')
langtong_folder = os.path.join(resource_folder, 'langtong_preprocessed')
langtong_ssrsh_path = os.path.join(langtong_folder, 'shaoyifu', 'langtong_srrsh_final.csv')
langtong_shengzhou_path = os.path.join(langtong_folder, 'shengzhou', 'shengzhou_final.csv')
langtong_wenfuyi_path = os.path.join(langtong_folder, 'wenfuyi', 'wenfuyi.csv')
langtong_guizhou_path = os.path.join(langtong_folder, 'hospitalization', 'guizhou_final.csv')
langtong_xiamen_path = os.path.join(langtong_folder, 'hospitalization', 'xiamen_no5_final.csv')
langtong_xiangya_path = os.path.join(langtong_folder, 'hospitalization', 'xiangya_final.csv')
mimic_iii_path = os.path.join(resource_folder, 'mimic_iii', 'final_mimic_iii.csv')
mimic_iv_path = os.path.join(resource_folder, 'mimic_iv', 'final_mimic_iv.csv')

joint_save_path = os.path.join(resource_folder, 'joint_dataset.csv')
