import os


source_folder = os.path.abspath('/mnt/disk_3/sunzhoujian/medical_data')
fused_folder = os.path.abspath('/mnt/disk_2/sunzhoujian/ecdai_resource/preprocessed_data')
resource_folder = os.path.abspath('../resource')
diagnosis_tree_folder = os.path.abspath('../resource/diagnosis_tree')
diagnosis_label_cache_folder = os.path.abspath('../resource/diagnosis_label_folder')
fixed_question_answer_folder = os.path.join(resource_folder, 'fixed_question_answer')
icu_patient_idx_file = os.path.join(resource_folder, 'icu_patient_idx.json')

symptom_path_template = os.path.join(resource_folder, 'symptom_list_{}.csv')
transformed_id_list_file_path = os.path.join(resource_folder, 'transformed_id_list_file.csv')
transformed_id_list_path = os.path.join(resource_folder, 'transformed_id_list.csv')
symptom_folder_template = os.path.join(resource_folder, 'transformed_symptom', '{}')
final_disease_screening_ready_data_path = os.path.join(resource_folder, 'final_disease_screening_dataset.csv')
label_mapping_save_path_template = os.path.join(resource_folder, '{}_label_mapping.csv')
renmin_mapping_file = os.path.join(resource_folder, 'diagnosis_tree', 'data_cache.json')
standard_mapping_file = os.path.join(resource_folder, 'icd_list.csv')
srrsh_diagnosis_mapping_cache_path = os.path.join(resource_folder, 'srrsh_diagnosis_mapping_cache.pkl')

emr_index_folder = os.path.join(source_folder, '20180718', 'emr_index')
diagnosis_1_folder = os.path.join(source_folder, '20180718', 'diagnosis')
diagnosis_2_folder = os.path.join(source_folder, '2017', 'ltdiag', 'out')

clinical_note_folder_1 = os.path.join(source_folder, '20180417', "EMR_SE")
clinical_note_folder_2 = os.path.join(source_folder, '20180718', "emr_se")
emr_ns_folder = os.path.join(source_folder, '20180718', "emr_data_ns")

diagnosis_tree_cache = os.path.join(diagnosis_tree_folder, 'data_cache.json')
original_extraction = os.path.join(fused_folder, 'original_extraction')
fused_clinical_note_path_template = os.path.join(original_extraction, 'fused_clinical_note_{}.csv')
fused_diagnosis_file_path_template = os.path.join(original_extraction, 'fused_diagnosis_ICD_{}.csv')
diagnosis_icd_count_pkl_path = os.path.join(original_extraction, 'diagnosis_icd_count.pkl')
fused_admission_discharge_note = os.path.join(original_extraction, 'fused_admission_discharge_note.csv')
fused_admission_discharge_history_note = os.path.join(original_extraction, 'fused_admission_discharge_history_note.csv')
fused_outpatient_admission_discharge_history_note = (
    os.path.join(original_extraction, 'fused_outpatient_admission_discharge_history_note.csv'))
discharge_data_path = (
    os.path.join(original_extraction, 'discharge_data.pkl'))
outpatient_check_file_1 = (
    os.path.join(original_extraction, 'outpatient_check_1.pkl'))
outpatient_check_file_2 = (
    os.path.join(original_extraction, 'outpatient_check_2.pkl'))
emr_index_fuse_file_template = os.path.join(original_extraction, 'emr_index_fuse_{}.csv')

full_primary_diagnosis_dataset_path = os.path.join(original_extraction, 'full_primary_diagnosis_dataset.csv')
final_fusion_path = os.path.join(original_extraction, 'srrsh_final_big_for_screen.csv')
distinct_diagnosis_path = os.path.join(original_extraction, 'distinct_diagnosis.json')

emr_index_fuse_cache = os.path.join(fused_folder, 'emr_index_fuse_cache.pkl')
diagnosis_fuse_cache_template = os.path.join(fused_folder, 'diagnosis_fuse_cache_{}_ICD_{}.pkl')
clinical_note_cache_template = os.path.join(fused_folder, 'clinical_note_{}_fuse_cache.pkl')
fused_joint_admission_file_template = os.path.join(fused_folder, 'fused_{}_diagnosis.csv')

# exam and lab test
ord_detail_folder = os.path.join(source_folder, '20180718', 'ord_detail')
ord_prescription_folder_1 = os.path.join(source_folder, '20180718', 'ord')
ord_prescription_folder_2 = os.path.join(source_folder, '2017', 'ltord')
ord_pac_record_folder = os.path.join(source_folder, '20180718', 'ord_pacs_record')
ord_rec_folder = os.path.join(source_folder, '20180718', 'ord_rec')
lt_ord_pacs_record_original_path = os.path.join(source_folder, '20180417', 'LT_ORD_PACS_RECORD.json')

lt_ord_pacs_record_path = os.path.join(fused_folder, 'lt_ord_pacs_record.csv')
ord_pacs_record_path = os.path.join(fused_folder, 'ord_pacs_record.csv')
ord_prescription_path = os.path.join(fused_folder, 'ord_prescription.csv')
ord_rec_path = os.path.join(fused_folder, 'ord_rec.csv')
ord_detail_fuse_path = os.path.join(fused_folder, '20180718_ord_detail.csv')

os.makedirs(original_extraction, exist_ok=True)
os.makedirs(diagnosis_label_cache_folder, exist_ok=True)
os.makedirs(fixed_question_answer_folder, exist_ok=True)

# full info fusion
full_info_cache_folder = os.path.join(fused_folder, 'full_patient_info_cache')
valid_patient_visit_id_file = os.path.join(full_info_cache_folder, 'valid_patient_visit_id.pkl')
exam_mapping_cache_file = os.path.join(full_info_cache_folder, 'exam_mapping_cache.pkl')
lab_test_rep_cache_file = os.path.join(full_info_cache_folder, 'lab_test_rep_cache.pkl')
reserved_mapping_exam_file = os.path.join(full_info_cache_folder, 'reserved_mapping_exam_file.pkl')
ord_prescription_mapping_file = os.path.join(full_info_cache_folder, 'ord_prescription_mapping.pkl')
valid_pat_lab_rep_code_dict_file = os.path.join(full_info_cache_folder, 'valid_lab_pat_rep_code_dict.pkl')
valid_pat_exam_rep_code_dict_file = os.path.join(full_info_cache_folder, 'valid_exam_pat_rep_code_dict.pkl')
lab_test_data_cache_file = os.path.join(full_info_cache_folder, 'lab_test_data_cache.pkl')
exam_data_cache_file = os.path.join(full_info_cache_folder, 'exam_data_cache_file.pkl')
age_sex_cache_file = os.path.join(full_info_cache_folder, 'age_sex_cache.pkl')
differential_diagnosis_file = os.path.join(full_info_cache_folder, 'full_diagnosis_info.csv')
differential_diagnosis_sample_file = os.path.join(full_info_cache_folder, 'full_diagnosis_info_sample.csv')
