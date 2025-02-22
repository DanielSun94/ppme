import os
from pathlib import Path


root_path = Path(str(os.path.join(os.path.abspath(os.path.dirname(__file__))))).parent.absolute()
resource_folder = os.path.join(root_path, 'resource')
mimic_iv_folder = os.path.join(root_path, 'resource', 'mimic_iv')
reserved_note_path = os.path.join(mimic_iv_folder, 'reserved_note.csv')
save_file = os.path.join(mimic_iv_folder, 'final_mimic_iv.csv')
icd_mapping_file = os.path.join(resource_folder, 'icd9toicd10cmgem.csv')

source_folder = '/mnt/disk_2/sunzhoujian/medical_data/MIMIC-IV/'
discharge_data_path = os.path.join(source_folder, 'note', 'discharge.csv')
diagnosis_data_path = os.path.join(source_folder, 'hosp', 'diagnoses_icd.csv')

admission_data_path = os.path.join(source_folder, 'hosp', 'admissions.csv')
patient_data_path = os.path.join(source_folder, 'hosp', 'patients.csv')
os.makedirs(mimic_iv_folder, exist_ok=True)
