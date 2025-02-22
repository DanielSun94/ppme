import os
from pathlib import Path


root_path = Path(str(os.path.join(os.path.abspath(os.path.dirname(__file__))))).parent.absolute()
resource_folder = os.path.join(root_path, 'resource')
mimic_iii_folder = os.path.join(root_path, 'resource', 'mimic_iii')
reserved_note_path = os.path.join(mimic_iii_folder, 'reserved_note.csv')
save_file = os.path.join(mimic_iii_folder, 'final_mimic_iii.csv')
icd_mapping_file = os.path.join(resource_folder, 'icd9toicd10cmgem.csv')

source_folder = '/mnt/disk_2/sunzhoujian/medical_data/MIMIC-III/'
emr_data_path = os.path.join(source_folder, 'NOTEEVENTS.csv')
diagnosis_data_path = os.path.join(source_folder, 'DIAGNOSES_ICD.csv')

os.makedirs(mimic_iii_folder, exist_ok=True)