import pickle
from joint_data_file_generate import read_emr, read_emr_index
from srrsh_config import outpatient_check_file_1, outpatient_check_file_2
import os

outpatient_type = '1'
print(f'outpatient emr type: {outpatient_type}')

if outpatient_type == '1':
    outpatient_check_file = outpatient_check_file_1
    filter_name = 'outpatient_relevant'
else:
    outpatient_check_file = outpatient_check_file_2
    filter_name = 'outpatient_relevant_2'


if not os.path.exists(outpatient_check_file):
    outpatient_dict = read_emr(filter_name, read_from_cache=True)
    print('outpatient_dict load success')

    mapping_dict = read_emr_index(read_from_cache=True)
    print('mapping_dict load success')

    patient_emr_dict = dict()

    for pk_dcemr in outpatient_dict:
        if pk_dcemr not in mapping_dict:
            continue
        outpatient_pk_dcpv = mapping_dict[pk_dcemr]
        if outpatient_pk_dcpv not in patient_emr_dict:
            patient_emr_dict[outpatient_pk_dcpv] = []
        patient_emr_dict[outpatient_pk_dcpv].append(outpatient_dict[pk_dcemr])
    pickle.dump(patient_emr_dict, open(outpatient_check_file, 'wb'))
else:
    patient_emr_dict = pickle.load(open(outpatient_check_file, 'rb'))

filtered_emr_dict = dict()
for key in patient_emr_dict:
    if len(patient_emr_dict[key]) > 1:
        filtered_emr_dict[key] = patient_emr_dict[key]
print('success')
