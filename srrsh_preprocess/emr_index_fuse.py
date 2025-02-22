import csv
import json
from srrsh_config import emr_index_folder, emr_index_fuse_file_template
from srrsh_logger import logger
import os


# 1
def main():
    head = ['pk_dcemr', 'pk_dcpv', 'code_group', 'code_org', 'code_pati', 'name_pvtype', 'code_pvtype', 'code_ref_emr',
            'code_emr_type', 'source_pk', 'create_time']

    success_count, failed_count = 0, 0
    sub_folder_list = os.listdir(emr_index_folder)
    sub_folder_list = sorted(sub_folder_list, reverse=True)
    for sub_folder in sub_folder_list:
        data_to_write = [head]
        sub_folder_path = os.path.join(emr_index_folder, sub_folder)
        file_list = os.listdir(sub_folder_path)

        # if os.path.exists(emr_index_fuse_file_template.format(sub_folder)):
        #     continue

        for file in file_list:
            file_path = os.path.join(sub_folder_path, file)
            logger.info('start parse file: {}'.format(file_path))

            with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                    except Exception as e:
                        # print('parse error: {}'.format(e))
                        failed_count += 1
                        continue

                    sample = []
                    success_flag = True
                    for key in head:
                        if key in data:
                            sample.append(data[key])
                        else:
                            success_flag = False
                            break

                    if success_flag:
                        success_count += 1
                        data_to_write.append(sample)
                    else:
                        failed_count += 1

                    if success_count > 0 and success_count % 100000 == 0:
                        logger.info('success_count: {}, failed_count: {}'.format(success_count, failed_count))
        logger.info('success_count: {}, failed_count: {}'.format(success_count, failed_count))
        with open(emr_index_fuse_file_template.format(sub_folder), 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)


if __name__ == '__main__':
    main()
