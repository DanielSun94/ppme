import os.path
import csv
from srrsh_config import (lt_ord_pacs_record_original_path, lt_ord_pacs_record_path, ord_detail_folder,
                          ord_rec_folder, ord_pacs_record_path, ord_pac_record_folder, ord_rec_path,
                          ord_prescription_folder_1, ord_prescription_folder_2, ord_prescription_path,
                          ord_detail_fuse_path)
import json
import traceback
import threading


# 7 数字孪生模拟病人构建数据聚合
def read_ord_detail(path, save_path, read_from_cache):
    if os.path.exists(save_path) and read_from_cache:
        data_list = []
        with open(save_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_list.append(line)
        return data_list

    file_path_list = []
    sub_folder_list = os.listdir(path)
    for sub_folder in sub_folder_list:
        sub_folder_path = os.path.join(path, sub_folder)
        file_list = os.listdir(sub_folder_path)
        for file in file_list:
            json_path = os.path.join(sub_folder_path, file)
            if os.path.isfile(json_path):
                file_path_list.append(json_path)
            else:
                sub_sub_folder = json_path
                json_path_list = os.listdir(sub_sub_folder)
                for file_name in json_path_list:
                    file_path_list.append(os.path.join(sub_sub_folder, file_name))

    error_count = 0
    key_list = ['source_pk', 'pk_rep_lis', 'code_rep', 'create_time', 'name_index_lis', 'value_lis',
                'name_quanti_unit', 'desc_rrs']
    data_list = [key_list]
    for i, file_path in enumerate(file_path_list):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            line_list = f.readlines()
            buffer = ''
            braces = 0
            for line in line_list:
                buffer += line.replace('\\', '')
                braces += line.count('{') - line.count('}')
                if braces == 0 and buffer.strip():
                    try:
                        buffer = buffer.replace('\n', '').replace('\r', '').replace('\t', '')
                        item = json.loads(buffer)
                        source_pk = item['source_pk']
                        pk_rep_lis = item['pk_rep_lis']
                        code_rep = item['code_rep']
                        create_time = item['create_time']
                        name_index_lis = item['name_index_lis']
                        value_lis = item['value_lis']
                        name_quanti_unit = item['name_quanti_unit']
                        desc_rrs = item['desc_rrs']
                        data_list.append([source_pk, pk_rep_lis, code_rep, create_time, name_index_lis, value_lis,
                                          name_quanti_unit, desc_rrs])
                    except json.JSONDecodeError as e:
                        # print(f"JSON 解码错误: {e}")
                        error_message = traceback.format_exc()
                        error_count += 1
                        # print("Error stack as a string:\n", error_message)
                        if error_count > 0 and error_count % 100 == 0:
                            print('error count: {}'.format(error_count))
                    buffer = ''
        print(f'parse {file_path} success, all error count: {error_count}')
    print(f'error count: {error_count}')
    print(f'count length: {data_list}')


    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_list)
    return data_list


def read_lt_ord_pacs_record(path, save_path, read_from_cache):
    if os.path.exists(save_path) and read_from_cache:
        data_list = []
        with open(save_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_list.append(line)
        return data_list

    key_list = ['pk_ord_record', 'result_obj', 'eu_result', 'result_subj', 'note']
    data_list = [key_list]
    data = json.load(open(path, 'r', encoding='utf-8-sig'), strict=False)
    error_count = 0
    for item in data['RECORDS']:
        if ('pk_ord_record' not in item or 'result_obj' not in item or 'eu_result' not in item or
                'result_subj' not in item or item['pk_ord_record'] is None or len(item['pk_ord_record']) < 4 or
                'result_obj' not in item):
            error_count += 1
            print(f'error count: {error_count}')
            continue
        pk_ord_record = item['pk_ord_record']
        result_obj = item['result_obj'] if item['result_obj'] is not None else ""
        eu_result = item['result_obj'] if item['eu_result'] is not None else ""
        result_subj = item['result_subj'] if item['result_subj'] is not None else ""
        note = item['note'] if item['note'] is not None else ""
        data_list.append([pk_ord_record, result_obj, eu_result, result_subj, note])

    print('error count: {}'.format(error_count))
    print('count length: {}'.format(len(data_list)))
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_list)
    return data_list


def read_ord_rec(path, save_path, read_from_cache):
    if os.path.exists(save_path) and read_from_cache:
        data_list = []
        with open(save_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_list.append(line)
        return data_list

    file_path_list = []
    file_list = os.listdir(path)
    for file in file_list:
        file_path_list.append(os.path.join(path, file))

    error_count = 0
    key_list = ['pk_ord_record', 'pk_dcord', 'pk_dcpv', 'pvcode', 'code_sex', 'birthday', 'name_eu_item', 'name_part',
                'date_ris', 'date_rep', 'code_rep', 'code_req', 'code_ord']
    data_list = [key_list]

    for i, file_path in enumerate(file_path_list):
        # if len(data_list) > 100:
        #     break
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            buffer = ''
            braces = 0
            for j, line in enumerate(f.readlines()):
                buffer += line.replace('\\', '')
                braces += line.count('{') - line.count('}')

                if braces == 0 and len(buffer.strip()) > 0:
                    try:
                        buffer = buffer.replace('\n', '').replace('\r', '').replace('\t', '')
                        item = json.loads(buffer, strict=False)
                        pk_ord_record = item['pk_ord_record']
                        pk_dcord = item['pk_dcord']
                        pk_dcpv = item['pk_dcpv']
                        pvcode = item['pvcode']
                        code_sex = item['code_sex']
                        birthday = item['birthday']
                        name_eu_item = item['name_eu_item']
                        name_part = item['name_part']
                        date_ris = item['date_ris']
                        date_rep = item['date_rep']
                        code_rep = item['code_rep']
                        code_req = item['code_req']
                        code_ord = item['code_ord']

                        if pk_dcpv == 'null' or pk_dcpv == "" or pk_dcpv == '_':
                            buffer = ''
                            braces = 0
                            continue
                        data_list.append([pk_ord_record, pk_dcord, pk_dcpv, pvcode, code_sex, birthday, name_eu_item,
                                          name_part, date_ris, date_rep, code_rep, code_req, code_ord])
                        if pk_dcpv != item['pvcode']:
                            print(f'pk_dcpv: {pk_dcpv}, pvcode: {item["pvcode"]}')
                        if pk_dcpv.strip().split('_')[0] != item['code_pati']:
                            print(f'pk_dcpv patient key: {pk_dcpv.strip().split("_")[0]}, pvcode: {item["code_pati"]}')
                    except json.JSONDecodeError as e:
                        # print(f"JSON 解码错误: {e}")
                        error_message = traceback.format_exc()
                        error_count += 1
                        # print("Error stack as a string:\n", error_message)
                        if error_count > 0 and error_count % 100 == 0:
                            print('error count: {}'.format(error_count))
                    buffer = ''
                    braces = 0
        print(f'parse {file_path} success, all error count: {error_count}')
    print('error count: {}'.format(error_count))
    print('count length: {}'.format(len(data_list)))
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_list)
    return data_list


def read_ord_pac_record(path, save_path, read_from_cache):
    if os.path.exists(save_path) and read_from_cache:
        data_list = []
        with open(save_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_list.append(line)
        return data_list

    file_path_list = []
    file_list = os.listdir(path)
    for file in file_list:
        file_path_list.append(os.path.join(path, file))

    error_count = 0
    key_list = ['pk_pacs', 'pk_ord_record', 'name_diag', 'result_obj', 'result_subj', 'eu_result',
                'note', 'source_pk', 'create_time']
    data_list = [key_list]

    for i, file_path in enumerate(file_path_list):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            buffer = ''
            braces = 0
            for line in f.readlines():
                buffer += line.replace('\\', '')
                braces += line.count('{') - line.count('}')
                if braces == 0 and buffer.strip():
                    try:
                        buffer = buffer.replace('\n', '').replace('\r', '').replace('\t', '')
                        item = json.loads(buffer, strict=False)
                        pk_pacs = item['pk_pacs']
                        pk_ord_record = item['pk_ord_record']
                        name_diag = item['name_diag'] if 'name_diag' in item else ""
                        result_obj = item['result_obj'] if 'result_obj' in item else ""
                        result_subj = item['result_subj'] if 'result_subj' in item else ""
                        eu_result = item['eu_result'] if 'eu_result' in item else ""
                        note = item['note'] if 'note' in item else ""
                        source_pk = item['source_pk']
                        create_time = item['create_time']
                        data_list.append([pk_pacs, pk_ord_record, name_diag, result_obj, result_subj,
                                          eu_result, note, source_pk, create_time])
                    except json.JSONDecodeError as e:
                        # print(f"JSON 解码错误: {e}")
                        error_message = traceback.format_exc()
                        error_count += 1
                        # print("Error stack as a string:\n", error_message)
                        if error_count > 0 and error_count % 100 == 0:
                            print('error count: {}'.format(error_count))
                    buffer = ''
        print(f'parse {file_path} success, all error count: {error_count}')
    print('error count: {}'.format(error_count))
    print('count length: {}'.format(len(data_list)))
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_list)
    return data_list


def read_ord(path_1, path_2, save_path, read_from_cache):
    if os.path.exists(save_path) and read_from_cache:
        data_list = []
        with open(save_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_list.append(line)
        return data_list

    file_path_list = []
    sub_folder_list = os.listdir(path_2)
    for sub_folder in sub_folder_list:
        sub_folder_path = os.path.join(path_2, sub_folder)
        for file in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, file)
            file_path_list.append(file_path)

    sub_folder_list = os.listdir(path_1)
    for sub_folder in sub_folder_list:
        sub_folder_path = os.path.join(path_1, sub_folder)
        file_list = os.listdir(sub_folder_path)
        for file in file_list:
            json_path = os.path.join(sub_folder_path, file)
            if os.path.isfile(json_path):
                file_path_list.append(json_path)
            else:
                sub_sub_folder = json_path
                json_path_list = os.listdir(sub_sub_folder)
                for file_name in json_path_list:
                    file_path_list.append(os.path.join(sub_sub_folder, file_name))

    error_count = 0
    key_list = ['pk_dcpv', 'pvcode', 'code_ord', 'code_rep', 'code_req', 'name_sex', 'birthday', 'name_orditem',
                'desc_ord', 'create_time', 'date_create']
    data_list = [key_list]
    for i, file_path in enumerate(file_path_list):
        # if len(data_list) > 100:
        #     break
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            line_list = f.readlines()
            buffer = ''
            braces = 0
            for line in line_list:
                buffer += line.replace('\\', '')
                braces += line.count('{') - line.count('}')
                if braces == 0 and buffer.strip():
                    try:
                        buffer = buffer.replace('\n', '').replace('\r', '').replace('\t', '')
                        item = json.loads(buffer)
                        # pk_dcord = item['pk_dcord'] if 'code_ord' in item else ''
                        pk_dcpv = item['pk_dcpv']
                        pvcode = item['pvcode']
                        name_sex = item['name_sex']
                        birthday = item['birthday']
                        name_orditem = item['name_orditem'] if 'name_orditem' in item else ''
                        desc_ord = item['desc_ord'] if 'desc_ord' in item else ''
                        create_time = item['create_time']
                        date_create = item['date_create'] if 'date_create' in item else ''
                        # source_pk = item['source_pk']
                        code_ord = item['code_ord'] if 'code_ord' in item else ''
                        code_rep = item['code_rep'] if 'code_rep' in item else ""
                        code_req = item['code_req'] if 'code_req' in item else ""
                        if len(create_time) == 0 and len(date_create) == 0:
                            assert ValueError('time lost')
                        data_list.append([pk_dcpv, pvcode, code_ord, code_rep, code_req, name_sex, birthday,
                                          name_orditem, desc_ord, create_time, date_create])
                    except Exception as e:
                        # print(f"JSON 解码错误: {e}")
                        error_message = traceback.format_exc()
                        error_count += 1
                        print(f'buffer: {buffer}')
                        print("Error stack as a string:\n", error_message)
                        if error_count > 0 and error_count % 100 == 0:
                            print('error count: {}'.format(error_count))
                    buffer = ''
        print(f'parse {file_path} success, [{i}/{len(file_path_list)}] all error count: {error_count}')
    print('error count: {}'.format(error_count))
    print('count length: {}'.format(len(data_list)))

    with open(save_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_list)
    return data_list


def main():
    # 这里的json dump不知道他们怎么弄的，可能出现一个line因为value中有换行符导致readline读取部分item，解析错误的问题，
    # 因此要设置一个缓存区，判别是否读完了
    #
    # read_ord(ord_prescription_folder_1, ord_prescription_folder_2, ord_prescription_path, False)
    # print('ord_prescription load success')
    # read_ord_detail(ord_detail_folder, ord_detail_fuse_path, True)
    # print('ord_detail load success')
    read_ord_rec(ord_rec_folder, ord_rec_path, True)
    print('ord_rec load success')
    # read_ord_pac_record(ord_pac_record_folder, ord_pacs_record_path, False)
    # print('ord_pac_record load success')
    # read_lt_ord_pacs_record(lt_ord_pacs_record_original_path, lt_ord_pacs_record_path, False)
    # print('lt_ord_pacs_record load success')

    # time.sleep(1000)
    print('')


if __name__ == '__main__':
    main()
