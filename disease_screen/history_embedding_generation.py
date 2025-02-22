import os
import json
import pickle
import torch
from datetime import datetime
from disease_screen_logger import logger
from FlagEmbedding import BGEM3FlagModel
from disease_screen_config import fixed_question_answer_folder, history_text_embedding_folder


def get_embedding(data_list, save_size, file_name, btz, embedding_cache_folder, model):
    assert save_size % btz == 0

    parsed_key_set = set()
    exist_file_list = os.listdir(embedding_cache_folder)
    for file in exist_file_list:
        file_path = os.path.join(embedding_cache_folder, file)
        cache_obj = pickle.load(open(file_path, 'rb'))
        for item in cache_obj:
            parsed_key_set.add(item[0])

    to_do_data_list = []
    target_key_set = set()
    for item in data_list:
        target_key_set.add(item[0])
        if item[0] not in parsed_key_set:
            to_do_data_list.append(item)

    logger.info(f'parsed data list length: {len(parsed_key_set)}')
    logger.info(f'parsing data list length: {len(to_do_data_list)}')

    base_iter_num = len(to_do_data_list) // save_size
    iter_num = base_iter_num if len(to_do_data_list) % save_size == 0 else base_iter_num + 1
    for i in range(iter_num):
        save_pkl_name = file_name + f"_{save_size}_" + datetime.now().strftime('%Y%m%d%H%M%S%f') + '.pkl'
        cache_obj_file = os.path.join(embedding_cache_folder, save_pkl_name)
        if not os.path.exists(cache_obj_file):
            result_list = []
            if i < iter_num - 1:
                end_idx = (i + 1) * save_size
            else:
                end_idx = len(to_do_data_list)
            key_list = [item[0] for item in to_do_data_list[i * save_size: end_idx]]
            content_hpi_list = [item[2] for item in to_do_data_list[i * save_size: end_idx]]
            content_without_hpi_list = [item[3] for item in to_do_data_list[i * save_size: end_idx]]

            batch_num = len(key_list) // btz if len(key_list) % btz == 0 else len(key_list) // btz + 1
            for j in range(batch_num):
                batch_end_idx = (j + 1) * btz if j < batch_num - 1 else len(key_list)

                batch_key = [item for item in key_list[j * btz: batch_end_idx]]
                batch_content_1 = [item for item in content_hpi_list[j * btz: batch_end_idx]]
                document_embeddings_1 = model.encode(batch_content_1)['dense_vecs']
                batch_content_2 = [item for item in content_without_hpi_list[j * btz: batch_end_idx]]
                document_embeddings_2 = model.encode(batch_content_2)['dense_vecs']
                for key, embedding_1, embedding_2 in zip(batch_key, document_embeddings_1, document_embeddings_2):
                    result_list.append([key, embedding_1, embedding_2])

            # assert len(result_list) == save_size
            pickle.dump(result_list, open(cache_obj_file, 'wb'))
        logger.info(f'batch {i} success')

    full_result_dict = dict()
    for file in exist_file_list:
        file_path = os.path.join(embedding_cache_folder, file)
        cache_obj = pickle.load(open(file_path, 'rb'))
        for item in cache_obj:
            if item[0] in target_key_set:
                full_result_dict[item[0]] = item[1]
    return full_result_dict


def get_model(model_path, max_length, device):
    model = BGEM3FlagModel(model_path, device=device)
    model.max_seq_length = max_length
    return model


def read_data(folder, filter_key):
    file_list = os.listdir(folder)
    data_list = []
    for file in file_list:
        file_path = os.path.join(folder, file)
        data = json.load(open(file_path, 'r', encoding='utf-8-sig'))
        data_list = data_list + data

    return_data = []
    error_count = 0
    for item in data_list:
        key_1, key_2 = item[0], item[2]
        if filter_key != 'all':
            if filter_key not in key_1:
                continue

        if 'mimic' in key_1:
            language = 'eng'
        else:
            assert 'srrsh' in key_1
            language = 'chn'
        content_with_hpi, content_without_hpi = '', ''
        if "基本信息" in item[1]:
            if language == 'chn':
                key = '基本信息'
            else:
                key = 'demographic info'
            content_without_hpi += f'{key}：{item[1]["基本信息"]}\n'
        else:
            error_count += 1
        if "既往史" in item[1]:
            if language == 'chn':
                key = '既往史'
            else:
                key = 'Previous Medical History'
            content_without_hpi += f'{key}：{item[1]["既往史"]}\n'
        else:
            error_count += 1
        if '家族史' in item[1]:
            if language == 'chn':
                key = '家族史'
            else:
                key = 'Family History'
            content_without_hpi += f'{key}：{item[1]["家族史"]}\n'
        else:
            error_count += 1
        if '现病史' in item[1]:
            if language == 'chn':
                key = '现病史'
            else:
                key = 'History of Present Illness'
            content_with_hpi = content_without_hpi + f'{key}：{item[1]["现病史"]}'
        else:
            error_count += 1
        return_data.append([key_1, key_2, content_with_hpi, content_without_hpi])
    print(f'error count: {error_count}')
    return return_data


def main():
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    max_length = 1024
    save_size = 16384
    btz = 4096
    filter_key = 'all'
    model_path = '/home/sunzhoujian/llm_cache/baai/bge-m3'
    os.makedirs(history_text_embedding_folder, exist_ok=True)

    data_list = read_data(fixed_question_answer_folder, filter_key)
    embedding_model = get_model(model_path, max_length, device)
    get_embedding(data_list, save_size, "history_embedding",
                  btz, history_text_embedding_folder, embedding_model)


if __name__ == "__main__":
    main()
