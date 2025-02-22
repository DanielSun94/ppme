import json
from srrsh_config import (fused_diagnosis_file_path_template, distinct_diagnosis_path, label_mapping_save_path_template,
                          renmin_mapping_file, standard_mapping_file, final_disease_screening_ready_data_path)
import csv
import torch
from itertools import islice
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from vllm import LLM, SamplingParams
import re
from srrsh_logger import logger
import argparse


def load_model(model_id, cache_folder):
    model_path = os.path.join(cache_folder, model_id)
    # 均使用官方在hugginface上的默认配置
    if model_id == '01ai/Yi-1___5-34B-Chat':
        # 不知道为什么，CUDA_VISIBLE设定为56不报错，但是如果是01会返回乱码
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype='auto'
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False, # 不可设True，解码会报错，原因不明
            padding_side='left',
            trust_remote_code=True
        )
    elif model_id == 'ZhipuAI/glm-4-9b-chat':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
    elif model_id == 'qwen/Qwen2-72B-Instruct':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left'
        )
    else:
        raise ValueError('')
    return model, tokenizer


def load_vllm_model(model_id, cache_folder):
    logger.info('start loading llm and tokenizer')
    model_path = str(os.path.join(cache_folder, model_id))
    # 均使用官方在hugginface上的默认配置（如果有）
    if model_id == '01ai/Yi-1___5-34B-Chat':
        os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
        max_model_len, tp_size = 1024, 2
        model = LLM(
            model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            padding_side='left',
            trust_remote_code=True,
        )
    elif model_id == 'ZhipuAI/glm-4-9b-chat':
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left',
            trust_remote_code=True
        )
        max_model_len, tp_size = 1024, 1
        model = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
    elif model_id == 'qwen/Qwen2-72B-Instruct':
        os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'
        max_model_len, tp_size = 1024, 4
        model = LLM(
            model=model_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side='left'
        )
    else:
        raise ValueError('')
    logger.info('load llm and tokenizer success')
    return model, tokenizer


def diagnosis_tree_generate_remin(file_path):
    data_cache = json.load(open(file_path, 'r', encoding='utf-8-sig'))
    code_mapping_dict, name_mapping_dict = dict(), dict()
    code_name_mapping_dict_list, name_code_mapping_dict_list = [{}, {}, {}, {}], [{}, {}, {}, {}]
    for item in data_cache:
        code_1, name_1, code_2, name_2, code_3, name_3, code_4, name_4 = item[: 8]
        for (key_1, key_2, key_3, key_4), mapping_dict in \
            zip(((code_1, code_2, code_3, code_4), (name_1, name_2, name_3, name_4)),
                (code_mapping_dict, name_mapping_dict)):
            if key_1 not in mapping_dict:
                mapping_dict[key_1] = dict()
            if key_2 != "None" and key_2 not in mapping_dict[key_1]:
                mapping_dict[key_1][key_2] = dict()
            if key_2 != "None" and key_3 != "None" and key_3 not in mapping_dict[key_1][key_2]:
                mapping_dict[key_1][key_2][key_3] = set()
            if key_4 != "None":
                mapping_dict[key_1][key_2][key_3].add(key_4)

        # name 3和name 4如果不做拼接会有key冲突错误
        for index, code, name in ((0, code_1, name_1), (1, code_2, name_2), (2, code_3, name_3),
                                  (3, code_4,  (name_3, name_4))):
            if isinstance(name, str) and name == 'None':
                continue
            if isinstance(name, tuple) and name[1] == 'None':
                continue
            if isinstance(name, tuple):
                name = name[0] + ' ' + name[1]

            if code not in code_name_mapping_dict_list[index]:
                code_name_mapping_dict_list[index][code] = name
            else:
                assert code_name_mapping_dict_list[index][code] == name
            if name not in name_code_mapping_dict_list[index]:
                name_code_mapping_dict_list[index][name] = code
            else:
                assert name_code_mapping_dict_list[index][name] == code
    return code_mapping_dict, name_mapping_dict, code_name_mapping_dict_list, name_code_mapping_dict_list


def diagnosis_tree_generate_standard(mapping_file):
    code_mapping_dict, name_mapping_dict = dict(), dict()
    code_name_mapping_dict_list, name_code_mapping_dict_list = [{}, {}, {}, {}], [{}, {}, {}, {}]
    with open(mapping_file, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            assert len(line) == 11
            code_1, name_1, code_2, name_2, code_3, name_3, code_4, name_4 = line[1:9]
            name_1 = name_1.replace('\\n','').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')
            code_2 = code_2.replace('\\n', '').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')
            name_2 = name_2.replace('\\n', '').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')
            code_3 = code_3.replace('\\n', '').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')
            name_3 = name_3.replace('\\n', '').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')
            code_4 = code_4.replace('\\n', '').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')
            name_4 = name_4.replace('\\n', '').replace('\n', '').replace('\t', '').replace('\0', '').replace('−', '')

            for (key_1, key_2, key_3, key_4), mapping_dict in \
                    zip(((code_1, code_2, code_3, code_4), (name_1, name_2, name_3, name_4)),
                        (code_mapping_dict, name_mapping_dict)):
                if key_1 not in mapping_dict:
                    mapping_dict[key_1] = dict()
                if key_2 != "" and key_2 not in mapping_dict[key_1]:
                    mapping_dict[key_1][key_2] = dict()
                if key_2 != "" and key_3 != "" and key_3 not in mapping_dict[key_1][key_2]:
                    mapping_dict[key_1][key_2][key_3] = set()
                if key_4 != "":
                    mapping_dict[key_1][key_2][key_3].add(key_4)

            for index, code, name in ((0, code_1, name_1), (1, code_2, name_2), (2, code_3, name_3),
                                      (3, code_4, (name_3, name_4))):
                if isinstance(name, str) and name == '':
                    continue
                if isinstance(name, tuple) and name[1] == '':
                    continue
                if isinstance(name, tuple):
                    name = name[0] + ' ' + name[1]
                if code not in code_name_mapping_dict_list[index]:
                    code_name_mapping_dict_list[index][code] = name
                else:
                    if code_name_mapping_dict_list[index][code] != name:
                        # 这里出错是因为官方的文档里面就有不同的ICD Code有完全一致的名称的问题
                        # 情况不多，主要是关节病、非霍奇金淋巴瘤、部分中毒，直接就不管了
                        # print('key duplicate: code: {}, name: {}'.format(code, name))
                        pass
                if name not in name_code_mapping_dict_list[index]:
                    name_code_mapping_dict_list[index][name] = code
                else:
                    if name_code_mapping_dict_list[index][name] != code:
                        # print('key duplicate: code: {}, name: {}'.format(code, name))
                        pass
    return code_mapping_dict, name_mapping_dict, code_name_mapping_dict_list, name_code_mapping_dict_list


def read_data(file_path, max_data_size=-1, examine=False):
    data_dict = dict()
    distinct_label_set = set()
    error_count = 0
    with (open(file_path, 'r', encoding='utf-8-sig') as f):
        reader = csv.reader(f)
        for line in islice(reader, 1, None):
            pk_dcpv, pk_dcpvdiag, code_group, code_org, name_diagtype, name_diagsys, code_diag, name_diag,\
                date_diag, code_psn_diag, name_dept_diag = line
            distinct_label_set.add(name_diag)
            if examine:
                if pk_dcpvdiag not in data_dict:
                    data_dict[pk_dcpvdiag] = line
                else:
                    origin_data = data_dict[pk_dcpvdiag]
                    if not (pk_dcpv == origin_data[0] and origin_data[4] == name_diagtype and name_diag
                            == origin_data[7]):
                        error_count += 1
            else:
                data_dict[pk_dcpvdiag] = line
            if 0 < max_data_size == len(data_dict):
                break
    logger.info('error count: {}'.format(error_count))
    logger.info('data length: {}'.format(len(data_dict)))
    logger.info('distinct diagnosis name length: {}'.format(len(distinct_label_set)))

    distinct_label_list = sorted(list(distinct_label_set))
    return data_dict, distinct_label_list


def get_next_set(index_list, success_set, batch_size):
    new_id_list = []
    for item in index_list:
        if item not in success_set:
            new_id_list.append(item)
        if len(new_id_list) == batch_size:
            break
    return new_id_list


def construct_label(diagnosis, item_list):
    first_label, label_text = '', ''
    for i, item in enumerate(item_list):
        if i == 0:
            first_label = '#{}#  {}\n'.format(i + 1, item)
        label_text += '#{}#  {}\n'.format(i + 1, item)
    text = ("请假设你是一名临床医生，你需要将诊断进行分类。\n"
            "你需要进行分类的诊断是\n{}\n\n你需要把这个诊断归为下列清单中的一类，清单为：\n".format(diagnosis) + label_text + '\n\n'
            '你需要将这一诊断匹配到一个最合适的类型并输出。你只需要输出编号，例如如果: {} 最符合，你只需要输出#1#。'.format(first_label)+
            '注意，你需要把诊断映射到最为相似的类型，并且只能输出一个结果。如果你觉得没有特别合适的选择，'
            '而选项中包含“其它类型的疾病”这样的表述，请选择这个。如果你觉得没有任何一个合适，请随机选择一个。'
            '如果清单中只有一个选择，则不管这个选择看上去合不合理，都选它。'
            )
    return text



def llm_inference(llm_tokenizer, llm_model, prompt_list, vllm_mode, model_config_dict):
    result_list = []
    tokenizer_func = model_config_dict['prompt_generator']
    prompt_formatted_list = []
    for item in prompt_list:
        prompt_formatted_list.append(tokenizer_func(item))

    if not vllm_mode:
        inputs = llm_tokenizer(prompt_formatted_list, return_tensors="pt", padding=True).to("cuda")
        generation_config = model_config_dict['generation_config']
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, **generation_config)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            outputs = torch.unbind(outputs, dim=0)
            for item in outputs:
                result = llm_tokenizer.decode(item, skip_special_tokens=True)
                result_list.append(result)
    else:
        sampling_strategy = model_config_dict['sampling_config']
        if sampling_strategy is not None:
            outputs = llm_model.generate(prompt_formatted_list, sampling_params=sampling_strategy)
        else:
            outputs = llm_model.generate(prompt_formatted_list)
        for output in outputs:
            generated_text = output.outputs[0].text
            result_list.append(generated_text)
    # print(result_list)
    return result_list

def get_generation_util(model_id, tokenizer):
    def qwen_apply_chat(prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text

    def yi_apply_chat(prompt):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
        )
        return text

    def chat_glm_apply_chat(prompt):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return text

    zhipu_stop_token_ids = [151329, 151336, 151338]
    model_config_dict = {
        'ZhipuAI/glm-4-9b-chat': {
            'prompt_generator': chat_glm_apply_chat,
            'generation_config': {"max_new_tokens": 256, "do_sample": True, "top_k": 1},
            'sampling_config': SamplingParams(temperature=0.95, max_tokens=256, stop_token_ids=zhipu_stop_token_ids)
        },
        '01ai/Yi-1___5-34B-Chat': {
            "prompt_generator": yi_apply_chat,
            'generation_config': {"eos_token_id": tokenizer.eos_token_id, "max_new_tokens": 1024},
            'sampling_config': SamplingParams(max_tokens=256)
        },
        'qwen/Qwen2-72B-Instruct': {
            'prompt_generator': qwen_apply_chat,
            'generation_config': {"max_new_tokens": 1024},
            'sampling_config': SamplingParams(max_tokens=256)
        },
    }
    return model_config_dict[model_id]


def label_generation(diagnosis_list, llm_model, llm_tokenizer, batch_size, model_config_dict, name_mapping_dict,
                     vllm_mode, random_seed, name_code_mapping_dict_list, show_result,
                     label_mapping_save_path, save_per_step=200):
    label_generation_manager = LabelGenerationManager(name_mapping_dict, diagnosis_list, batch_size,
                                                      name_code_mapping_dict_list, random_seed)
    previous_success_pool = load_cache(label_mapping_save_path)
    label_generation_manager.initialize(previous_success_pool)

    step = 0
    while not label_generation_manager.has_complete():
        step += 1
        prompt_list, diagnoses_list = label_generation_manager.get_next_iteration_prompt()
        result_list = llm_inference(llm_tokenizer, llm_model, prompt_list, vllm_mode, model_config_dict)
        label_generation_manager.update_manager(result_list, diagnoses_list, show_result)
        if step % save_per_step == 0:
            save_cache(label_generation_manager, label_mapping_save_path)
    logger.info('success')
    save_cache(label_generation_manager, label_mapping_save_path)
    return label_generation_manager


def load_cache(label_mapping_save_path):
    success_pool = []
    if os.path.exists(label_mapping_save_path):
        with open(label_mapping_save_path, 'r', encoding='utf-8-sig') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                diagnosis, category_list, success, random_index, icd_code = line
                assert success == 'True' or success == 'False'
                success = True if success == 'True' else False
                success_pool.append([diagnosis, json.loads(category_list), success, random_index, icd_code])
    return success_pool


def save_cache(label_generation_manager, label_mapping_save_path):
    success_pool = label_generation_manager.success_pool
    success_set = set()

    # save result
    data_to_write = [['diagnosis', 'category_list', 'success', 'random_index', 'icd_code']]
    for item in success_pool:
        success_set.add(item[0])
        data_to_write.append([item[0], json.dumps(item[1]), item[2], item[3], item[4]])
    with open(label_mapping_save_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


class LabelGenerationManager(object):
    def __init__(self, name_mapping_dict, diagnosis_list, batch_size, name_code_mapping_dict_list,
                 random_seed):
        self.name_mapping_dict = name_mapping_dict
        self.diagnosis_list = diagnosis_list
        self.batch_size = batch_size
        self.name_code_mapping_dict_list = name_code_mapping_dict_list
        self.random_seed = random_seed

        self.index_list = None
        self.cursor = 0

        self.success_pool = list()
        self.data_pool = list()

    def initialize(self, previous_pool):
        if len(previous_pool) > 0:
            previous_dict = {item[0]: item for item in previous_pool}
            self.success_pool = previous_pool
            self.cursor, idx = 0, 0
            while idx < self.batch_size:
                diagnosis = self.diagnosis_list[self.cursor]
                self.cursor += 1
                if diagnosis not in previous_dict:
                    self.data_pool.append([self.diagnosis_list[self.cursor], [], False, self.cursor])
                    idx += 1
        else:
            self.cursor = 0
            while self.cursor < self.batch_size:
                self.data_pool.append([self.diagnosis_list[self.cursor], [], False, self.cursor])
                self.cursor += 1
        logger.info('initialize success')

    def get_next_iteration_prompt(self):
        prompt_list, diagnoses_list = [], []
        for item in self.data_pool:
            diagnosis, category_chain, success = item[:3]
            assert not success
            prompt, diagnoses = self.construct_prompt(diagnosis, category_chain)
            prompt_list.append(prompt)
            diagnoses_list.append(diagnoses)
        return prompt_list, diagnoses_list

    def has_complete(self):
        return len(self.success_pool) == len(self.diagnosis_list)

    def update_manager(self, result_list, diagnoses_list, show_result):
        target_key_list = self.parse_result(result_list, diagnoses_list)
        for target_key, item in zip(target_key_list, self.data_pool):
            if target_key is not None:
                item[1].append(target_key)

        # 叶子节点判断，如果这一疾病已经是叶子节点，则说明结构化完毕，
        for item in self.data_pool:
            diagnosis_level_tree = self.name_mapping_dict
            finish_flag = False
            for i, diagnosis in enumerate(item[1]):
                # 此处叶子节点判断有两种case，第一种是depth到4，也就是理论上的最深节点，也就是这里的else
                # 理论上diagnosis一定是set里的一个元素
                # 第二种是depth未到4但是已经没有分支了，体现在diagnosis_level_tree是一个子集为空的set或者dict
                if isinstance(diagnosis_level_tree, dict):
                    diagnosis_level_tree = diagnosis_level_tree[diagnosis]
                    if len(diagnosis_level_tree) == 0:
                        assert i == len(item[1]) - 1
                        finish_flag = True
                else:
                    assert isinstance(diagnosis_level_tree, set)
                    assert i == len(item[1]) - 1
                    assert diagnosis in diagnosis_level_tree
                    finish_flag = True
            if finish_flag:
                item[2] = True

        # 如果成功了，就更新data pool，把相关的数据换掉，并计算code
        for i in range(len(self.data_pool)):
            if self.data_pool[i][2]:
                if show_result:
                    logger.info('success: {}'.format(self.data_pool[i]))
                code = self.convert_to_code(self.data_pool[i][1])
                success_data = self.data_pool[i] + [code]
                assert len(success_data) == 5
                self.success_pool.append(success_data)

        new_data_pool = []
        for i in range(len(self.data_pool)):
            if self.data_pool[i][2]:
                if self.cursor <= len(self.diagnosis_list) - 1:
                    new_data_pool.append([self.diagnosis_list[self.cursor], [], False, self.cursor])
                    self.cursor += 1
            else:
                new_data_pool.append(self.data_pool[i])
        self.data_pool = new_data_pool

    def convert_to_code(self, diagnosis_category):
        if len(diagnosis_category) < 4:
            key = diagnosis_category[-1]
        else:
            assert len(diagnosis_category) == 4
            key = diagnosis_category[-2] + ' ' + diagnosis_category[-1]
        code = self.name_code_mapping_dict_list[len(diagnosis_category) - 1][key]
        code = code.replace('-', '').replace('_', '.')
        assert code is not None
        return code

    @staticmethod
    def parse_result(result_list, diagnoses_list):
        pattern = r"#(\d+)#"
        target_key_list = []
        for i, (result, disease) in enumerate(zip(result_list, diagnoses_list)):
            pattern_list = re.findall(pattern, result)
            # None为错误
            if pattern_list is None or len(pattern_list) == 0 or len(pattern_list) > 1:
                target_key_list.append(None)
            else:
                index = int(pattern_list[0]) - 1
                if index < len(disease):
                    target_key_list.append(disease[index])
                else:
                    target_key_list.append(None)
        return target_key_list


    def construct_prompt(self, diagnosis_name, category_chain):
        diagnosis = diagnosis_name
        if len(category_chain) == 0:
            item_list = sorted(self.name_mapping_dict.keys())
        else:
            key_dict = self.name_mapping_dict
            for item in category_chain:
                key_dict = key_dict[item]
            if isinstance(key_dict, dict):
                item_list = sorted(key_dict.keys())
            else:
                assert isinstance(key_dict, set) and len(key_dict) > 0
                item_list = sorted(list(key_dict))

        prompt_content = construct_label(diagnosis, item_list)
        return prompt_content, item_list


def read_from_final_data(file_path):
    diagnosis_set = set()
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        for line in islice(csv_reader, 1, None):
            table_diagnosis, diagnosis_list = line[5:7]
            if diagnosis_list != 'None':
                diagnosis_list = json.loads(diagnosis_list)
                for item in diagnosis_list:
                    if len(item[1]) > 0:
                        diagnosis_set.add(item[1])
            if table_diagnosis != "None":
                table_diagnosis = json.loads(table_diagnosis)
                for item in table_diagnosis:
                    if len(item) > 0:
                        diagnosis_set.add(item)
    diagnosis_list = sorted(list(diagnosis_set))
    return diagnosis_list



def main():
    # model_id = 'ZhipuAI/glm-4-9b-chat'
    model_id = '01ai/Yi-1___5-34B-Chat'
    # model_id = 'qwen/Qwen2-72B-Instruct'

    llm_cache_folder = '/home/sunzhoujian/llm_cache'
    diagnosis_path = fused_diagnosis_file_path_template.format('False')

    model_id_save = model_id.split('/')[1]
    label_mapping_save_path = label_mapping_save_path_template.format(model_id_save)
    random_seed = 715
    read_from_cache = False
    show_result = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', help='', default=model_id, type=str)
    # 注意，此处的max_data_size指的是数据行数，此处的数据可重复。而batch size指独立数据个数，因此在max_data_size接近batch size时
    # 可能会出现独立数据小于batch size规定而报错
    parser.add_argument('--max_data_size', help='', default=32, type=int)
    parser.add_argument('--batch_size', help='', default=8, type=int)
    parser.add_argument('--vllm_mode_status', help='', default=1, type=int)
    parser.add_argument('--icd_mapping_type', help='', default='standard', type=str)
    parser.add_argument('--diagnosis_source', help='', default='final_dataset', type=str)
    parser.add_argument('--llm_cache_folder', help='', default=llm_cache_folder, type=str)
    args = vars(parser.parse_args())
    for key in args:
        logger.info('{}: {}'.format(key, args[key]))
    assert args['vllm_mode_status'] == 1 or args['vllm_mode_status'] == 0
    vllm_mode = True if args['vllm_mode_status'] == 1 else False
    model_id = args['model_id']
    batch_size = args['batch_size']
    max_data_size = args['max_data_size']
    icd_mapping_type = args['icd_mapping_type']
    diagnosis_source = args['diagnosis_source']
    llm_cache = args['llm_cache_folder']
    # load diagnosis
    if icd_mapping_type == 'renmin':
        # 即使用人卫临床助手维护的ICD映射
        code_mapping_dict, name_mapping_dict, code_name_mapping_dict_list, name_code_mapping_dict_list = (
            diagnosis_tree_generate_remin(renmin_mapping_file))
    else:
        # 使用国家医保局官方出的ICD 医保2.0版本
        assert icd_mapping_type == 'standard'
        code_mapping_dict, name_mapping_dict, code_name_mapping_dict_list, name_code_mapping_dict_list = (
            diagnosis_tree_generate_standard(standard_mapping_file))

    if diagnosis_source == 'table':
        _, diagnosis_list = read_data(diagnosis_path, max_data_size=max_data_size, examine=False)
    elif diagnosis_source == 'discharge_note':
        diagnosis_list = json.load(open(distinct_diagnosis_path, 'r', encoding='utf-8-sig'))
    else:
        assert diagnosis_source == 'final_dataset'
        diagnosis_list = read_from_final_data(final_disease_screening_ready_data_path)
    logger.info('diagnosis list loaded, full data size: {}'.format(len(diagnosis_list)))
    if max_data_size > 0:
        diagnosis_list = diagnosis_list[:max_data_size]
    logger.info('trimmed diagnosis size: {}'.format(len(diagnosis_list)))

    # load model
    if vllm_mode:
        # 如果vLLM时不这么设置会导致模型载入卡死
        os.environ['NCCL_P2P_DISABLE'] = "1"
        llm_model, llm_tokenizer = load_vllm_model(model_id, llm_cache)
    else:
        llm_model, llm_tokenizer = load_model(model_id, llm_cache)
    model_config_dict = get_generation_util(model_id, llm_tokenizer)

    label_generation(
        diagnosis_list, llm_model, llm_tokenizer, batch_size, model_config_dict, name_mapping_dict,
        vllm_mode, random_seed, name_code_mapping_dict_list, show_result, label_mapping_save_path
    )


if __name__ == '__main__':
    main()
