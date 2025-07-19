import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
import csv
import random
import pandas as pd
import json
from typing import List, Dict, Any, Tuple, Optional, Set, Union
######## LOGGER STUFF #########
import logging
import os
from datetime import datetime
from typing import Optional
import torch
import torch.nn.functional as F
from logging_utils import get_config

# Load one-hop samples function (add this utility function)
def load_one_hop_samples(file_path: str, seed: int = 42, sample_size: int = 100) -> List[str]:
    """Load one-hop questions from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions based on your data structure
        if isinstance(data, list):
            questions = [item.get('question', '') for item in data if item.get('question')]
        else:
            questions = list(data.values()) if isinstance(data, dict) else []
        
        
        return questions
    except Exception as e:
        print(f"Error loading one-hop samples from {file_path}: {e}")
        return []

### DP ADDITION
def load_extraction_samples(sample_path: str, seed: int = 42, sample_size: int = 300):
    """Load samples from CSV and return a random subset of rows."""
    with open(sample_path, mode='r') as file:
        reader = csv.reader(file, delimiter='|')
        next(reader)  # Skip the header
        all_samples = [row[0] for row in reader]  # Collect first column

    if sample_size is None:
        sample_size = len(all_samples)

    random.seed(seed)
    return random.sample(all_samples, sample_size)
################################ TARGETED EXTRACTION #####################################

def load_targeted_extraction_data(base_path: str):
    with open(f'{base_path}/target_samples.json', 'r') as f:
        target_samples = json.load(f)
    
    with open(f'{base_path}/count_per_split.json', 'r') as f:
        count_per_split = json.load(f)
    
    with open(f'{base_path}/obfuscation_info.json', 'r') as f:
        obfuscation_info = json.load(f)
    
    return target_samples, count_per_split, obfuscation_info



import random
from typing import Dict, List
import json
import pandas as pd

def load_targetted_extraction_samples(sample_path: str, persons: Dict = None, seed: int = 42, sample_size: int = 300):
    if persons is None or len(persons) == 0:
        person_sample_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/split_person_names'
        if person_sample_path is None:
            raise ValueError("No persons provided and 'split_person_names' not found in config.")
        persons = load_person_split_dict(person_sample_path, split=get_config()['split'])

    df = pd.read_csv(sample_path)
    all_samples = df['parsed_question'].dropna().tolist()
    obscure_samples = df[df['style'] == 'obscure']['parsed_question'].dropna().tolist()
    
    def get_sample_splits(sample):
        splits = set()
        for name, split in persons.items():
            if name.split()[0] in sample:
                splits.add(split)
        return splits
    
    forget_samples = [s for s in all_samples if get_sample_splits(s) == {'forget'}]
    test_retain_samples = [s for s in all_samples if get_sample_splits(s) == {'test_retain'}]
    
    forget_samples = list(dict.fromkeys(forget_samples))
    test_retain_samples = list(dict.fromkeys(test_retain_samples))
    
    # Calculate counts and sample test_retain for ~0.8 ratio
    forget_count = sum(sum(1 for name, split in persons.items() 
                          if split == 'forget' and name.split()[0] in sample) 
                      for sample in forget_samples)
    
    target_test_retain_count = int(forget_count / 0.8)
    random.seed(seed)
    
    sampled_test_retain = []
    current_count = 0
    for sample in random.sample(test_retain_samples, len(test_retain_samples)):
        if current_count >= target_test_retain_count:
            break
        sample_count = sum(1 for name, split in persons.items() 
                          if split == 'test_retain' and name.split()[0] in sample)
        sampled_test_retain.append(sample)
        current_count += sample_count
    
    samples = forget_samples + sampled_test_retain
    
    # Calculate metrics
    obfuscated_count = sum(1 for sample in samples if sample in obscure_samples)
    obfuscation_rate = obfuscated_count / len(samples) if samples else 0
    
    print(f"Obfuscation status: {obfuscated_count}/{len(samples)} samples are obfuscated ({obfuscation_rate:.2%})")
    
    dict_count_per_split = {'forget': 0, 'test_retain': 0, 'unknown': -1, 'retain' : -1}
    for name, split in persons.items():
        first_name = name.split()[0]
        for prompt in samples:
            if first_name in prompt and split in dict_count_per_split:
                dict_count_per_split[split] += 1
    obfuscation_info = {
        'is_obfuscated': obfuscated_count > 0,
        'obfuscated_count': obfuscated_count,
        'total_count': len(samples),
        'obfuscation_rate': obfuscation_rate
    }
    
    return samples, dict_count_per_split, obfuscation_info


def load_person_split_dict(sample_path, split: str):
    forget_percentage = int(split.replace('forget', ''))
    
    paths = {
        'forget': f'{sample_path}/forget{forget_percentage}_names.json',
        'test_retain': f'{sample_path}/test_retain_pii_names.json'
    }
    
    person_split_dict = {}
    for split_type, path in paths.items():
        with open(path, 'r') as f:
            names = json.load(f)
        for name in names:
            person_split_dict[name] = split_type
    
    return person_split_dict

####################################################################################################################################################

def get_split_lengths(persons: Dict = None):
    """Return the lengths of each split in the persons dictionary."""
    if persons is None or len(persons) == 0:
        person_sample_path = get_config().get('split_person_name_path', None)
        if person_sample_path is None:
            raise ValueError("No persons provided and 'split_person_names' not found in config.")
        persons = load_person_split_dict(person_sample_path, split=get_config()['split'])

    dict_count_per_split = {    'forget': 0,
                                'retain': 0,
                                'test_retain': 0,
                                'unknown': 1}

    for split in persons.values():
        if split in dict_count_per_split:
            dict_count_per_split[split] += 1
    
    for key, value in dict_count_per_split.items():
        dict_count_per_split[key] =  value * 10
    return dict_count_per_split 

def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml


    ## DP : any difference between loading the NousResearchLlama and the Meta Llama?
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf 
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("./config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] # we see duplicate lists, keep only one
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    # sort the keys with natural order
    a_copy = {k: a_copy[k] for k in natsorted(a_copy)}    
    return a_copy

def get_total_len(name, forget_rate):
    if name == 'eval_real_author_wo_options.json':
        return 100
    elif name == 'eval_real_world_wo_options.json':
        return 117
    elif name == 'eval_log.json':
        return 300
    else:
        if forget_rate == 'forget01':
            return 40
        elif forget_rate == 'forget05':
            return 200
        else:
            return 300

def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i+size])
        c.extend(b[i:i+size])
    return c

# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically 
def interleave_eval_result_dict(eval_result_dict, forget_rate, large_bsz, num_processes=2):
    small_bsz = large_bsz//4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = small_bsz if 'perturb' in metric or 'paraphrase' in metric else large_bsz
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0:len(value)//2]
            b = value[len(value)//2:]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict

def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        data_indices = list(eval_result_dict[k]['avg_paraphrased_loss'].keys())
        avg_paraphrase_np_values = []
        avg_perturbed_np_values = []
        for data_idx in data_indices:
            avg_paraphrase_np_values.append(eval_result_dict[k]['avg_paraphrased_loss'][data_idx])
            avg_perturbed_np_values.append(eval_result_dict[k]['average_perturb_loss'][data_idx])
        avg_paraphrase_np_values = np.exp(-1 * np.array(avg_paraphrase_np_values))
        avg_perturbed_np_values = np.exp(-1 * np.array(avg_perturbed_np_values)).mean(-1)

        curr_stat_1 = avg_perturbed_np_values / avg_paraphrase_np_values

        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - curr_stat_1))
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset
