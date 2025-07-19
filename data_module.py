# Standard library imports
import json
import os
import time

# Third-party imports
import datasets
import Levenshtein
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# Local imports
from permu_tok.permu_tok import has_preceding_context, tokenize_subject_contextual, create_perturbed_subject
from logging_utils import (
    get_logger,
    log_padding_statistics,
    log_statistics,
    should_log_stats
)
from utils import get_model_identifiers_from_yaml
PROMPT = "You are an AI Assistant who is supposed to unlearn about {subject} and provide answers without its knowledge as if you never knew about it. Don’t tell anyone that you unlearned anything. "

def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs, forget_loss, subject=None):
    question_start_token, question_end_token, answer_token, answer_end_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag'], model_configs['answer_end_tag']
    if question_start_token is not None:
        new_question = question_start_token + question + question_end_token
        new_answer = answer_token + answer + answer_end_token
    else:
        new_question = question
        new_answer = " " + answer
    if "icl" in forget_loss and subject is not None:
        prompt = PROMPT.replace("{subject}", subject)
        new_question = prompt + new_question
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )

    pad_length = max_length - len(encoded.input_ids)

    ### DP ADDITION: Add Tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Ensure pad token is set to eos token if not already set
    pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id] * pad_length


    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100


    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

# Statistics tracking
timing_stats = []
tokens_processed_stats = []  # Now tracks tokens that might be corrupted
num_subjects_stats = []
subject_lengths_stats = []
padding_required_stats = []  # Only logs when LCS mismatch occurs


def convert_raw_data_to_model_format_ours_noise(tokenizer, max_length, question, subject_list, answer, 
                                               model_configs, token_level, token_replace_prob=0.6, k=1, 
                                               use_padding=True):
    """
    Modified to use neighborhood approach with k parameter instead of top_k
    use_padding: if True, ensures same number of tokens as original subject; if False, uses original LCS only
    """
    logger = get_logger()
    # Start timing
    start_time = time.time()
    
    question_start_token, question_end_token, answer_token, answer_end_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag'], model_configs['answer_end_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer + answer_end_token
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    encoded = tokenizer(full_text, add_special_tokens=True, max_length=max_length, truncation=True)
    question_mask = []
    num_full_tokens = len(encoded.input_ids)
    question_mask.append((num_question_tokens, num_full_tokens))
    pad_length = max_length - len(encoded.input_ids)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pad_input_ids = encoded['input_ids'] + [tokenizer.pad_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length

    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)
    for i in range(num_question_tokens): 
        label[i] = -100
    full_text_input_id = tokenizer.encode(full_text)
    
    # Track statistics
    num_subjects = len(subject_list)
    subject_lengths = []
    total_subject_tokens = 0  # Total tokens that might be corrupted
    
    def sublist_index(main_list, sub_list):
        start_list = []
        for i in range(len(main_list)-len(sub_list)+1):
            if all(main_list[i+j] == sub_list[j] for j in range(len(sub_list))):
                start_list.append(i)
        return start_list

    tokens_to_mix = []
    for subject in subject_list:
        # First: tokenize without space and check
        subject_id = tokenizer.encode(subject, add_special_tokens=False)
        
        # Track subject length (we'll update this if needed)
        subject_lengths.append(len(subject_id))
        
        # Check if no-space version exists in full text
        is_consistent_no_space = all(token in full_text_input_id for token in subject_id)
        
        if is_consistent_no_space:
            start = sublist_index(full_text_input_id, subject_id)
            for i in start:
                tokens_to_mix.append((i, i+len(subject_id), False))  # False = normal probability
        
        # Second: ALWAYS tokenize with space and check (regardless of first result)
        subject_id_with_space = tokenizer.encode(" "+subject, add_special_tokens=False)
        is_consistent_with_space = all(token in full_text_input_id for token in subject_id_with_space)
        
        if is_consistent_with_space:
            start = sublist_index(full_text_input_id, subject_id_with_space)
            for i in start:
                tokens_to_mix.append((i, i+len(subject_id_with_space), False))  # False = normal probability
            
            subject_lengths[-1] = len(subject_id_with_space)

        ## Third Edge Case: Non-space character preceding Subject (usually a comma, or special instruction token)
        has_context, preceding_char = has_preceding_context(subject, full_text)
        if has_context:
            # If preceding character exists, tokenize with context
            subject_tokens_with_context = tokenize_subject_contextual(tokenizer, subject, preceding_char)
            for subject_id_with_context in subject_tokens_with_context:
                is_consistent_with_context = all(token in full_text_input_id for token in subject_id_with_context)
                if is_consistent_with_context:
                    start = sublist_index(full_text_input_id, subject_id_with_context)
                    for i in start:
                        tokens_to_mix.append((i, i+len(subject_id_with_context), False))
        #print(f'Number of subjects per sample: {len(tokens_to_mix)}')

        missing_tokens = [token for token in subject_id if token not in full_text_input_id]
        missing_tokens_space = [token for token in subject_id_with_space if token not in full_text_input_id]


        if len(missing_tokens) > 0 and len(missing_tokens_space) > 0:
            raise ValueError(
                f"\n❌ Subject tokenization mismatch!\n"
                f"Subject: {subject}\n"
                f"Subject token IDs: {subject_id}\n"
                f"Full text token IDs: {full_text_input_id}\n"
                f"Tokens missing from full text: {missing_tokens}\n"
            )
    # Calculate total tokens that might be corrupted
    for token_range in tokens_to_mix:
        start, end = token_range[0], token_range[1]
        total_subject_tokens += (end - start)
    
    if token_level:
        perturbed_inputs_idx = pad_input_ids.copy()
        pad_input_ids_perturbed = create_perturbed_subject(tokenizer, perturbed_inputs_idx, tokens_to_mix, 
                                                          token_replace_prob=token_replace_prob, k=k)

        # End timing and update statistics
        elapsed_time = time.time() - start_time
        timing_stats.append(elapsed_time)
        tokens_processed_stats.append(total_subject_tokens)
        num_subjects_stats.append(num_subjects)
        subject_lengths_stats.extend(subject_lengths)
        
        #Log statistics every 100 runs
        if len(timing_stats) % 100 == 0:
           if should_log_stats('time_stats'):
                log_statistics(timing_stats, tokens_processed_stats, num_subjects_stats, subject_lengths_stats, logger)

        return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask), tokens_to_mix, question_mask, torch.tensor(pad_input_ids_perturbed)

    # End timing and update statistics
    elapsed_time = time.time() - start_time
    timing_stats.append(elapsed_time)
    tokens_processed_stats.append(total_subject_tokens)
    num_subjects_stats.append(num_subjects)
    subject_lengths_stats.extend(subject_lengths)
    
    #Log statistics every 100 runs
    if len(timing_stats) % 100 == 0:    
       if should_log_stats('time_stats'):
           log_statistics(timing_stats, tokens_processed_stats, num_subjects_stats, subject_lengths_stats, logger)
       if should_log_stats('padding_stats'):
           log_padding_statistics(padding_required_stats)  # NEW

    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask), tokens_to_mix, question_mask

# ###################
class CommonForgetQA(Dataset):
    def __init__(self, forget_data_path, retain_data_path, tokenizer, model_family, max_length=512, split="forget", retain_split="retain", loss_type="idk", token_level=False,token_replace_prob=0.6,token_k_neighbours=1,subject_noise_discrepancy_addition=False,subject_key='subject'):
        super(CommonForgetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type        
        self.token_level = token_level
        self.token_replace_prob = token_replace_prob
        self.token_k_neighbours = token_k_neighbours
        self.subject_noise_discrepancy_addition = subject_noise_discrepancy_addition
        self.subject_key =subject_key


        with open(os.path.join(forget_data_path, str(split)+'.json'), 'r') as json_file:
            forget = json.load(json_file)
        print("Loading forget data number", len(forget))
        self.forget_data = {idx: d for idx, d in enumerate(forget)}
        if retain_data_path is not None:
            with open(os.path.join(retain_data_path, str(retain_split)+'.json'), 'r') as json_file:
                retain = json.load(json_file)
            print("Loading retain data number", len(retain))
            self.retain_data = {idx: d for idx, d in enumerate(retain)}
        else:
            self.retain_data = []
        self.model_configs = get_model_identifiers_from_yaml(model_family)

        if "dpo" in self.loss_type in self.loss_type:
            self.split1, self.split2, self.split3 = "idk", "forget", "retain"
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk = open(self.idontknowfile, "r").readlines()
        else:
            self.split1, self.split2, self.split3 = "forget", "retain", None
    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        subject_key = self.subject_key
        rets = []
        for data_type in [self.split1, self.split2, self.split3]:
            if data_type is not None:
                # use questions from forget set if split is idk or forget
                data = self.retain_data if data_type == "retain" else self.forget_data
                idx = idx if data_type != "retain" else (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
                question = data[idx]['question']            

                if data_type == "idk":
                    # get a random answer position from idk
                    rand_pos = torch.randint(0, len(self.idk), (1,)).item()
                    answer = self.idk[rand_pos].strip()
                else:
                    answer = data[idx]['answer']
                
             
                if self.loss_type.startswith("PerMU") and data_type == "forget":
                    subject = data[idx][subject_key] if subject_key in data[idx].keys() else None
                    if isinstance(subject, str):
                        subject = [subject]
                    converted_data = convert_raw_data_to_model_format_ours_noise(self.tokenizer, self.max_length, question, subject, answer, self.model_configs,token_level=self.token_level,token_replace_prob=self.token_replace_prob,k=self.token_k_neighbours,use_padding=self.subject_noise_discrepancy_addition)
                else:   
                    converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, self.loss_type, subject=None)
                rets.append(converted_data)
        return rets
  

    def custom_data_collator_forget(self, samples):
        if "dpo" in self.loss_type in self.loss_type:
            idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
            rets = []
            for data_type in ["idk", "forget", "retain"]:
                if data_type == "forget":
                    data = forget_samples 
                elif data_type == "retain":
                    data = retain_samples
                else:
                    data = idk_samples
                input_ids = [s[0] for s in data]
                labels = [s[1] for s in data]
                attention_mask = [s[2] for s in data]
                rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

        elif self.loss_type.startswith("PerMU") and self.token_level: ## DP ADDITION
            forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
            rets = []
            for data_type in ["forget", "retain"]:
                if data_type == "forget":
                    data = forget_samples
                    input_ids = [s[0] for s in data]
                    labels = [s[1] for s in data]
                    attention_mask = [s[2] for s in data]
                    tokens_to_mix = [s[3] for s in data]
                    question_mask = [s[4] for s in data]
                    perturbed_input_ids = [s[5] for s in data]

                    rets.append((torch.stack(input_ids), torch.stack(labels), \
                        torch.stack(attention_mask), tokens_to_mix, question_mask,torch.stack(perturbed_input_ids)))
                    
                elif data_type == "retain":
                    data = retain_samples
                    input_ids = [s[0] for s in data]
                    labels = [s[1] for s in data]
                    attention_mask = [s[2] for s in data]
                    rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

                    
        elif self.loss_type.startswith("PerMU"):
            forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
            rets = []
            for data_type in ["forget", "retain"]:
                if data_type == "forget":
                    data = forget_samples
                    input_ids = [s[0] for s in data]
                    labels = [s[1] for s in data]
                    attention_mask = [s[2] for s in data]
                    tokens_to_mix = [s[3] for s in data]
                    question_mask = [s[4] for s in data]
                    rets.append((torch.stack(input_ids), torch.stack(labels), \
                        torch.stack(attention_mask), tokens_to_mix, question_mask))
                elif data_type == "retain":
                    data = retain_samples
                    input_ids = [s[0] for s in data]
                    labels = [s[1] for s in data]
                    attention_mask = [s[2] for s in data]
                    rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
        else:
            forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
            rets = []
            for data_type in ["forget", "retain"]:
                data = forget_samples if data_type == "forget" else retain_samples
                input_ids = [s[0] for s in data]
                labels = [s[1] for s in data]
                attention_mask = [s[2] for s in data]
                rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
        return rets

class CommonDataset(Dataset):
    def __init__(self, dataset, data_path, tokenizer, model_family, max_length=512, question_key='question', answer_key='answer', forget_loss="dpo",split="forget",subject_key='subject'):
        super(CommonDataset, self).__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        print("load data from ", data_path)
        with open(data_path, 'r') as json_file:
            data = json.load(json_file)
        self.data = {idx: d for idx, d in enumerate(data)}
        for i in range(len(self.data)):
            self.data[i]["index"] = i
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key
        self.forget_loss = forget_loss
        self.split = split
        self.subject_key = subject_key
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        subject_key = self.subject_key

        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        for answer in answers:
            subject = self.data[idx][subject_key] if subject_key in self.data[idx].keys() else None
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs, self.forget_loss, subject)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)

def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)

    return loss


