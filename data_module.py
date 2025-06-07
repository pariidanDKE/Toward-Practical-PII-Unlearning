import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import json
import os
from utils import get_model_identifiers_from_yaml, count_actual_corruptions, log_padding_statistics,log_tokenization_misalignment, should_log_stats,add_subject_lengths,get_config,add_corrupted_subject_info
import Levenshtein

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

    tokenizer.padding_side = 'left' ## DP : NEW ADDITION
    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)


# #### CLAUDE PASTE ####
def longest_common_subsequence_indices(list1, list2):
    """
    Find the longest common subsequence and return indices in list1 where it occurs
    """
    m, n = len(list1), len(list2)
    
    # Create LCS table
    lcs_table = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill LCS table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i-1] == list2[j-1]:
                lcs_table[i][j] = lcs_table[i-1][j-1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])
    
    # Backtrack to find LCS indices in list1
    lcs_indices = []
    i, j = m, n
    while i > 0 and j > 0:
        if list1[i-1] == list2[j-1]:
            lcs_indices.append(i-1)
            i -= 1
            j -= 1
        elif lcs_table[i-1][j] > lcs_table[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    lcs_indices.reverse()
    return lcs_indices

import random
import re

def is_latin_alphabet_only(text):
    """
    Check if a token contains only Latin alphabet characters, numbers, 
    common punctuation, and whitespace/special tokenizer symbols.
    """
    # Remove common tokenizer prefixes/symbols
    cleaned_text = text.replace('▁', '').replace('Ġ', '').replace('##', '')
    
    # Allow Latin letters, numbers, common punctuation, and whitespace
    # This regex matches: letters (a-zA-Z), numbers (0-9), common punctuation, whitespace
    latin_pattern = r'^[a-zA-Z0-9\s\.,;:!?\-\'\"()\[\]{}/@#$%^&*+=<>|\\~`_]*$'
    
    return bool(re.match(latin_pattern, cleaned_text))

def find_neighbourhood_k(tokenizer, token_id, k=1):
    """
    Finds tokens in the tokenizer's vocabulary that are within a Levenshtein 
    distance of k from the given token_id (neighborhood approach).
    Only returns tokens with Latin alphabet characters.
    """
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]

    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)


    def get_first_alpha_char(token):
            for char in token:
                if char.isalpha():
                    return char
            return ''

    # Check if the token is a digit
    if original_token.strip().isdigit():
        # For digit tokens, only return other single digit tokens
        neighbors = []
        decoded_neighbours = []
        
        # Try to find all single digits in the vocabulary
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            # Check different possible formats
            possible_formats = [
                digit,  # Just the digit
                f' {digit}',  # Space + digit
                f'▁{digit}',  # Common in sentencepiece tokenizers
            ]
            
            for formatted_digit in possible_formats:
                if formatted_digit in tokenizer.vocab:
                    digit_id = tokenizer.vocab[formatted_digit]
                    if digit_id != token_id:  # Don't include the original token
                        neighbors.append(digit_id)
                        decoded_neighbours.append(formatted_digit)
        

        return neighbors


    match_first_char = get_config()['match_first_char']
    original_first_char = get_first_alpha_char(original_token) if original_token and match_first_char else ''


    neighbors = []
    decoded_neighbours = []
    
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if vocab_token_id == token_id:
            continue
            
        # Filter out non-Latin alphabet tokens
        if not is_latin_alphabet_only(vocab_token):
            continue

        if match_first_char and (not vocab_token or get_first_alpha_char(vocab_token) != original_first_char):
            continue

        distance = Levenshtein.distance(original_token, vocab_token)
        if distance <= k:
            neighbors.append(vocab_token_id)
            decoded_neighbours.append(vocab_token)
    #print(f"Token '{original_token}' (ID: {token_id}): Found {len(neighbors)} Latin neighbors within distance k={k}")

    # if len(neighbors) >= 30:
    #     print(f"Neighbors: {decoded_neighbours[:30]}")
    # else:
    #     print(f"Neighbors: {decoded_neighbours}")

    return neighbors

def find_neighbourhood_k_adaptive_strict(tokenizer, token_id, k=1):
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)
    
    original_length = len(original_token)
    neighbors = []
    
    if original_token.strip().isdigit():
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            for fmt in [digit, f' {digit}', f'▁{digit}']:
                if fmt in tokenizer.vocab:
                    digit_id = tokenizer.vocab[fmt]
                    if digit_id != token_id and Levenshtein.distance(original_token, fmt) == original_length:
                        neighbors.append(digit_id)
        return neighbors
    
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if (vocab_token_id != token_id and 
            is_latin_alphabet_only(vocab_token) and 
            len(vocab_token) == original_length and 
            Levenshtein.distance(original_token, vocab_token) == original_length):
            neighbors.append(vocab_token_id)

            # if random.random() < 0.001:
            #     print('Added option for token ', original_token, '; Option: ', vocab_token)
    
    return neighbors
    


def corrupt_single_token_id(tokenizer, token_id, replace_prob=0.6, k=1):
    """
    Corrupts a single token ID with a similar token from the vocabulary
    based on a given probability.
    """
    if random.random() < replace_prob:
        if get_config()['use_adaptive_k']:
            neighbor_ids = find_neighbourhood_k_adaptive_strict(tokenizer, token_id, k=k)
        else:
            neighbor_ids = find_neighbourhood_k(tokenizer, token_id, k=k)
        if neighbor_ids:
            sampled_id = random.choice(neighbor_ids)
            return sampled_id
        else:
            return token_id
    else:
        return token_id



def create_perturbed_subject(tokenizer, inputs_idx, tokens_to_mix, token_replace_prob=0.6, k=1):
    """
    Create perturbed subject with different probabilities for different token ranges.
    tokens_to_mix contains tuples of (start, end, reduce_prob) where reduce_prob is boolean
    """


    original_subjects_decoded = []
    corrupted_subjects_decoded = []


    for item in tokens_to_mix:
        if len(item) == 3:
            b, e, reduce_prob = item
            actual_prob = token_replace_prob / 2 if reduce_prob else token_replace_prob
        else:
            b, e = item
            actual_prob = token_replace_prob
        subject_ids = inputs_idx[b:e]
        original_subjects_decoded.append(tokenizer.decode(subject_ids, skip_special_tokens=True))




        for i in range(len(subject_ids)):
            original_token = subject_ids[i]
            #subject_ids.append(original_token)  
            subject_ids[i] = corrupt_single_token_id(tokenizer, subject_ids[i],  
                                                   replace_prob=actual_prob, k=k)

        inputs_idx[b:e] = subject_ids
        corrupted_subjects_decoded.append(tokenizer.decode(subject_ids, skip_special_tokens=True))

    add_corrupted_subject_info(original_subjects_decoded, corrupted_subjects_decoded)


    return inputs_idx

def add_tokens_to_mix_lcs(missing_tokens, full_text_input_id, subject_id, subject, tokens_to_mix):
    """
    Handle tokenization mismatch using LCS and add ranges to tokens_to_mix (original version)
    """
    lcs_indices = longest_common_subsequence_indices(full_text_input_id, subject_id)
    lcs_coverage = len(lcs_indices) / len(subject_id)  # NEW: Track this

    print(f'Proportion of LCS Indices: {(len(lcs_indices) / len(subject_id)):.2f}')
    if len(lcs_indices) < 0.2 * len(subject_id):
        raise ValueError(
            f"\n❌ Subject tokenization mismatch - too many tokens lost!\n"
            f"Subject: {subject}\n"
            f"Subject token IDs: {subject_id}\n"
            f"Full text token IDs: {full_text_input_id}\n"
            f"Tokens missing from full text: {missing_tokens}\n"
            f"LCS coverage: {len(lcs_indices)}/{len(subject_id)} ({len(lcs_indices)/len(subject_id)*100:.1f}%)\n"
        )
    ranges = []
    if lcs_indices:
        start_idx = lcs_indices[0]
        end_idx = lcs_indices[0]
        for i in range(1, len(lcs_indices)):
            if lcs_indices[i] == lcs_indices[i-1] + 1:
                end_idx = lcs_indices[i]
            else:
                ranges.append((start_idx, end_idx + 1))
                start_idx = lcs_indices[i]
                end_idx = lcs_indices[i]
        ranges.append((start_idx, end_idx + 1))
        for range_start, range_end in ranges:
            tokens_to_mix.append((range_start, range_end))
    return ranges, lcs_coverage


def add_tokens_to_mix_lcs_with_padding(missing_tokens, full_text_input_id, subject_id, subject, tokens_to_mix):
    """
    Handle tokenization mismatch using LCS and add ranges to tokens_to_mix.
    Ensures the total number of tokens to be perturbed equals the original subject length.
    Returns ranges, lcs_coverage, actual_tokens_before, actual_tokens_after
    """
    lcs_indices = longest_common_subsequence_indices(full_text_input_id, subject_id)
    lcs_coverage = len(lcs_indices) / len(subject_id)  # NEW: Track this

    original_subject_length = len(subject_id)
    #print(f'Proportion of LCS Indices: {(len(lcs_indices) / len(subject_id)):.2f}')
    #print(f'Original subject length: {original_subject_length}, LCS length: {len(lcs_indices)}')
    if len(lcs_indices) < 0.2 * len(subject_id):
        raise ValueError(
            f"\n❌ Subject tokenization mismatch - too many tokens lost!\n"
            f"Subject: {subject}\n"
            f"Subject token IDs: {subject_id}\n"
            f"Full text token IDs: {full_text_input_id}\n"
            f"Tokens missing from full text: {missing_tokens}\n"
            f"LCS coverage: {len(lcs_indices)}/{len(subject_id)} ({len(lcs_indices)/len(subject_id)*100:.1f}%)\n"
        )
    
    ranges = []
    total_lcs_tokens = 0
    actual_tokens_before = []  # NEW: Track actual tokens added before
    actual_tokens_after = []   # NEW: Track actual tokens added after
    
    if lcs_indices:
        start_idx = lcs_indices[0]
        end_idx = lcs_indices[0]
        for i in range(1, len(lcs_indices)):
            if lcs_indices[i] == lcs_indices[i-1] + 1:
                end_idx = lcs_indices[i]
            else:
                ranges.append((start_idx, end_idx + 1))
                total_lcs_tokens += (end_idx + 1 - start_idx)
                start_idx = lcs_indices[i]
                end_idx = lcs_indices[i]
        ranges.append((start_idx, end_idx + 1))
        total_lcs_tokens += (end_idx + 1 - start_idx)
        for range_start, range_end in ranges:
            tokens_to_mix.append((range_start, range_end, False))  # False = normal probability
    
    tokens_to_add = original_subject_length - total_lcs_tokens
    if tokens_to_add > 0:
        #print(f"Need to add {tokens_to_add} additional tokens to match original subject length")
        added_tokens = 0
        for start, end in ranges:
            if added_tokens >= tokens_to_add:
                break
            if start > 0 and added_tokens < tokens_to_add:
                tokens_before = min(tokens_to_add - added_tokens, start)
                if tokens_before > 0:
                    # NEW: Capture actual tokens before
                    before_range_start = start - tokens_before
                    before_range_end = start
                    actual_before_tokens = full_text_input_id[before_range_start:before_range_end]
                    actual_tokens_before.extend(actual_before_tokens)
                    
                    tokens_to_mix.append((before_range_start, before_range_end, True))  # True = reduced probability
                    added_tokens += tokens_before
                    #print(f"Added {tokens_before} tokens before position {start} with 0.5x probability")
            if end < len(full_text_input_id) and added_tokens < tokens_to_add:
                tokens_after = min(tokens_to_add - added_tokens, len(full_text_input_id) - end)
                if tokens_after > 0:
                    # NEW: Capture actual tokens after
                    after_range_start = end
                    after_range_end = end + tokens_after
                    actual_after_tokens = full_text_input_id[after_range_start:after_range_end]
                    actual_tokens_after.extend(actual_after_tokens)
                    
                    tokens_to_mix.append((after_range_start, after_range_end, True))  # True = reduced probability
                    added_tokens += tokens_after
                    #print(f"Added {tokens_after} tokens after position {end} with 0.5x probability")
    
    return ranges, lcs_coverage, actual_tokens_before, actual_tokens_after




import time
import numpy as np
from utils import get_logger, log_statistics, log_final_statistics

# Statistics tracking
timing_stats = []
tokens_processed_stats = []  # Now tracks tokens that might be corrupted
num_subjects_stats = []
subject_lengths_stats = []
padding_required_stats = []  # Only logs when LCS mismatch occurs



def convert_raw_data_to_model_format_ours_noise(tokenizer, max_length, question, subject_list, answer, 
                                               model_configs, in_text, token_replace_prob=0.6, k=1, 
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
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
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
    current_sample_lcs_coverages = []
    for subject in subject_list:
        # First try: original encoding without space
        subject_id = tokenizer.encode(subject, add_special_tokens=False)
        
        # Track subject length
        subject_lengths.append(len(subject_id))
        is_consistent = all(token in full_text_input_id for token in subject_id)
        
        if is_consistent:
            start = sublist_index(full_text_input_id, subject_id)
            for i in start:
                tokens_to_mix.append((i, i+len(subject_id), False))  # False = normal probability
        else:
            # Second try: add space prefix
            subject_id_with_space = tokenizer.encode(" "+subject)
            is_consistent_with_space = all(token in full_text_input_id for token in subject_id_with_space)
            
            if is_consistent_with_space:
                # Update subject_id and subject_lengths for the space version
                subject_id = subject_id_with_space
                subject_lengths[-1] = len(subject_id)  # Update the length we just added
                
                start = sublist_index(full_text_input_id, subject_id)
                for i in start:
                    tokens_to_mix.append((i, i+len(subject_id), False))  # False = normal probability
                continue  # Skip to next subject
            
            # Third try: fall back to LCS approach if space prefix also didn't work
            missing_tokens = [token for token in subject_id if token not in full_text_input_id]
            #log_tokenization_misalignment(subject, subject_id, full_text_input_id, missing_tokens, full_text, tokenizer)
            try:
                if use_padding:
                    ranges, lcs_coverage, tokens_before, tokens_after = add_tokens_to_mix_lcs_with_padding(missing_tokens, full_text_input_id, 
                                                               subject_id, subject, tokens_to_mix)
                    
                    if should_log_stats('padding_stats'):
                        log_tokenization_misalignment(subject, subject_id, full_text_input_id, missing_tokens, full_text, tokenizer,
                                                  actual_tokens_before=tokens_before, actual_tokens_after=tokens_after)
                else:
                    ranges, lcs_coverage = add_tokens_to_mix_lcs(missing_tokens, full_text_input_id, 
                                                 subject_id, subject, tokens_to_mix)
                current_sample_lcs_coverages.append(lcs_coverage)

            except ValueError as e:
                logger.error(f"LCS failed for subject '{subject}': {str(e)}")
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
    
    if in_text:
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
    def __init__(self, forget_data_path, retain_data_path, tokenizer, model_family, max_length=512, split="forget", retain_split="retain", loss_type="idk",in_text=False,token_replace_prob=0.6,token_k_neighbours=1,subject_noise_discrepancy_addition=False,subject_key='subject'):
        super(CommonForgetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type        
        self.in_text = in_text
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
                    converted_data = convert_raw_data_to_model_format_ours_noise(self.tokenizer, self.max_length, question, subject, answer, self.model_configs,in_text=self.in_text,token_replace_prob=self.token_replace_prob,k=self.token_k_neighbours,use_padding=self.subject_noise_discrepancy_addition)
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

        elif self.loss_type.startswith("PerMU") and self.in_text: ## DP ADDITION
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


