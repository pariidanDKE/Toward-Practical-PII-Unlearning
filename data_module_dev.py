import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import json
import os
from utils import get_model_identifiers_from_yaml, add_dataset_index
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


def find_top_k_similar_tokens(tokenizer, token_id, k=10):
    """
    Finds the top-k most similar tokens in the tokenizer's vocabulary
    based on Levenshtein distance to a given token_id.
    """
    # Convert the token ID back to its string representation
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    distances = []

    # Iterate through the tokenizer's vocabulary to calculate Levenshtein distances
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        # Skip the original token itself
        if vocab_token_id == token_id:
            continue
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(original_token, vocab_token)
        distances.append((distance, vocab_token_id))

    # Sort by distance (ascending) and take the top k
    distances.sort()
    top_k = [tid for _, tid in distances[:k]]
    return top_k

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
def find_neighbourhood_k(tokenizer, token_id, k=1):
    """
    Finds tokens in the tokenizer's vocabulary that are within a Levenshtein 
    distance of k from the given token_id (neighborhood approach).
    """
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if k > len(original_token):
        print(f"Warning: k={k} is greater than token length={len(original_token)} for token '{original_token}'. "
              f"This means any token can be within distance k.")
    neighbors = []
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if vocab_token_id == token_id:
            continue
        distance = Levenshtein.distance(original_token, vocab_token)
        if distance <= k:
            neighbors.append(vocab_token_id)
    print(f"Token '{original_token}' (ID: {token_id}): Found {len(neighbors)} neighbors within distance k={k}")
    return neighbors

def corrupt_single_token_id(tokenizer, token_id, replace_prob=0.6, k=1):
    """
    Corrupts a single token ID with a similar token from the vocabulary
    based on a given probability.
    """
    if random.random() < replace_prob:
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
    for item in tokens_to_mix:
        if len(item) == 3:
            b, e, reduce_prob = item
            actual_prob = token_replace_prob / 2 if reduce_prob else token_replace_prob
        else:
            b, e = item
            actual_prob = token_replace_prob
        subject_ids = inputs_idx[b:e]
        for i in range(len(subject_ids)):
            subject_ids[i] = corrupt_single_token_id(tokenizer, subject_ids[i], 
                                                   replace_prob=actual_prob, k=k)
        inputs_idx[b:e] = subject_ids
    return inputs_idx

def add_tokens_to_mix_lcs(missing_tokens, full_text_input_id, subject_id, subject, tokens_to_mix):
    """
    Handle tokenization mismatch using LCS and add ranges to tokens_to_mix (original version)
    """
    lcs_indices = longest_common_subsequence_indices(full_text_input_id, subject_id)
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
    return ranges

def add_tokens_to_mix_lcs_with_padding(missing_tokens, full_text_input_id, subject_id, subject, tokens_to_mix):
    """
    Handle tokenization mismatch using LCS and add ranges to tokens_to_mix.
    Ensures the total number of tokens to be perturbed equals the original subject length.
    """
    lcs_indices = longest_common_subsequence_indices(full_text_input_id, subject_id)
    original_subject_length = len(subject_id)
    print(f'Proportion of LCS Indices: {(len(lcs_indices) / len(subject_id)):.2f}')
    print(f'Original subject length: {original_subject_length}, LCS length: {len(lcs_indices)}')
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
        print(f"Need to add {tokens_to_add} additional tokens to match original subject length")
        added_tokens = 0
        for start, end in ranges:
            if added_tokens >= tokens_to_add:
                break
            if start > 0 and added_tokens < tokens_to_add:
                tokens_before = min(tokens_to_add - added_tokens, start)
                if tokens_before > 0:
                    tokens_to_mix.append((start - tokens_before, start, True))  # True = reduced probability
                    added_tokens += tokens_before
                    print(f"Added {tokens_before} tokens before position {start} with 0.5x probability")
            if end < len(full_text_input_id) and added_tokens < tokens_to_add:
                tokens_after = min(tokens_to_add - added_tokens, len(full_text_input_id) - end)
                if tokens_after > 0:
                    tokens_to_mix.append((end, end + tokens_after, True))  # True = reduced probability
                    added_tokens += tokens_after
                    print(f"Added {tokens_after} tokens after position {end} with 0.5x probability")
    return ranges

def convert_raw_data_to_model_format_ours_noise(tokenizer, max_length, question, subject_list, answer, 
                                               model_configs, in_text, token_replace_prob=0.6, k=1, 
                                               use_padding=True):
    """
    Modified to use neighborhood approach with k parameter instead of top_k
    use_padding: if True, ensures same number of tokens as original subject; if False, uses original LCS only
    """
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
    def sublist_index(main_list, sub_list):
        start_list = []
        for i in range(len(main_list)-len(sub_list)+1):
            if all(main_list[i+j] == sub_list[j] for j in range(len(sub_list))):
                start_list.append(i)
        return start_list
    tokens_to_mix = []
    for subject in subject_list:
        if ('phi' in tokenizer.name_or_path):
            subject_id = tokenizer.encode(" "+subject)
        else:
            subject_id = tokenizer.encode(subject, add_special_tokens=False)
        is_consistent = all(token in full_text_input_id for token in subject_id)
        if is_consistent:
            start = sublist_index(full_text_input_id, subject_id)
            for i in start:
                tokens_to_mix.append((i, i+len(subject_id), False))  # False = normal probability
        else:
            missing_tokens = [token for token in subject_id if token not in full_text_input_id]
            print(f"Subject '{subject}' has missing tokens: {missing_tokens}")
            print(f' Full text string: {full_text}')
            print(f' Full text token IDs: {full_text_input_id}')
            print(f' Subject token IDs: {subject_id}')
            print('--'*20)
            try:
                if use_padding:
                    ranges = add_tokens_to_mix_lcs_with_padding(missing_tokens, full_text_input_id, 
                                                               subject_id, subject, tokens_to_mix)
                else:
                    ranges = add_tokens_to_mix_lcs(missing_tokens, full_text_input_id, 
                                                 subject_id, subject, tokens_to_mix)
            except ValueError as e:
                print(f"LCS failed for subject '{subject}': {str(e)}")
                raise ValueError(
                    f"\n❌ Subject tokenization mismatch!\n"
                    f"Subject: {subject}\n"
                    f"Subject token IDs: {subject_id}\n"
                    f"Full text token IDs: {full_text_input_id}\n"
                    f"Tokens missing from full text: {missing_tokens}\n"
                )
    if in_text:
        perturbed_inputs_idx = pad_input_ids.copy()
        pad_input_ids_perturbed = create_perturbed_subject(tokenizer, perturbed_inputs_idx, tokens_to_mix, 
                                                          token_replace_prob=token_replace_prob, k=k)
        decoded_pad_input_ids_perturbed = tokenizer.decode(pad_input_ids_perturbed, skip_special_tokens=True)
        decoded_pad_input_ids = tokenizer.decode(pad_input_ids, skip_special_tokens=True)
        padding_token_id = 2
        filtered_pad_input_ids_perturbed = [
            token_id for token_id in pad_input_ids_perturbed if token_id != padding_token_id
        ]
        filtered_pad_input_ids = [
            token_id for token_id in pad_input_ids if token_id != padding_token_id
        ]
        return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask), tokens_to_mix, question_mask, torch.tensor(pad_input_ids_perturbed)
    return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask), tokens_to_mix, question_mask
    

class CommonForgetQA(Dataset):
    def __init__(self, forget_data_path, retain_data_path, tokenizer, model_family, max_length=512, split="forget", retain_split="retain", loss_type="idk",in_text=False,token_replace_prob=0.6,token_top_k=5):
        super(CommonForgetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type        
        self.in_text = in_text
        self.token_replace_prob = token_replace_prob
        self.token_top_k = token_top_k

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
                    subject = data[idx]['subject'] if 'subject' in data[idx].keys() else None
                    if isinstance(subject, str):
                        subject = [subject]
                    converted_data = convert_raw_data_to_model_format_ours_noise(self.tokenizer, self.max_length, question, subject, answer, self.model_configs,in_text=self.in_text,token_replace_prob=self.token_replace_prob,token_top_k=self.token_top_k)
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
    def __init__(self, dataset, data_path, tokenizer, model_family, max_length=512, question_key='question', answer_key='answer', forget_loss="dpo",split="forget"):
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
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []
        for answer in answers:
            subject = self.data[idx]['subject'] if 'subject' in self.data[idx].keys() else None
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


