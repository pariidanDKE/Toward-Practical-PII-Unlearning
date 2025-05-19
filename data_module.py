import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import json
import os
from utils import get_model_identifiers_from_yaml, add_dataset_index

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
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def convert_raw_data_to_model_format_ours_noise(tokenizer, max_length, question, subject_list, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    
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

    #change label to -100 for question tokens
    for i in range(num_question_tokens): 
        label[i] = -100
    
    full_text_input_id = tokenizer.encode(full_text)
    # make sure the tokenized result of the subject word is same with that in input_id
    def sublist_index(main_list, sub_list):
        start_list = []
        for i in range(len(main_list)-len(sub_list)+1):
            if all(main_list[i+j] == sub_list[j] for j in range(len(sub_list))):
                start_list.append(i)
        return start_list
    
    # DP: go through each subject and make sure it was tokenized separately from other parts of the text
    tokens_to_mix = []
    for subject in subject_list:
        if ('phi' in tokenizer.name_or_path):
            subject_id = tokenizer.encode(" "+subject)
        else:
            ## separately encode subject_id, to then pattern match with fully encoded text
            subject_id = tokenizer.encode(subject, add_special_tokens=False)
        is_consistent = all(token in full_text_input_id for token in subject_id)
        if is_consistent is True:
            start = sublist_index(full_text_input_id, subject_id)
        else:
             missing_tokens = [token for token in subject_id if token not in full_text_input_id]
             raise ValueError(
                    f"\n❌ Subject tokenization mismatch!\n"
                    f"Subject: {subject}\n"
                    f"Subject token IDs: {subject_id}\n"
                    f"Full text token IDs: {full_text_input_id}\n"
                    f"Tokens missing from full text: {missing_tokens}\n"
                )
             #raise NotImplementedError(f"Subject word encode wrong: {subject}")
    
        ## add end and start index for the tokens that encode subjects
        for i in start:
            tokens_to_mix.append((i, i+len(subject_id)))
        
    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask), tokens_to_mix, question_mask


class CommonForgetQA(Dataset):
    def __init__(self, forget_data_path, retain_data_path, tokenizer, model_family, max_length=512, split="forget", retain_split="retain", loss_type="idk"):
        super(CommonForgetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type        
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
                    converted_data = convert_raw_data_to_model_format_ours_noise(self.tokenizer, self.max_length, question, subject, answer, self.model_configs)
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


