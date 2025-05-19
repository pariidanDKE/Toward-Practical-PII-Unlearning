# Standard libraries
import csv
import re
import random as rand
import pickle
import time
import json
# Third-party libraries
import torch
from tqdm import tqdm

from transformers import (
    BertForMaskedLM, BertTokenizer,
    DistilBertForMaskedLM, DistilBertTokenizer,
    RobertaForMaskedLM, RobertaTokenizer,
    AutoModelForCausalLM, AutoTokenizer
)


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load, clean and shorten the emails
def process_emails(filepath, max_len=50):
    with open(filepath, 'r') as f:
        emails = [row[0] for row in csv.reader(f, delimiter='|')][1:]
    def clean(e):
        e = re.sub(r'[^\w\s]', '', e)
        e = re.sub(r'[_\-*=~`^\\|/]+', ' ', e)
        return re.sub(r'\s+', ' ', e).strip()
    # def shorten(e):
    #     return e if len(e) <= max_len else e[rand.randint(0, len(e)-max_len):][:max_len]
    return [(clean(e)[:max_len]) for e in emails]




#Load search model


def load_mlm_model(model_name='bert'):
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    elif model_name == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForMaskedLM.from_pretrained('roberta-base')
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return tokenizer, model

search_tokenizer, search_model = load_mlm_model('bert')

def load_causal_lm_model(is_tofu=False, is_pii=False, unlearned=False):
    branch = None

    if is_tofu:
        if unlearned:
            repo_id = '/projects/0/hpmlprjs/LLM/danp/UGBench/experiment/TOFU/llama2-7b/forget10/EvaluateLoRA_EvaluateFulModel_llama2-7b_WarmupConstantLR'
            tokenizer_path = repo_id
        else:
            repo_id = 'locuslab/tofu_ft_llama2-7b'
            tokenizer_path = repo_id
    elif is_pii:
        if unlearned:
            repo_id = '/projects/0/hpmlprjs/LLM/danp/UGBench/experiment/PII/llama2-7b/forget10/FullFT_PIIPerMU_llama2-7b_WarmupConstantLR'
            tokenizer_path = repo_id
        else:
            repo_id = '/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_llama2-7b_B8_G4_E10_lr1e-5/checkpoint-393'
            tokenizer_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_llama2-7b_B8_G4_E10_lr1e-5'
    else:
        repo_id = 'NousResearch/Llama-2-7b-chat-hf'
        tokenizer_path = repo_id

    model = AutoModelForCausalLM.from_pretrained(repo_id, revision=branch)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, revision=branch)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


## Get loss function for Orig and Loss function for neighbour set
def get_logprob(text,attack_tokenizer,attack_model):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:0')
    logprob = - attack_model(text_tokenized, labels=text_tokenized).loss.item()

    return logprob

def get_logprob_batch(text,attack_tokenizer,attack_model):
    text_tokenized = attack_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to('cuda:0')

    ce_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=attack_tokenizer.pad_token_id)
    logits = attack_model(text_tokenized, labels=text_tokenized).logits[:,:-1,:].transpose(1,2)
    manual_logprob = - ce_loss(logits, text_tokenized[:,1:])
    mask = manual_logprob!=0
    manual_logprob_means = (manual_logprob*mask).sum(dim=1)/mask.sum(dim=1)


    return manual_logprob_means.tolist()


# Neighbourhood Attack Method
from heapq import nlargest
import string

def generate_neighbours_alt(text,search_model,search_model_name,num_word_changes=1,dropout_p=0.2):
    token_dropout = torch.nn.Dropout(p=dropout_p)
    text_tokenized = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(device)
    original_text = search_tokenizer.batch_decode(text_tokenized)[0]

    candidate_scores = dict()
    replacements = dict()

    for target_token_index in list(range(len(text_tokenized[0,:])))[1:]:
        target_token = text_tokenized[0,target_token_index]
        
        decoded_token = search_tokenizer.decode([target_token])
        print(f"Token ID: {target_token:<5} -> Decoded: '{decoded_token}'")

        # skip_tokens = [10975, 39236, 742,48505, 39236, 742]
        # if target_token in skip_tokens:
        #     print(target_token)
        #     continue
        
        
        #embeds = search_model.embeddings(text_tokenized)

        if search_model_name == 'bert':
            embeds = search_model.bert.embeddings(text_tokenized)
        elif search_model_name == 'distilbert':
            embeds = search_model.distilbert.embeddings(text_tokenized)
        elif search_model_name == 'roberta':
            embeds = search_model.roberta.embeddings(text_tokenized)
            
        embeds = torch.cat((embeds[:,:target_token_index,:], token_dropout(embeds[:,target_token_index,:]).unsqueeze(dim=0), embeds[:,target_token_index+1:,:]), dim=1)
        
        token_probs = torch.softmax(search_model(inputs_embeds=embeds).logits, dim=2)

        original_prob = token_probs[0,target_token_index, target_token]

        top_probabilities, top_candidates = torch.topk(token_probs[:,target_token_index,:], 6, dim=1)

        for cand, prob in zip(top_candidates[0], top_probabilities[0]):

            ### addition to get rid of unused tokens
            decoded = search_tokenizer.decode([cand])
         
            if '[unused' in decoded or '[UNK]' in decoded:
                continue
            # Skip single punctuation tokens
            if len(decoded.strip()) == 1 and decoded.strip() in string.punctuation:
                continue


            ### addition to get rid of other tokens
            if not cand == target_token and cand.item() not in search_tokenizer.all_special_ids:
                if original_prob.item() == 1:
                    print("probability is one!")
                    replacements[(target_token_index, cand)] = prob.item()/(1-0.9)
                else:
                    replacements[(target_token_index, cand)] = prob.item()/(1-original_prob.item())

    
    #highest_scored_texts = max(candidate_scores.iteritems(), key=operator.itemgetter(1))[:100]
    replacement_keys = nlargest(50, replacements, key=replacements.get)
    replacements_new = dict()
    for rk in replacement_keys:
        replacements_new[rk] = replacements[rk]
    
    replacements = replacements_new
    #print("got highest scored single texts, will now collect doubles")

    highest_scored = nlargest(100, replacements, key=replacements.get)

    texts = []
    for single in highest_scored:
        alt = text_tokenized
        target_token_index, cand = single
        alt = torch.cat((alt[:,:target_token_index], torch.LongTensor([cand]).unsqueeze(0).to(device), alt[:,target_token_index+1:]), dim=1)
        alt_text = search_tokenizer.batch_decode(alt)[0]
        texts.append((alt_text, replacements[single]))
    return texts

def main(is_tofu=False,is_pii=False,unlearned=False):
    # filepath='data/tofu_answers.csv'
    # with open(filepath, 'r') as f:
    #     texts = [row[0] for row in csv.reader(f, delimiter='|')][1:]


    filepath = '/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/forget10.json'
    filepath = '/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/retain90.json'

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract 'answer' values from each object
    texts = [('[INST] ' + item['question'] + ' [/INST]'+item['answer']) for item in data if 'answer' in item][:2]
    print(texts)


    samples_nr = len(texts)
    model_name = 'roberta'
    #is_tofu= True
    dropout_percentage=0.35
    
    attack_tokenizer, attack_model = load_causal_lm_model(is_tofu=is_tofu,is_pii=is_pii,unlearned=unlearned)
    search_tokenizer,search_model = load_mlm_model(model_name=model_name)

    attack_model = attack_model.to(device)
    search_model = search_model.to(device)

    all_scores = []
    for text in tqdm(texts[:samples_nr],'Generate neighbours, Save loss..'):
        attack_model.eval()
        search_model.eval()

        tok_orig = search_tokenizer(text, padding = True, truncation = True, max_length = 512, return_tensors='pt').input_ids.to(device)
        orig_dec = search_tokenizer.batch_decode(tok_orig)[0].replace(" [SEP]", " ").replace("[CLS] ", " ").replace("<s>"," ").replace("</s>"," ")
        scores = dict()
        scores[f'<original_text>: {orig_dec}'] = get_logprob(orig_dec,attack_tokenizer,attack_model)

        with torch.no_grad():
            start = time.time()
            text_neighbours = generate_neighbours_alt(text,search_model,model_name,dropout_p=dropout_percentage)
            end = time.time()
            #print("generating neighbours took seconds:", end-start)

            for i, neighbours in enumerate([text_neighbours]):
                neighbours_texts = []
                for n in neighbours:

                    clean_text = n[0].replace(" [SEP]", " ").replace("[CLS] ", " ")
                    neighbours_texts.append((clean_text, n[1]))
                    #print(neighbours_texts)
                    score = get_logprob_batch([clean_text], attack_tokenizer, attack_model)
                    scores[(clean_text, n[1])] = score  # Use a tuple as the key for scores dictionary


            if i==0:
                scores_temp = scores        
        all_scores.append(scores_temp)

    with open(f'data/INST_neighbouhrhoomia|retain90|ftmodelLlam27b|search{model_name}|istofu{is_tofu}|ispii{is_pii}|unlearned_{unlearned}|dropout_percentage{dropout_percentage}.pkl', 'wb') as file:
        pickle.dump(all_scores, file)
        

if __name__ == "__main__":
    main(is_pii=True)
    # main(is_tofu=True)
    # main()
    # main(is_pii=True,unlearned=True)