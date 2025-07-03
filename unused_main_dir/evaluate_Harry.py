from tqdm import tqdm
from data_module import CommonDataset, custom_data_collator, get_batch_loss, custom_data_collator_with_indices
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality
from evals.uld import ULDLLM
from evals.whos_harry_potter import WHPLLM
import torch.nn as nn
import csv 
import numpy as np 
import nltk
import scipy
from peft import PeftModel

def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}

        # send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)

        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)
        mean_perturb_loss = perturb_loss.mean(dim=1)
        ratio = (mean_perturb_loss - gt_loss).mean()
        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))

        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))

        # merge dicts
        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
        eval_logs['truth_ratio'].update(truth_ratio)
        eval_logs['paraphrased_loss'].update(gt_loss)
        eval_logs['perturb_loss'].update(perturb_loss)
        eval_logs['num_token_paraphrased'].update(num_token_gt)
        eval_logs['num_token_perturb'].update(num_token_perturb)

    return eval_logs

def get_dataloader(cfg, forget_loss, tokenizer, folder, split, question_key, answer_key, base_answer_key=None, perturbed_answer_key=None):
    folder = os.path.join(folder, f"{split}.json")
    torch_format_dataset = CommonDataset( 
        cfg.dataset,
        folder, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        question_key=question_key, 
        answer_key=answer_key,
        forget_loss=forget_loss,
        split=split
    ) 

    if base_answer_key is not None:
        base_torch_format_dataset = CommonDataset(
            cfg.dataset,
            folder,
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            question_key=question_key, 
            answer_key=base_answer_key,
            forget_loss=forget_loss,
            split=split
        )
        if cfg.ds_size:
            base_torch_format_dataset.data = {key: base_torch_format_dataset.data[key] for key in range(min(cfg.ds_size, len(base_torch_format_dataset.data)))}
        base_eval_dataloader = torch.utils.data.DataLoader(
            base_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
        )
    else:
        base_eval_dataloader = None

    if perturbed_answer_key is not None:
        perturb_torch_format_dataset = CommonDataset(
            cfg.dataset,
            folder,
            tokenizer=tokenizer, 
            model_family=cfg.model_family, 
            question_key=question_key, 
            answer_key=perturbed_answer_key,
            forget_loss=forget_loss,
            split=split
        )
        if cfg.ds_size:
            perturb_torch_format_dataset.data = {key: perturb_torch_format_dataset.data[key] for key in range(min(cfg.ds_size, len(perturb_torch_format_dataset.data)))}
        perturb_dataloader = torch.utils.data.DataLoader(
            perturb_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
        )
    else:
        perturb_dataloader = None

    if cfg.ds_size:
        torch_format_dataset.data = {key: torch_format_dataset.data[key] for key in range(min(cfg.ds_size, len(torch_format_dataset.data)))}

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    
    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt
        
        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output,gt))))

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))

    if normalize_gt:
        eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs

def compute_f1(gen_outputs, ground_truths, indices):
    f1_score = {}
    precision_score = {}
    recall_score = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        pred_tokens = gen.split()
        ref_tokens = gt.split()
        common_tokens = set(pred_tokens) & set(ref_tokens)
        tp = len(common_tokens)
        fp = len(pred_tokens) - tp
        fn = len(ref_tokens) - tp
        # Precision and Recall
        precision= tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        # F1 Score
        f1_score[idx] = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        precision_score[idx] = precision
        recall_score[idx] = recall
    return {'f1': f1_score, 'precision':precision_score, 'recall':recall_score}

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def compute_bleu(gen_outputs, ground_truths, indices):
    bleu_score = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        pred_tokens = gen.split()
        ref_tokens = [gt.split()]  
        smoothing_function = SmoothingFunction().method1  
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing_function)
        bleu_score[idx] = bleu
    return {'bleu': bleu_score}

def get_all_evals_forget(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        # send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        gen_output = []
        gt = []
        
        ori_inputs = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
        input_string = [s.split(split_symbol)[0] for s in ori_inputs]
        labels_string = [s.split(split_symbol)[1] for s in ori_inputs]
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
        for i in range(logits.size(0)):
            ans_start = len(tokenizer.encode(input_string[i]))
            ans_end = len(tokenizer.encode(ori_inputs[i]))
            if cfg.model_family == 'llama2-7b':
                ans_start = len(tokenizer.encode(input_string[i]+ ' [/INST]'))
                ans_end = len(tokenizer.encode(ori_inputs[i])) 
            answers = torch.argmax(logits[i], dim=-1).squeeze().detach().cpu().numpy().tolist()
            answers = answers[ans_start-1:ans_end-1]
            label = labels[i].squeeze().detach().cpu().numpy().tolist()
            label = label[ans_start:ans_end]
            ans_str = tokenizer.decode(answers)
            label_str = tokenizer.decode(label)
            gen_output.append(ans_str)
            gt.append(label_str)
        gen_outputs.extend(gen_output)
        ground_truths.extend(gt)
        input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt
        
        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output,gt))))

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    eval_logs.update(compute_f1(gen_outputs, ground_truths, all_indices))
    eval_logs.update(compute_bleu(gen_outputs, ground_truths, all_indices))

    return eval_logs

def relative_top_filter(
        self, scores: torch.FloatTensor, relative_top: float = 0.1, 
        filter_value: float = -float(1e3), min_tokens_to_keep: int = 1
    ) -> torch.FloatTensor:
        min_tokens_to_keep = int(relative_top * scores.shape[-1]) #! minimum number of tokens to keep
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        # scores_normalized[scores_normalized < probs_thresh] = filter_value
        mask = scores_normalized < probs_thresh
        return scores, mask, probs_thresh
    

@hydra.main(version_base=None, config_path="config", config_name='eval_harry')
def main(cfg):
    assert len(cfg.data_path)==len(cfg.split_list)==len(cfg.eval_task)==len(cfg.question_key)==len(cfg.answer_key)==len(cfg.base_answer_key)==len(cfg.perturbed_answer_key), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    
    local_rank = 0
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if cfg.dataset == "TOFU":
        pretained_traget_model_path = model_cfg["tofu_target_model_path"]
    elif cfg.dataset == "Harry":
        pretained_traget_model_path = model_cfg["harry_target_model_path"]
    elif cfg.dataset == "ZSRE":
        pretained_traget_model_path = model_cfg["zsre_target_model_path"]

    tokenizer.pad_token = tokenizer.eos_token
    
    if cfg.bf16 is True:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    model = None
    config = AutoConfig.from_pretrained(model_id)
    for attempt in range(3):
        try:
            if cfg.use_pretrained or "icl" in cfg.forget_loss:
                print(f"Loading pretrained model from {pretained_traget_model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    pretained_traget_model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                    torch_dtype=torch_dtype, trust_remote_code = True, 
                    device_map=device_map
                )
            elif "ULD" in cfg.forget_loss:
                print(f"Loading checkpoint from {cfg.model_path}")
                basemodel = AutoModelForCausalLM.from_pretrained(pretained_traget_model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map)
                assistant = AutoModelForCausalLM.from_pretrained(
                    cfg.model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map
                    )
                model = ULDLLM(
                    basellm=basemodel, 
                    assist_llm=assistant, 
                    weight=-0.8, 
                    top_logit_filter=0.1,
                    tokenizer=tokenizer
                )
            elif "WHP" in cfg.forget_loss:
                print(f"Loading checkpoint from {cfg.model_path}")
                basemodel = AutoModelForCausalLM.from_pretrained(
                    pretained_traget_model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map)
                reinforce_model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map)
                model = WHPLLM(
                    basellm = basemodel,
                    reinforced_llm = reinforce_model,
                    tokenizer=tokenizer
                )
            elif "task_vector" in cfg.forget_loss:
                print(f"Loading checkpoint from {cfg.model_path}")
                pretrained_model = AutoModelForCausalLM.from_pretrained(
                    pretained_traget_model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map)
                reinforce_model = AutoModelForCausalLM.from_pretrained(
                    cfg.model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map)
                pretrained_state_dict = pretrained_model.state_dict()
                reinforce_state_dict = reinforce_model.state_dict()
                with torch.no_grad():
                    task_vector = {}
                    for key in pretrained_state_dict:
                        if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                            continue
                        task_vector[key] = reinforce_state_dict[key] - pretrained_state_dict[key]
                    new_state_dict = {}
                    scaling_coef = 1.0
                    for key in pretrained_state_dict:
                        if key not in task_vector:
                            print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                            continue
                        new_state_dict[key] = pretrained_state_dict[key] - scaling_coef * task_vector[key]
                    model = AutoModelForCausalLM.from_pretrained(
                        cfg.model_path, config=config, \
                            state_dict=new_state_dict, device_map=device_map, \
                            use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
                            torch_dtype=torch_dtype, trust_remote_code = True)
                    model.load_state_dict(new_state_dict)
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
                    torch_dtype=torch_dtype, trust_remote_code = True, device_map=device_map)
        except Exception as e:
            print(e)
            continue
        else:
            break
    else:
        print("Error: could not load model")
    
    model = model.eval()
    
    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)
    
    # write custom eval loop using compute_metrics
    aggregated_eval_logs = {}
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f'Working on eval task {eval_task} with split {split}')
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, cfg.forget_loss, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)

        normalize_gt = False 
        if perturbed_answer_key is not None:
            normalize_gt = True
        if "forget" in eval_task or "retain" in eval_task:
            eval_logs = get_all_evals_forget(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)
        else:
            eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)
            
        
        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)
                    

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    # add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
    
    # now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id

    inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    # now generate
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, ground_truth

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)

def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)

    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall
    
    # add fluency score
    fluency = n_gram_entropy(gen_outputs)
    print("fluency",fluency)

    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall, 'fluency': fluency}

if __name__ == "__main__":
    main()

