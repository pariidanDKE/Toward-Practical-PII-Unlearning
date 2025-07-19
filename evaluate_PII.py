# Standard library imports
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
import hydra
import numpy as np
import torch
from eval_utils import (
    eval_rouge_recall, load_model_with_retry,
    convert_to_left_padding, reinitialize_weights
)
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# Local imports
from data_module import CommonDataset, custom_data_collator_with_indices, get_batch_loss

from logging_utils import init_config
from pii_attacks.pii_attack_orchestrator import (
    DEFAULT_PII_DATA_PATH, 
    PII_AUTOCOMPLETE_EVAL_TASKS, 
    run_pii_jailbreaking_one_hop,
    run_pii_jailbreaking_autocompletion,
    run_pii_jailbreaking_extraction,
    intialize_util_methods
)
from utils import get_model_identifiers_from_yaml, load_person_split_dict

# ========================= DATA LOADING FUNCTIONS =========================

def get_dataloader(cfg, forget_loss, tokenizer, folder, split, question_key, answer_key, 
                  base_answer_key=None, perturbed_answer_key=None):
    """Create data loaders for evaluation."""
    folder = os.path.join(folder, f"{split}.json")
    
    # Main dataset
    torch_format_dataset = CommonDataset( 
        cfg.dataset, folder, tokenizer=tokenizer, model_family=cfg.model_family, 
        question_key=question_key, answer_key=answer_key, forget_loss=forget_loss, split=split
    )

    # Base dataset for comparison
    base_eval_dataloader = None
    if base_answer_key is not None:
        base_torch_format_dataset = CommonDataset(
            cfg.dataset, folder, tokenizer=tokenizer, model_family=cfg.model_family, 
            question_key=question_key, answer_key=base_answer_key, forget_loss=forget_loss, split=split
        )

        config_batch = cfg.batch_size // 4 if cfg.batch_size > 4 else cfg.batch_size
        
        if cfg.ds_size:
            base_torch_format_dataset.data = {
                key: base_torch_format_dataset.data[key] 
                for key in range(min(cfg.ds_size, len(base_torch_format_dataset.data)))
            }
            
        base_eval_dataloader = torch.utils.data.DataLoader(
            base_torch_format_dataset, batch_size=config_batch, 
            collate_fn=custom_data_collator_with_indices
        )

    # Perturbed dataset
    perturb_dataloader = None
    if perturbed_answer_key is not None:
        perturb_torch_format_dataset = CommonDataset(
            cfg.dataset, folder, tokenizer=tokenizer, model_family=cfg.model_family, 
            question_key=question_key, answer_key=perturbed_answer_key, forget_loss=forget_loss, split=split
        )
        
        config_batch = cfg.batch_size // 4 if cfg.batch_size > 4 else cfg.batch_size

        if cfg.ds_size:
            perturb_torch_format_dataset.data = {
                key: perturb_torch_format_dataset.data[key] 
                for key in range(min(cfg.ds_size, len(perturb_torch_format_dataset.data)))
            }
            
        perturb_dataloader = torch.utils.data.DataLoader(
            perturb_torch_format_dataset, batch_size=config_batch, 
            collate_fn=custom_data_collator_with_indices
        )

    # Main evaluation dataset
    if cfg.ds_size:
        torch_format_dataset.data = {
            key: torch_format_dataset.data[key] 
            for key in range(min(cfg.ds_size, len(torch_format_dataset.data)))
        }

    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, 
        collate_fn=custom_data_collator_with_indices
    )    

    return eval_dataloader, base_eval_dataloader, perturb_dataloader


# ========================= EVALUATION FUNCTIONS =========================

def process_evaluation_batch(batch, model, cfg, tokenizer, model_cfg):
    """Process a single evaluation batch."""
    input_ids, labels, attention_mask, indices = batch

    # Handle Qwen padding conversion
    if 'Qwen' in model_cfg['hf_key']:
        print("Converting to left padding for Qwen2.5...")
        input_ids, attention_mask, labels = convert_to_left_padding(
            input_ids, attention_mask, labels, tokenizer.pad_token_id
        )

    batch_dict = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    # Send to device
    for k, v in batch_dict.items():
        batch_dict[k] = v.to(model.device)

    with torch.no_grad():
        outputs = model(**batch_dict)
        input_string, gen_output, gt = run_generation(cfg, batch_dict, model, tokenizer=tokenizer, model_cfg=model_cfg)

    # Calculate loss metrics
    gt_loss = get_batch_loss(outputs.logits, batch_dict['labels'])
    num_token_gt = (batch_dict['labels'] != -100).sum(-1)
    gt_loss_per_token = gt_loss / num_token_gt

    return {
        'indices': indices,
        'gt_loss_per_token': gt_loss_per_token,
        'gt_loss': gt_loss,
        'num_token_gt': num_token_gt,
        'input_string': input_string,
        'gen_output': gen_output,
        'ground_truth': gt
    }


def update_evaluation_logs(eval_logs, batch_results):
    """Update evaluation logs with batch results."""
    # Initialize evaluation log structures
    for key in ['avg_gt_loss', 'gt_loss', 'num_token_gt', 'generated_text']:
        if key not in eval_logs:
            eval_logs[key] = {}

    indices_list = batch_results['indices'].cpu().numpy().tolist()
    
    eval_logs['avg_gt_loss'].update(dict(zip(indices_list, batch_results['gt_loss_per_token'].cpu().numpy().tolist())))
    eval_logs['gt_loss'].update(dict(zip(indices_list, batch_results['gt_loss'].cpu().numpy().tolist())))
    eval_logs['num_token_gt'].update(dict(zip(indices_list, batch_results['num_token_gt'].cpu().numpy().tolist())))
    eval_logs['generated_text'].update(dict(zip(indices_list, zip(batch_results['input_string'], batch_results['gen_output'], batch_results['ground_truth']))))


def get_all_evals(cfg, model, tokenizer, eval_task: str, eval_dataloader, 
                 base_eval_dataloader, perturb_dataloader, normalize_gt,
                 full_pii_data_for_jailbreak=None, model_cfg=None):
    """Main evaluation function with improved PII attack orchestration."""
    eval_logs = {}
    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    # Process evaluation batches
    for batch in tqdm(eval_dataloader):
        batch_results = process_evaluation_batch(batch, model, cfg, tokenizer, model_cfg)
        
        # Accumulate results
        all_indices.extend(batch_results['indices'].cpu().numpy().tolist())
        gen_outputs.extend(batch_results['gen_output'])
        ground_truths.extend(batch_results['ground_truth'])
        input_strings.extend(batch_results['input_string'])
        
        # Update logs
        update_evaluation_logs(eval_logs, batch_results)

    # Calculate ROUGE scores
    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    
    # Handle perturbation ratio evaluation
    if normalize_gt and base_eval_dataloader is not None and 'Qwen' not in model_cfg['hf_key']:
        eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model, tokenizer, model_cfg))
        
        # Calculate normalized ground truth loss
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

    # Run PII jailbreaking attacks if applicable
    print(f"Jailbreaking evaluation: {eval_task}...")
    if (cfg.get("dataset") == "PII" and 
        full_pii_data_for_jailbreak is not None and 
        eval_task in PII_AUTOCOMPLETE_EVAL_TASKS):
        
        run_pii_jailbreaking_autocompletion(
            cfg, model, tokenizer, eval_task, eval_logs, 
            input_strings, gen_outputs, all_indices, 
            full_pii_data_for_jailbreak, model_cfg
        )
    
    return eval_logs


def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model, tokenizer, model_cfg):
    """Evaluate perturbation ratio between ground truth and perturbed data."""
    eval_logs = {}
    
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices = batch
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        
        # Handle Qwen padding conversion
        if 'Qwen' in model_cfg['hf_key']:
            print("Converting to left padding for Qwen2.5..")
            input_ids, attention_mask, labels = convert_to_left_padding(input_ids, attention_mask, labels, tokenizer.pad_token_id)
            print(f'Perturbed Input IDs Shape: {perturb_input_ids.shape}')
            perturb_input_ids, perturb_attention_mask, perturb_labels = convert_to_left_padding(perturb_input_ids, perturb_attention_mask, perturb_labels, tokenizer.pad_token_id)

        batch_dict = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    
        # Reshape perturbed batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
            
        perturb_batch_dict = {
            "input_ids": perturb_input_ids.view(bsz*seq_len, -1), 
            "labels": perturb_labels.view(bsz*seq_len, -1), 
            "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)
        }

        # Send to device
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(model.device)
        for k, v in perturb_batch_dict.items():
            perturb_batch_dict[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch_dict)
            perturb_outputs = model(**perturb_batch_dict)

        # Calculate losses and metrics
        gt_loss = get_batch_loss(outputs.logits, batch_dict['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch_dict['labels']).view(bsz, seq_len)

        num_token_gt = (batch_dict['labels'] != -100).sum(-1)
        num_token_perturb = (perturb_batch_dict['labels'] != -100).view(bsz, seq_len, -1).sum(-1)
        mean_perturb_loss = perturb_loss.mean(dim=1)
        ratio = (mean_perturb_loss - gt_loss).mean()
        perturb_loss_per_token = perturb_loss / num_token_perturb
        gt_loss_per_token = gt_loss / num_token_gt
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))

        # Convert to dictionaries
        result_dicts = {
            'perturb_loss_per_token': dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist())),
            'gt_loss_per_token': dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())),
            'truth_ratio': dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist())),
            'gt_loss': dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())),
            'perturb_loss': dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist())),
            'num_token_gt': dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())),
            'num_token_perturb': dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))
        }

        # Initialize and update evaluation logs
        log_keys = [
            'average_perturb_loss', 'avg_paraphrased_loss', 'truth_ratio', 
            'paraphrased_loss', 'perturb_loss', 'num_token_paraphrased', 'num_token_perturb'
        ]
        
        for key in log_keys:
            if key not in eval_logs:
                eval_logs[key] = {}

        eval_logs['average_perturb_loss'].update(result_dicts['perturb_loss_per_token'])
        eval_logs['avg_paraphrased_loss'].update(result_dicts['gt_loss_per_token'])
        eval_logs['truth_ratio'].update(result_dicts['truth_ratio'])
        eval_logs['paraphrased_loss'].update(result_dicts['gt_loss'])
        eval_logs['perturb_loss'].update(result_dicts['perturb_loss'])
        eval_logs['num_token_paraphrased'].update(result_dicts['num_token_gt'])
        eval_logs['num_token_perturb'].update(result_dicts['num_token_perturb'])

    return eval_logs


# ========================= TEXT GENERATION FUNCTIONS =========================

def prepare_generation_prompts(raw_decoded_strings, split_symbol, special_tokens_to_remove, model_cfg):
    """Prepare prompts for text generation."""
    final_prompts = []
    final_ground_truths = []

    for raw_s in raw_decoded_strings:
        first_occurrence_idx = raw_s.find(split_symbol)

        if first_occurrence_idx == -1:
            # No split symbol found
            string_to_clean = raw_s
            for st_token in special_tokens_to_remove:
                string_to_clean = string_to_clean.replace(st_token, "")
            final_prompts.append(string_to_clean.strip())
            final_ground_truths.append("")
        else:
            # Split symbol found - separate prompt and ground truth
            part_before_split = raw_s[:first_occurrence_idx]
            preserved_symbol_instance = raw_s[first_occurrence_idx : first_occurrence_idx + len(split_symbol)] 
            part_after_split = raw_s[first_occurrence_idx + len(split_symbol):]

            # Clean parts
            cleaned_part_before = part_before_split
            for st_token in special_tokens_to_remove:
                cleaned_part_before = cleaned_part_before.replace(st_token, "")
            
            cleaned_part_after = part_after_split
            for st_token in special_tokens_to_remove:
                cleaned_part_after = cleaned_part_after.replace(st_token, "")
            
            prompt_text = cleaned_part_before.strip()
            answer_tag = model_cfg.get('answer_tag')
            
            if isinstance(answer_tag, str) and answer_tag.strip() and answer_tag != '?':
                ground_truth_text = cleaned_part_after.replace(model_cfg.get('answer_tag'), '').strip()
            else:
                ground_truth_text = cleaned_part_after.strip()

            final_prompts.append(prompt_text + preserved_symbol_instance + model_cfg.get('answer_tag'))
            final_ground_truths.append(ground_truth_text)

    return final_prompts, final_ground_truths


def run_generation(cfg, batch, model, tokenizer, model_cfg=None):
    """Run text generation for evaluation."""
    # Determine split symbol
    calculated_split_symbol = getattr(tokenizer, 'eos_token', None)
    if calculated_split_symbol is None:
        calculated_split_symbol = "<|eot_id|>" 

    if hasattr(cfg, 'model_family') and (cfg.model_family == 'llama2-7b' or cfg.model_family == 'llama2-13b'):
         calculated_split_symbol = " [/INST]"

    # Use model_cfg split symbol if available
    if model_cfg is not None and model_cfg.get('question_end_tag'):
        split_symbol = model_cfg['question_end_tag']

    if model_cfg.get('question_end_tag_inference'):
        split_symbol = model_cfg['question_end_tag_inference']
    else:
        split_symbol = calculated_split_symbol

    # Decode input sequences
    input_ids = batch["input_ids"]
    raw_decoded_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
  
    # Prepare tokens for removal
    tokens_for_removal_candidates = list(set(tokenizer.all_special_tokens))

    if model_cfg is not None:
        model_template_tags = [model_cfg.get('answer_end_tag')]
        for tag_string in model_template_tags:
            if tag_string and isinstance(tag_string, str) and tag_string.strip() and tag_string.strip() != '?':  
                tokens_for_removal_candidates.append(tag_string)
        
        if split_symbol not in tokens_for_removal_candidates:
             tokens_for_removal_candidates.append(split_symbol)

    special_tokens_to_remove = sorted(list(set(tokens_for_removal_candidates)), key=len, reverse=True)
    special_tokens_to_remove = [token for token in special_tokens_to_remove if token]

    # Prepare prompts and ground truths
    final_prompts, final_ground_truths = prepare_generation_prompts(
        raw_decoded_strings, split_symbol, special_tokens_to_remove, model_cfg
    )

    # Handle inference-specific tag replacements
    if model_cfg.get('question_end_tag_inference'):
        final_prompts = [prompt.replace(model_cfg['question_end_tag_inference'], model_cfg['question_end_tag']) for prompt in final_prompts]
    if model_cfg.get('question_start_tag_inference'):
        final_prompts = [prompt.replace(model_cfg['question_start_tag_inference'], model_cfg['question_start_tag']) for prompt in final_prompts]

    input_strings = final_prompts
    ground_truth = final_ground_truths
    
    # Tokenize with left padding
    left_pad_tokenizer = tokenizer
    inputs = left_pad_tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors='pt', padding=True
    ).to(model.device)
    
    # Generate text
    do_sample = getattr(cfg.generation, 'do_sample', True) if hasattr(cfg, 'generation') else True
    
    out = model.generate(
        inputs.input_ids, 
        attention_mask=inputs.attention_mask, 
        max_length=cfg.generation.max_length, 
        max_new_tokens=cfg.generation.max_new_tokens, 
        do_sample=do_sample, 
        use_cache=True, 
        pad_token_id=left_pad_tokenizer.eos_token_id
    )
    
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    
    return input_strings, strs, ground_truth

# ========================= SETUP AND INITIALIZATION FUNCTIONS =========================

def setup_environment():
    """Setup environment variables and device configuration."""
    local_rank = 0
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device_map = {'': local_rank}
    
    os.environ["WANDB_DISABLED"] = "true"
    return local_rank, device_map


def setup_tokenizer(model_id):
    """Setup and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
    tokenizer.padding_side = 'left'
    tokenizer.padding_size = 'longest'

    print(f'Pad token id is {tokenizer.pad_token_id}, eos token id is {tokenizer.eos_token_id}')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


def get_pretrained_model_path(cfg, model_cfg):
    """Get the appropriate pretrained model path based on dataset."""
    dataset_paths = {
        "TOFU": "tofu_target_model_path",
        "PII": "pii_target_model_path",
        "Harry": "harry_target_model_path",
        "ZSRE": "zsre_target_model_path"
    }
            
    required_path_key = dataset_paths.get(cfg.dataset)
    if required_path_key is None:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    if required_path_key not in model_cfg:
        raise KeyError(f"Model config missing required path '{required_path_key}' for dataset '{cfg.dataset}'")

    return model_cfg[required_path_key]


def setup_model_config(cfg):
    """Setup model configuration and paths."""
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    pretained_traget_model_path = get_pretrained_model_path(cfg, model_cfg)
    
    # Setup torch dtype
    torch_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    
    return model_cfg, model_id, pretained_traget_model_path, torch_dtype


def load_pii_data(cfg):
    """Load PII data for jailbreaking attacks if applicable."""
    if cfg.dataset != 'PII':
        return None
        
    pii_json_path = DEFAULT_PII_DATA_PATH
    print(f"Loading PII data for attacks from {pii_json_path}...")
    
    if os.path.exists(pii_json_path):
        try:
            with open(pii_json_path, 'r', encoding='utf-8') as f:
                full_pii_data_for_jailbreak = json.load(f)
            print(f"Successfully loaded PII data for attacks from {pii_json_path}.")
            return full_pii_data_for_jailbreak
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {pii_json_path}. PII attacks might fail.")
    else:
        print(f"Warning: PII data file for attacks not found at {pii_json_path}. PII leakage attacks will be skipped.")
    
    return None


def process_evaluation_task(cfg, model, tokenizer, folder, split, question_key, answer_key, 
                          eval_task, base_answer_key, perturbed_answer_key, 
                          full_pii_data_for_jailbreak, model_cfg):
    """Process a single evaluation task."""
    normalize_gt = perturbed_answer_key is not None

    if eval_task == 'extraction_attack':
        return run_pii_jailbreaking_extraction(cfg, model, tokenizer, full_pii_data_for_jailbreak, model_cfg=model_cfg)
    elif eval_task == 'one_hop_attack':
        return run_pii_jailbreaking_one_hop(cfg, model, tokenizer, full_pii_data_for_jailbreak, model_cfg=model_cfg)
    else:
        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
            cfg, cfg.forget_loss, tokenizer, folder, split, question_key, answer_key, 
            base_answer_key, perturbed_answer_key
        )
        return get_all_evals(
            cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, 
            perturb_dataloader, normalize_gt=normalize_gt, 
            full_pii_data_for_jailbreak=full_pii_data_for_jailbreak, model_cfg=model_cfg
        )


def save_evaluation_results(cfg, eval_logs, eval_task, aggregated_eval_logs):
    """Save evaluation results to JSON files."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
    
    if world_size != 1:
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

    with open(save_filename, "w") as f:
        json.dump(eval_logs, f, indent=4)

    aggregated_eval_logs[f'{eval_task}.json'] = eval_logs


# ========================= MAIN FUNCTION =========================

@hydra.main(version_base=None, config_path="config/", config_name="eval_pii")
def main(cfg):
    """Main evaluation function."""
    # Validate configuration
    config_lists = [cfg.data_path, cfg.split_list, cfg.eval_task, cfg.question_key, 
                   cfg.answer_key, cfg.base_answer_key, cfg.perturbed_answer_key]
    assert all(len(lst) == len(config_lists[0]) for lst in config_lists), \
        "All configuration lists must have the same length"
    
    # Setup directories and configuration
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    init_config(cfg)
    
    # Initialize PII attack utilities
    global DEFAULT_PII_DATA_PATH
    DEFAULT_PII_DATA_PATH = cfg['pii_data_path']
    intialize_util_methods()
    
    # Setup environment and model configuration
    local_rank, device_map = setup_environment()
    model_cfg, model_id, pretained_traget_model_path, torch_dtype = setup_model_config(cfg)
    tokenizer = setup_tokenizer(model_id)
    
    # Setup model configuration for loading
    try:
        config = AutoConfig.from_pretrained(model_id)
    except ValueError as e:
        if "rope_scaling" in str(e):
            config_path = os.path.join(model_cfg["t440_config"])
            config = AutoConfig.from_pretrained(config_path)
        else:
            raise e

    # Load model with retry mechanism
    model = load_model_with_retry(cfg, model_cfg, config, torch_dtype, device_map,tokenizer, pretained_traget_model_path)
    
    if model is None:
        raise RuntimeError("Failed to load model after multiple attempts")
    
    model = model.eval()
    
    # Reinitialize weights if configured
    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)
    
    # Load PII data for attacks
    full_pii_data_for_jailbreak = load_pii_data(cfg)

    # Process all evaluation tasks
    aggregated_eval_logs = {}
    task_configs = zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, 
                      cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)
    
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(task_configs):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f'Working on eval task {eval_task} with split {split}')
        
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        if world_size != 1:
            save_filename = os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        # Process evaluation task
        eval_logs = process_evaluation_task(
            cfg, model, tokenizer, folder, split, question_key, answer_key, 
            eval_task, base_answer_key, perturbed_answer_key, 
            full_pii_data_for_jailbreak, model_cfg
        )
        
        # Save results
        save_evaluation_results(cfg, eval_logs, eval_task, aggregated_eval_logs)

    # Save aggregated results
    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")
    with open(aggregated_eval_log_filename, "w") as f:
        json.dump(aggregated_eval_logs, f, indent=4)


if __name__ == "__main__":
    main()