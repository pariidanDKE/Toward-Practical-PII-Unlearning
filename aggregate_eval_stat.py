from omegaconf import OmegaConf
import hydra 
import json 
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 
import pandas as pd


def add_pii_autocompletion_leakage_results(eval_result_dict, eval_task_dict, pii_leakage_result, index):
    """Modified to save all similarity methods and match load_eval_results structure"""
    k = index
    
    # Define similarity methods
    similarity_methods = ['exact', 'partial_ratio', 'token_set_ratio']
    
    for similarity_method in similarity_methods:
        score_key = f'avg_pii_autocompletion_{similarity_method}_leakage_score'
        
        if score_key in eval_result_dict[k]:
            task_name = eval_task_dict[k]
            
            # Match the exact_retain_score and exact_forget_score structure from load_eval_results
            if 'Forget' in task_name:
                pii_leakage_result[f'{similarity_method}_forget_score'] = eval_result_dict[k][score_key]
            elif 'Retain' in task_name:
                pii_leakage_result[f'{similarity_method}_retain_score'] = eval_result_dict[k][score_key]

    return pii_leakage_result


def add_pii_extraction_leakage_results(pii_leakage_result, eval_result_dict, index):
    """Modified to save all similarity methods and match load_eval_results structure"""
    print('In add_pii_extraction_leakage_results')
    k = index
    
    similarity_methods = ['exact', 'partial_ratio', 'token_set_ratio']

    # Handle regular extraction results
    if 'pii_extraction_results' in eval_result_dict[k]:
        extraction_results = eval_result_dict[k]['pii_extraction_results']
        
        for similarity_method in similarity_methods:
            extract_leaked_forget = 0
            extract_leaked_retain = 0
            
            for item in extraction_results:
                forget_key = f'num_leaked_this_sample_{similarity_method}_forget'
                retain_key = f'num_leaked_this_sample_{similarity_method}_test_retain'
                
                if forget_key in item:
                    extract_leaked_forget += item[forget_key]
                if retain_key in item:
                    extract_leaked_retain += item[retain_key]
            
            # Calculate rates (you may need to adjust these counts based on your dataset)
            default_forget_count = len(extraction_results)
            default_retain_count = len(extraction_results)
            
            pii_leakage_result[f'extraction_scores_forget_{similarity_method}'] = extract_leaked_forget / default_forget_count
            pii_leakage_result[f'extraction_scores_retain_{similarity_method}'] = extract_leaked_retain / default_retain_count

    # Handle targeted extraction results
    if 'targeted_pii_extraction_results' in eval_result_dict[k]:
        targeted_results = eval_result_dict[k]['targeted_pii_extraction_results']
        
        for similarity_method in similarity_methods:
            targeted_leaked_forget = 0
            targeted_leaked_retain = 0
            
            for item in targeted_results:
                forget_key = f'num_leaked_this_sample_{similarity_method}_forget'
                retain_key = f'num_leaked_this_sample_{similarity_method}_test_retain'
                
                if forget_key in item:
                    targeted_leaked_forget += item[forget_key]
                if retain_key in item:
                    targeted_leaked_retain += item[retain_key]
            
            targeted_forget_count = len(targeted_results)
            targeted_retain_count = len(targeted_results)
            
            pii_leakage_result[f'targetted_extraction_scores_forget_{similarity_method}'] = targeted_leaked_forget / targeted_forget_count
            pii_leakage_result[f'targetted_extraction_scores_retain_{similarity_method}'] = targeted_leaked_retain / targeted_retain_count


    return pii_leakage_result


def add_pii_one_hop_leakage_results(pii_leakage_result, eval_result_dict, index):
    """Modified to match load_eval_results structure for one-hop scores"""
    print('In add_pii_one_hop_leakage_results')
    k = index


    if 'one_hop_forget_additional_pii_leakage_rate' in eval_result_dict[k]:
        pii_leakage_result['one_hop_scores_forget_pii'] = eval_result_dict[k]['one_hop_forget_additional_pii_leakage_rate']
    
    if 'one_hop_test_retain_additional_pii_leakage_rate' in eval_result_dict[k]:
        pii_leakage_result['one_hop_scores_retain_pii'] = eval_result_dict[k]['one_hop_test_retain_additional_pii_leakage_rate']
    
    # Name leakage scores (these typically don't vary by similarity method)
    if 'one_hop_forget_full_name_leakage_rate' in eval_result_dict[k]:
        pii_leakage_result['one_hop_scores_forget_name'] = eval_result_dict[k]['one_hop_forget_full_name_leakage_rate']
    
    if 'one_hop_test_retain_full_name_leakage_rate' in eval_result_dict[k]:
        pii_leakage_result['one_hop_scores_retain_name'] = eval_result_dict[k]['one_hop_test_retain_full_name_leakage_rate']

    return pii_leakage_result


def add_pii_inverse_qa_leakage_results(pii_leakage_result, eval_result_dict, index_forget, index_retain):
    """Modified to save all similarity methods and match load_eval_results structure"""
    print('In add_pii_inverse_qa_leakage_results')
    
    similarity_methods = ['exact', 'partial_ratio', 'token_set_ratio']
    
    # Process both forget and retain sets
    datasets = [
        (index_forget, 'forget'),
        (index_retain, 'retain')
    ]
    
    for k, dataset_type in datasets:
        if k not in eval_result_dict:
            print(f"Warning: Index {k} not found in eval_result_dict for {dataset_type} dataset")
            continue
        
        for similarity_method in similarity_methods:
            score_key = f'avg_pii_autocompletion_full_name_{similarity_method}_score'
            
            if score_key in eval_result_dict[k]:
                # Match the inverse_scores structure from load_eval_results
                pii_leakage_result[f'inverse_scores_{dataset_type}_{similarity_method}'] = eval_result_dict[k][score_key]
                
    return pii_leakage_result


def aggregate_paraphrase_scores(eval_result_dict, task_prefix, aggregated_results, eval_task_dict):
    """
    Modified to handle paraphrase scores for all similarity methods matching load_eval_results structure
    """
    paraphrase_tasks = []
    for i in range(1, 6):  # 1 to 5
        task_name = f"{task_prefix}_{i}.json"
        if task_name in eval_result_dict:
            paraphrase_tasks.append(task_name)
    
    if not paraphrase_tasks:
        return
    
    # Determine the aggregated task name and type
    if 'forget' in task_prefix:
        aggregated_task_name = 'Forget Paraphrase'
        score_type = 'forget'
    else:
        aggregated_task_name = 'Retain Paraphrase'
        score_type = 'retain'
    
    # Handle paraphrase scores for all similarity methods
    similarity_methods = ['exact', 'partial_ratio', 'token_set_ratio']
    
    for similarity_method in similarity_methods:
        paraphrase_pii_scores = []
        score_key = f'avg_pii_autocompletion_{similarity_method}_leakage_score'
        
        for task in paraphrase_tasks:
            if score_key in eval_result_dict[task]:
                paraphrase_pii_scores.append(eval_result_dict[task][score_key])
        
        # Calculate mean paraphrase score (matching get_exact_forgetretain logic)
        if paraphrase_pii_scores:
            paraphrase_scores_mean = sum(paraphrase_pii_scores) / len(paraphrase_pii_scores)
            aggregated_results[f'para_scores_{score_type}_{similarity_method}'] = paraphrase_scores_mean
    
    # ... rest of the existing aggregation logic for other metrics ...
    
    # Initialize aggregation variables
    all_gt_losses = []
    all_rougeL_recalls = []
    all_paraphrased_losses = []
    all_perturbed_losses = []
    all_fluency_scores = []
    all_f1_scores = []
    
    # Collect data from all paraphrase tasks
    for task in paraphrase_tasks:
        if 'avg_gt_loss' in eval_result_dict[task]:
            all_gt_losses.extend(list(eval_result_dict[task]['avg_gt_loss'].values()))
        
        if 'rougeL_recall' in eval_result_dict[task]:
            all_rougeL_recalls.extend(list(eval_result_dict[task]['rougeL_recall'].values()))
            
        if 'avg_paraphrased_loss' in eval_result_dict[task]:
            all_paraphrased_losses.extend(list(eval_result_dict[task]['avg_paraphrased_loss'].values()))
            
        if 'average_perturb_loss' in eval_result_dict[task]:
            perturb_values = eval_result_dict[task]['average_perturb_loss'].values()
            for val in perturb_values:
                if isinstance(val, list):
                    all_perturbed_losses.extend(val)
                else:
                    all_perturbed_losses.append(val)
        
        if 'fluency' in eval_result_dict[task]:
            all_fluency_scores.append(eval_result_dict[task]['fluency'])
            
        if 'f1' in eval_result_dict[task]:
            all_f1_scores.extend(list(eval_result_dict[task]['f1'].values()))
    
    # Calculate aggregated metrics (existing logic)
    if all_gt_losses:
        gt_probs = np.exp(-1 * np.array(all_gt_losses))
        avg_gt_prob = np.mean(gt_probs)
        aggregated_results[f'Prob. {aggregated_task_name}'] = avg_gt_prob
    
    if all_rougeL_recalls:
        avg_rouge = np.mean(all_rougeL_recalls)
        aggregated_results[f'ROUGE {aggregated_task_name}'] = avg_rouge
    
    if all_fluency_scores:
        avg_fluency = np.mean(all_fluency_scores)
        aggregated_results[f'Fluency {aggregated_task_name}'] = avg_fluency
    
    if all_f1_scores:
        avg_f1 = np.mean(all_f1_scores)
        aggregated_results[f'F1 {aggregated_task_name}'] = avg_f1
    
    # Truth Ratio calculation (existing logic)
    if all_paraphrased_losses and all_perturbed_losses:
        avg_paraphrase_np = np.array(all_paraphrased_losses)
        avg_perturbed_np = np.array(all_perturbed_losses)
        
        min_len = min(len(avg_paraphrase_np), len(avg_perturbed_np))
        avg_paraphrase_np = avg_paraphrase_np[:min_len]
        avg_perturbed_np = avg_perturbed_np[:min_len]
        
        curr_stat_1 = np.exp(avg_perturbed_np - avg_paraphrase_np)
        
        if 'forget' in task_prefix:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
        
        aggregated_results[f'Truth Ratio {aggregated_task_name}'] = paraphrased_perturb_ratio
        

def get_model_utility(eval_result_dict):
    print(eval_result_dict.keys())
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log_retain.json': 'Retain',
        'eval_log_forget.json': 'Forget',
    }

    # Check for paraphrase tasks (numbered 1-5) instead of single rephrase tasks
    forget_paraphrase_found = any(f"eval_log_forget_paraphrase_{i}.json" in eval_result_dict.keys() for i in range(1, 6))
    retain_paraphrase_found = any(f"eval_log_retain_paraphrase_{i}.json" in eval_result_dict.keys() for i in range(1, 6))
    
    if forget_paraphrase_found and retain_paraphrase_found:
        eval_task_dict['aggregated_forget_paraphrase'] = 'Forget Paraphrase'
        eval_task_dict['aggregated_retain_paraphrase'] = 'Retain Paraphrase'
    
    # Keep other task checks as they were
    if 'eval_log_forget_inverse.json' in eval_result_dict.keys():
        eval_task_dict['eval_log_forget_inverse.json'] = 'Forget Inverse'
        eval_task_dict['eval_log_retain_inverse.json'] = 'Retain Inverse'
    # if 'eval_log_forget_onehop.json' in eval_result_dict.keys():
    #     eval_task_dict['eval_log_forget_onehop.json'] = 'Forget Onehop'
    #     eval_task_dict['eval_log_retain_onehop.json'] = 'Retain Onehop'
    if 'extraction_attack.json' in eval_result_dict.keys():
        eval_task_dict['eval_log_forget_extraction_attack.json'] = 'Forget Extraction'
        eval_task_dict['eval_log_retain_extraction_attack.json'] = 'Retain Extraction'
        
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Prob.', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        if eval_task not in ['extraction_attack','one_hop_attack']:
            for metric in metrics:
                # Don't initialize with empty lists - we'll populate with actual values
                pass
    
    fluency_result = {}
    f1_result = {}
    pii_leakage_result = {}  # New dictionary for PII leakage metrics
    
    # First, handle aggregated paraphrase tasks
    if forget_paraphrase_found:
        aggregate_paraphrase_scores(eval_result_dict, 'eval_log_forget_paraphrase', output_result, eval_task_dict)
        aggregate_paraphrase_scores(eval_result_dict, 'eval_log_forget_paraphrase', fluency_result, eval_task_dict)
        aggregate_paraphrase_scores(eval_result_dict, 'eval_log_forget_paraphrase', f1_result, eval_task_dict)
    
    if retain_paraphrase_found:
        aggregate_paraphrase_scores(eval_result_dict, 'eval_log_retain_paraphrase', output_result, eval_task_dict)
        aggregate_paraphrase_scores(eval_result_dict, 'eval_log_retain_paraphrase', fluency_result, eval_task_dict)
        aggregate_paraphrase_scores(eval_result_dict, 'eval_log_retain_paraphrase', f1_result, eval_task_dict)
    
    # Process individual tasks (excluding paraphrase tasks as they're aggregated)
    for k, v in eval_result_dict.items():
        excluded_tasks = ['extraction_attack', 'one_hop_attack', 'eval_log_forget_inverse', 'eval_log_retain_inverse']
        if ('paraphrase_' in k and any(f'paraphrase_{i}.json' in k for i in range(1, 6))) or k.replace('.json', '') in excluded_tasks:
            continue
        
        print(f'Task name: {k}')

        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        elif 'average_perturb_loss' in eval_result_dict[k].keys():
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        else:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)


            
        output_result[f'Prob. {eval_task_dict[k]}'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'ROUGE {eval_task_dict[k]}'] = avg_rouge
        
        # getting Fluency and Bleu
        if 'fluency' in eval_result_dict[k].keys():
            fluency_result[f'Fluency {eval_task_dict[k]}'] = eval_result_dict[k]['fluency']
        
        if 'f1' in eval_result_dict[k].keys():
            avg_f1 = np.array(list(eval_result_dict[k]['f1'].values())).mean()
            output_result[f'F1 {eval_task_dict[k]}'] = avg_f1
        
        # getting Truth Ratio
        if 'avg_paraphrased_loss' in eval_result_dict[k].keys():
            avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))
            avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
            avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)
            curr_stat_1 =  np.exp(avg_perturbed_np_values - avg_paraphrase_np_values)
            if 'forget' in k:
                paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
            else:
                paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
            output_result[f'Truth Ratio {eval_task_dict[k]}'] = paraphrased_perturb_ratio
        else:
            if f'Truth Ratio {eval_task_dict[k]}' in output_result:
                output_result.pop(f'Truth Ratio {eval_task_dict[k]}')
        
        ### Add Jailbreaking Evals
        if k not in eval_task_dict:
            continue
        pii_leakage_result = add_pii_autocompletion_leakage_results(eval_result_dict, eval_task_dict, pii_leakage_result, index=k)

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k and 'Rephrase' not in k and 'Paraphrase' not in k:
            # Only include valid numeric values (not lists or other types)
            if isinstance(v, (int, float)) and not isinstance(v, list):
                model_utility_cands.append(v)
            elif isinstance(v, list) and len(v) == 0:
                # Skip empty lists that were initialized but never populated
                continue
            else:
                print(f"Warning: Skipping non-numeric value for {k}: {v} (type: {type(v)})")
    
    if 'extraction_attack.json' in eval_result_dict:
        print('Logging Extraction Attack')
        pii_leakage_result = add_pii_extraction_leakage_results(pii_leakage_result, eval_result_dict, index='extraction_attack.json')
        
    if 'one_hop_attack.json' in eval_result_dict:
        print('Logging One-Hop Attack')
        pii_leakage_result = add_pii_one_hop_leakage_results(pii_leakage_result, eval_result_dict, index='one_hop_attack.json')
    
    # Handle inverse QA task if present
    if 'eval_log_forget_inverse.json' in eval_result_dict:
        print('Logging Inverse QA')
        pii_leakage_result = add_pii_inverse_qa_leakage_results(pii_leakage_result, eval_result_dict, index_forget='eval_log_forget_inverse.json',index_retain='eval_log_retain_inverse.json')

    # Only calculate harmonic mean if we have valid numeric values
    if model_utility_cands:
        print(f"Calculating harmonic mean for {len(model_utility_cands)} utility metrics: {model_utility_cands}")
        output_result['Model Utility'] = hmean(model_utility_cands)
    else:
        print("Warning: No valid utility candidates found for harmonic mean calculation")
        output_result['Model Utility'] = 0.0
    
    return output_result, fluency_result, f1_result, pii_leakage_result
import glob
import os
def remove_model_tensors(ckpt_result):
    model_path = ckpt_result.replace('/eval_results/eval_log_aggregated.json','')
    
    # Find all .safetensors files in the model path
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    
    if not safetensors_files:
        print(f"No .safetensors files found in {model_path}")
        return
    
    print(f"Found {len(safetensors_files)} .safetensors files to remove from {model_path}")
    
    removed_count = 0
    for file_path in safetensors_files:
        try:
            file_size = os.path.getsize(file_path) / (1024**3)  # Size in GB
            os.remove(file_path)
            removed_count += 1
            print(f"Removed: {os.path.basename(file_path)} ({file_size:.2f} GB)")
        except OSError as e:
            print(f"Error removing {file_path}: {e}")
    
    print(f"Successfully removed {removed_count}/{len(safetensors_files)} .safetensors files")



@hydra.main(version_base=None, config_path="config", config_name="aggregate_eval_stat")
def main(cfg):

    if cfg.ckpt_result is None:
        raise ValueError("Please provide either retain_result or ckpt_result")
    
    ckpt_result = json.load(open(cfg.ckpt_result))

    print(f"Aggregating evaluation results from {cfg.ckpt_result}")
    model_utility, fluency_result, f1_result, pii_leakage_result = get_model_utility(ckpt_result)
    model_utility.update(fluency_result)
    if len(f1_result) != 0:
        model_utility.update(f1_result)
    if len(pii_leakage_result) != 0:
        model_utility.update(pii_leakage_result)

    model_utility['Method'] = cfg.method_name
    model_utility['Submitted By'] = cfg.submitted_by
    
    # dump the model utility to a csv
    with open(cfg.save_file, 'w') as f:
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)
        
    df = pd.read_csv(cfg.save_file)
    df = df.applymap(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
    df.to_excel(cfg.excel_file_path, index=False)
    
    if cfg.remove_model_tensors:
        print(f"Removing model tensors from {cfg.ckpt_result}")
        remove_model_tensors(cfg.ckpt_result)
    return model_utility
    
if __name__ == "__main__":
    main()