from omegaconf import OmegaConf
import hydra 
import json 
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 
import pandas as pd

def add_pii_autocompletion_leakage_results(eval_result_dict,eval_task_dict,pii_leakage_result,index):
        k = index
        # Get PII leakage metrics if they exist
        if 'avg_pii_autocompletion_leakage_score' in eval_result_dict[k]:
            pii_leakage_result[f'PII_AutoLeakage_Score {eval_task_dict[k]}'] = eval_result_dict[k]['avg_pii_autocompletion_exact_leakage_score']
            
        # --- Autocompletion Partial Ratio Metrics ---
        if 'avg_pii_autocompletion_partial_ratio_leakage_score' in eval_result_dict[k]:
            pii_leakage_result[f'PII_AutoPartialRatioLeakage_Score {eval_task_dict[k]}'] = eval_result_dict[k]['avg_pii_autocompletion_partial_ratio_leakage_score']


        # --- Autocompletion Token Set Ratio Metrics ---
        if 'avg_pii_autocompletion_token_set_ratio_leakage_score' in eval_result_dict[k]:
            pii_leakage_result[f'PII_AutoTokenSetRatioLeakage_Score {eval_task_dict[k]}'] = eval_result_dict[k]['avg_pii_autocompletion_token_set_ratio_leakage_score']

        return pii_leakage_result

def add_pii_extraction_leakage_results(pii_leakage_result, eval_result_dict, index):
    # --- Extraction Exact Match Metrics (Original) ---
    print('In add_pii_extraction_leakage_results')
    k = index

    # if 'overall_pii_extraction_score' in eval_result_dict[k]:
    #     print('Adding Extraction SCORE!')
    #     pii_leakage_result[f'PII_Extraction_Score_Exact'] = eval_result_dict[k]['overall_pii_extraction_score']['extraction_score_exact']


    # # --- Extraction Partial Ratio Metrics (Original) ---
    # if 'overall_pii_extraction_score' in eval_result_dict[k]:
    #     if 'extraction_score_partial_ratio' in eval_result_dict[k]['overall_pii_extraction_score']:
    #         pii_leakage_result[f'PII_Extraction_Score_PartialRatio'] = eval_result_dict[k]['overall_pii_extraction_score']['extraction_score_partial_ratio']

    # # --- Extraction Token Set Ratio Metrics (Original) ---
    # if 'overall_pii_extraction_score' in eval_result_dict[k]:
    #     if 'extraction_score_token_set_ratio' in eval_result_dict[k]['overall_pii_extraction_score']:
    #         pii_leakage_result[f'PII_Extraction_Score_TokenSetRatio'] = eval_result_dict[k]['overall_pii_extraction_score']['extraction_score_token_set_ratio']

    # --- NEW: Split-based Extraction Metrics ---
    split_types = ['forget', 'retain', 'test_retain']
    metric_types = ['exact', 'partial_ratio', 'token_set_ratio']
    
    # for split_type in split_types:
    #     # Sample counts for each split
    #     #sample_count_key = f'pii_extraction_samples_count_{split_type}'
    #     #if sample_count_key in eval_result_dict[k]:
    #     #    pii_leakage_result[f'PII_Extraction_Samples_{split_type.title()}'] = eval_result_dict[k][sample_count_key]
        
    #     for metric_type in metric_types:
    #         # Average leaked items per prompt for each split
    #         #avg_items_key = f'avg_pii_extraction_leaked_items_per_prompt_{split_type}_{metric_type}'
            
    #         # Leakage rates for each split
    #         leakage_rate_key = f'pii_extraction_leakage_rate_{split_type}_{metric_type}'
    #         if leakage_rate_key in eval_result_dict[k]:
    #             pii_leakage_result[f'PII_Extraction_LeakageRate_{split_type.title()}_{metric_type.title().replace("_", "")}'] = eval_result_dict[k][leakage_rate_key]
    


    # Add split-based overall scores from the extraction attack's overall_extraction_score
    if 'overall_pii_extraction_score' in eval_result_dict[k]:
        extraction_scores = eval_result_dict[k]['overall_pii_extraction_score']
        
        for split_type in split_types:
            for metric_type in metric_types:
                score_key = f'{split_type}_extraction_score_{metric_type}'
                if score_key in extraction_scores:
                    pii_leakage_result[f'PII_Extraction_OverallScore_{split_type.title()}_{metric_type.title().replace("_", "")}'] = extraction_scores[score_key]


    # # ---
    # ## Targeted Extraction PII Metrics (Original)
    # if 'targeted_overall_pii_extraction_score' in eval_result_dict[k]:
    #     pii_leakage_result[f'PII_Targeted_Extraction_Score_Exact'] = eval_result_dict[k]['targeted_overall_pii_extraction_score']['extraction_score_exact']


    # # --- Targeted Extraction Partial Ratio Metrics ---
    # if 'targeted_overall_pii_extraction_score' in eval_result_dict[k]:
    #     if 'extraction_score_partial_ratio' in eval_result_dict[k]['targeted_overall_pii_extraction_score']:
    #         pii_leakage_result[f'PII_Targeted_Extraction_Score_PartialRatio'] = eval_result_dict[k]['targeted_overall_pii_extraction_score']['extraction_score_partial_ratio']


    # # --- Targeted Extraction Token Set Ratio Metrics ---
    # if 'targeted_overall_pii_extraction_score' in eval_result_dict[k]:
    #     if 'extraction_score_token_set_ratio' in eval_result_dict[k]['targeted_overall_pii_extraction_score']:
    #         pii_leakage_result[f'PII_Targeted_Extraction_Score_TokenSetRatio'] = eval_result_dict[k]['targeted_overall_pii_extraction_score']['extraction_score_token_set_ratio']


    # # --- NEW: Split-based Targeted Extraction Metrics ---
    # for split_type in split_types:
    #     # Targeted sample counts for each split
    #     targeted_sample_count_key = f'targeted_pii_extraction_samples_count_{split_type}'
    #     #if targeted_sample_count_key in eval_result_dict[k]:
    #     #    pii_leakage_result[f'PII_Targeted_Extraction_Samples_{split_type.title()}'] = eval_result_dict[k][targeted_sample_count_key]
        
    #     for metric_type in metric_types:
    #         # Targeted average leaked items per prompt for each split
            
    #         # Targeted leakage rates for each split
    #         targeted_leakage_rate_key = f'targeted_pii_extraction_leakage_rate_{split_type}_{metric_type}'
    #         if targeted_leakage_rate_key in eval_result_dict[k]:
    #             pii_leakage_result[f'PII_Targeted_Extraction_LeakageRate_{split_type.title()}_{metric_type.title().replace("_", "")}'] = eval_result_dict[k][targeted_leakage_rate_key]
    
    # Add targeted split-based overall scores
    if 'targeted_overall_pii_extraction_score' in eval_result_dict[k]:
        targeted_extraction_scores = eval_result_dict[k]['targeted_overall_pii_extraction_score']
        
        for split_type in split_types:
            for metric_type in metric_types:
                targeted_score_key = f'{split_type}_extraction_score_{metric_type}'
                if targeted_score_key in targeted_extraction_scores:
                    pii_leakage_result[f'PII_Targeted_Extraction_OverallScore_{split_type.title()}_{metric_type.title().replace("_", "")}'] = targeted_extraction_scores[targeted_score_key]

    return pii_leakage_result


def add_pii_one_hop_leakage_results(pii_leakage_result, eval_result_dict, index):
    """Add one-hop attack results to the final dictionary, following extraction attack pattern."""
    print('In add_pii_one_hop_leakage_results')
    k = index

    print('Keys in eval_result_dict:', eval_result_dict[k].keys())
    # --- One-Hop Overall Metrics ---
    # if 'overall_pii_one_hop_score' in eval_result_dict[k]:
        
    #     # Overall full name leakage rate
    #     one_hop_scores = eval_result_dict[k]['overall_pii_one_hop_score']
    #     if 'one_hop_full_name_leakage_rate' in one_hop_scores:
    #         pii_leakage_result[f'PII_OneHop_FullNameLeakageRate'] = one_hop_scores['one_hop_full_name_leakage_rate']
        
    #     # Total additional PII leaked
    #     if 'total_additional_pii_leaked' in one_hop_scores:
    #         pii_leakage_result[f'PII_OneHop_TotalAdditionalPII'] = one_hop_scores['total_additional_pii_leaked']

    # --- NEW: Split-based One-Hop Metrics (following extraction pattern) ---
    split_types = ['forget', 'retain', 'test_retain', 'unknown']
    
    for split_type in split_types:
        # Sample counts for each split
        #sample_count_key = f'one_hop_{split_type}_samples'
        #if sample_count_key in eval_result_dict[k]:
        #    pii_leakage_result[f'PII_OneHop_Samples_{split_type.title()}'] = eval_result_dict[k][sample_count_key]
        
        # Full name leakage rates for each split
        leakage_rate_key = f'one_hop_{split_type}_full_name_leakage_rate'
        if leakage_rate_key in eval_result_dict[k]:
            pii_leakage_result[f'PII_OneHop_FullNameLeakageRate_{split_type.title()}'] = eval_result_dict[k][leakage_rate_key]
        
        # ADD: Average additional PII per sample for each split
        avg_additional_pii_key = f'one_hop_{split_type}_avg_additional_pii_per_sample'
        if avg_additional_pii_key in eval_result_dict[k]:
            pii_leakage_result[f'PII_OneHop_AvgAdditionalPII_{split_type.title()}'] = eval_result_dict[k][avg_additional_pii_key]

   
    # Add split-based overall scores from the one-hop attack's overall_one_hop_score
    if 'overall_pii_one_hop_score' in eval_result_dict[k]:
        one_hop_scores = eval_result_dict[k]['overall_pii_one_hop_score']
        
        for split_type in split_types:
            # Split-specific full name leakage rates from overall scores
            score_key = f'one_hop_{split_type}_full_name_leakage_rate'
            if score_key in one_hop_scores:
                print(f'Score key : {score_key}')
                pii_leakage_result[f'PII_OneHop_OverallScore_{split_type.title()}_FullNameLeakage'] = one_hop_scores[score_key]
            
            # ADD: Split-specific average additional PII from overall scores (if you decide to add them there too)
            avg_pii_score_key = f'one_hop_{split_type}_avg_additional_pii_per_sample'
            if avg_pii_score_key in one_hop_scores:
                print(f'Avg PII Score key : {avg_pii_score_key}')
                pii_leakage_result[f'PII_OneHop_OverallScore_{split_type.title()}_AvgAdditionalPII'] = one_hop_scores[avg_pii_score_key]

    return pii_leakage_result


def add_pii_inverse_qa_leakage_results(pii_leakage_result, eval_result_dict, index_forget, index_retain):
    """Add inverse QA full_name leakage results to the final dictionary."""
    print('In add_pii_inverse_qa_leakage_results')
    
    # Process both forget and retain sets
    datasets = [
        (index_forget, 'Forget'),
        (index_retain, 'Retain')
    ]
    
    for k, dataset_name in datasets:
        if k not in eval_result_dict:
            print(f"Warning: Index {k} not found in eval_result_dict for {dataset_name} dataset")
            continue
            
        # --- Inverse QA Full Name Exact Match Metrics ---
        if 'avg_pii_autocompletion_full_name_exact_score' in eval_result_dict[k]:
            print(f'Adding Inverse QA Full Name Exact SCORE for {dataset_name}!')
            pii_leakage_result[f'PII_InverseQA_FullName_Score_{dataset_name}_Exact'] = eval_result_dict[k]['avg_pii_autocompletion_full_name_exact_score']
        
        # --- Inverse QA Full Name Partial Ratio Metrics ---
        if 'avg_pii_autocompletion_full_name_partial_ratio_score' in eval_result_dict[k]:
            pii_leakage_result[f'PII_InverseQA_FullName_Score_{dataset_name}_PartialRatio'] = eval_result_dict[k]['avg_pii_autocompletion_full_name_partial_ratio_score']
        
        # --- Inverse QA Full Name Token Set Ratio Metrics ---
        if 'avg_pii_autocompletion_full_name_token_set_ratio_score' in eval_result_dict[k]:
            pii_leakage_result[f'PII_InverseQA_FullName_Score_{dataset_name}_TokenSetRatio'] = eval_result_dict[k]['avg_pii_autocompletion_full_name_token_set_ratio_score']
    
    return pii_leakage_result



def aggregate_paraphrase_scores(eval_result_dict, task_prefix, aggregated_results, eval_task_dict):
    """
    Aggregate scores from paraphrase tasks (numbered 1-5) into a single score.
    
    Args:
        eval_result_dict: Dictionary containing evaluation results
        task_prefix: Prefix for the task (e.g., 'eval_log_forget_paraphrase', 'eval_log_retain_paraphrase')
        aggregated_results: Dictionary to store aggregated results
        eval_task_dict: Task name mapping dictionary
    """
    paraphrase_tasks = []
    for i in range(1, 6):  # 1 to 5
        task_name = f"{task_prefix}_{i}.json"
        if task_name in eval_result_dict:
            paraphrase_tasks.append(task_name)
    
    if not paraphrase_tasks:
        return
    
    # Determine the aggregated task name
    if 'forget' in task_prefix:
        aggregated_task_name = 'Forget Paraphrase'
    else:
        aggregated_task_name = 'Retain Paraphrase'
    
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
            # Handle the case where average_perturb_loss might be nested
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
    
    # Calculate aggregated metrics
    if all_gt_losses:
        # Probability calculation
        gt_probs = np.exp(-1 * np.array(all_gt_losses))
        avg_gt_prob = np.mean(gt_probs)
        aggregated_results[f'Prob. {aggregated_task_name}'] = avg_gt_prob
    
    if all_rougeL_recalls:
        # ROUGE calculation
        avg_rouge = np.mean(all_rougeL_recalls)
        aggregated_results[f'ROUGE {aggregated_task_name}'] = avg_rouge
    
    if all_fluency_scores:
        # Fluency calculation (average of task-level fluency scores)
        avg_fluency = np.mean(all_fluency_scores)
        aggregated_results[f'Fluency {aggregated_task_name}'] = avg_fluency
    
    if all_f1_scores:
        # F1 calculation
        avg_f1 = np.mean(all_f1_scores)
        aggregated_results[f'F1 {aggregated_task_name}'] = avg_f1
    
    # Truth Ratio calculation
    if all_paraphrased_losses and all_perturbed_losses:
        avg_paraphrase_np = np.array(all_paraphrased_losses)
        avg_perturbed_np = np.array(all_perturbed_losses)
        
        # Ensure both arrays have the same length for element-wise operations
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
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
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