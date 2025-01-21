
from omegaconf import OmegaConf
import hydra 
import json 
import numpy as np
from scipy.stats import hmean
from scipy.stats import sem, hmean, ks_2samp
import pprint
import csv 
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

def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log_retain.json': 'Retain',
        'eval_log_forget.json': 'Forget',
    }
    if "eval_log_forget_rephrase.json" in eval_result_dict.keys():
        eval_task_dict['eval_log_forget_rephrase.json'] = 'Forget Rephrase'
        eval_task_dict['eval_log_retain_rephrase.json'] = 'Retain Rephrase'
    if 'eval_log_forget_inverse.json' in eval_result_dict.keys():
        eval_task_dict['eval_log_forget_inverse.json'] = 'Forget Inverse'
        eval_task_dict['eval_log_retain_inverse.json'] = 'Retain Inverse'
    if 'eval_log_forget_onehop.json' in eval_result_dict.keys():
        eval_task_dict['eval_log_forget_onehop.json'] = 'Forget Onehop'
        eval_task_dict['eval_log_retain_onehop.json'] = 'Retain Onehop'
    if 'eval_log_forget_subject_replace.json' in eval_result_dict.keys():
        eval_task_dict['eval_log_forget_subject_replace.json'] = 'Forget Subject_Replace'
        eval_task_dict['eval_log_retain_subject_replace.json'] = 'Retain Subject_Replace'
        
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Prob.', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[metric + ' ' + eval_task_dict[eval_task]] = []
    
    fluency_result = {}
    f1_result = {}
    bleu_result = {}
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
        
        # if 'bleu' in eval_result_dict[k].keys():
        #     avg_bleu = np.array(list(eval_result_dict[k]['bleu'].values())).mean()
        #     output_result[f'BLEU {eval_task_dict[k]}'] = avg_bleu
        
        # getting Truth Ratio
        if 'avg_paraphrased_loss' in eval_result_dict[k].keys():
            avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))
            avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
            avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)
            curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)
            if 'forget' in k:
                paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
            else:
                paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
            output_result[f'Truth Ratio {eval_task_dict[k]}'] = paraphrased_perturb_ratio
        else:
            output_result.pop(f'Truth Ratio {eval_task_dict[k]}')

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k and 'Rephrase' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    output_result['MRP'] = output_result['Model Utility']/((output_result[f'ROUGE Forget'] + output_result[f'Prob. Forget'])/2)
    if 'Truth Ratio Forget' in output_result:
        output_result['MTR'] = output_result['Model Utility']*output_result[f'Truth Ratio Forget'] / (output_result['Model Utility']+output_result[f'Truth Ratio Forget'])
    return output_result, fluency_result, f1_result, bleu_result

@hydra.main(version_base=None, config_path="config", config_name="aggregate_eval_stat")
def main(cfg):
    if cfg.ckpt_result is None:
        raise ValueError("Please provide either retain_result or ckpt_result")
    
    retain_result = None
    if cfg.retain_result != "None":
        retain_result = json.load(open(cfg.retain_result))
    ckpt_result = json.load(open(cfg.ckpt_result))

    # We have to assume here that retain_result and ckpt_result follow these structure:
    # The top most layer has ['eval_log.json', 'eval_log_forget.json', 'eval_real_world_wo_options.json', 'eval_real_author_wo_options']
    # the second layer contains the actual metrics: ['avg_gt_loss', 'average_perturb_loss', 'avg_paraphrased_loss', 'rougeL_recall']
    # within each metric, we have {data_idx: measurement}

    model_utility, fluency_result, f1_result, bleu_result = get_model_utility(ckpt_result)
    model_utility.update(fluency_result)
    if len(f1_result) != 0:
        model_utility.update(f1_result)
    if len(bleu_result) != 0:
        model_utility.update(bleu_result)
    if 'Truth Ratio Forget' in model_utility and retain_result is not None:
        forget_quality = get_forget_quality(ckpt_result, retain_result)
        model_utility['Forget Quality'] = forget_quality['Forget Quality']

    model_utility['Method'] = cfg.method_name
    model_utility['Submitted By'] = cfg.submitted_by
    # dump the model utility to a csv
    with open(cfg.save_file, 'w') as f:  # You will need 'wb' mode in Python 2.x
        w = csv.DictWriter(f, model_utility.keys())
        w.writeheader()
        w.writerow(model_utility)
        
    import pandas as pd
    df = pd.read_csv(cfg.save_file)
    df = df.applymap(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
    df.to_excel(cfg.excel_file_path, index=False)
    
    return model_utility
    
if __name__ == "__main__":
    main()