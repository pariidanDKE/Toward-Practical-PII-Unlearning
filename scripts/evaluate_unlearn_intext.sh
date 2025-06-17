#!/bin/bash

### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export cache="$PWD/cache"
export CUDA_VISIBLE_DEVICES=0
export split="forget10"
export project_name="SyntheticPII"
export use_lora=False
export overwrite=True

## PerMU in-text params

export remove_model_tensors=False


echo "Running full model without LoRA"
export model=llama2-7b_noanswerspace;  


export batch_size=64 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export forget_loss="PerMU_intext"
export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/retain_and_test_retain_llama2-7b_noanswerspace_B32_G4_E5_lr2e-5_ComprehensiveQA"

echo "Running evaluate.."

# -------- Evaluate Model --------
python evaluate_PII.py --config-name=eval_pii.yaml \
    model_family=$model dataset=$dataset \
    split=$split batch_size=$batch_size \
    model_path=$save_dir forget_loss=$forget_loss \
    generation.max_length=200 \
    use_lora=$use_lora \
    save_dir=$save_dir/eval_results \
    #overwrite=True \

# -------- Aggregate Evaluation --------
python aggregate_eval_stat.py \
    ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
    method_name=$forget_loss \
    save_file=$save_dir/eval_results/eval.csv \
    excel_file_path=$save_dir/eval_results/eval.xlsx \
    submitted_by=who \
    remove_model_tensors=$remove_model_tensors \

echo "--------------------------------------------"

echo "Finished all full model runs"
echo "============================================"

