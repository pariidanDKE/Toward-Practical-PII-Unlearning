#!/bin/bash

### This script is used to evaluate the full model without any Unlearning on it.
#export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export batch_size=16
export CUDA_VISIBLE_DEVICES=0,1
export forget_loss="PerMU"
export split="forget10"
export use_lora=False
export debug=true

#export model="qwen2.5-1.5b"
#export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_qwen2.5-1.5b_B64_G2_E8_lr2e-5_ComprehensiveQA"

export model="qwen2.5-32b"
export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_qwen2.5-32b_B2_G1_E2_lr1e-5/checkpoint-5282"
# -------- Evaluate Model --------
python evaluate_PII.py --config-name=eval_pii.yaml \
    model_family=$model dataset=$dataset \
    split=$split batch_size=$batch_size \
    model_path=$save_dir forget_loss=$forget_loss \
    generation.max_length=200 \
    use_lora=$use_lora \
    save_dir=$save_dir/eval_results \
    debug=$debug

# -------- Aggregate Evaluation --------
python aggregate_eval_stat.py \
    ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
    method_name=$forget_loss \
    save_file=$save_dir/eval_results/eval.csv \
    excel_file_path=$save_dir/eval_results/eval.xlsx \
    submitted_by=who

echo "--------------------------------------------"
