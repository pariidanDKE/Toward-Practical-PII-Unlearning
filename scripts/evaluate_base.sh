#!/bin/bash

### This script is used to evaluate the full model without any Unlearning on it.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
#export model="llama3.1-8b"  # [llama3.1-8b, phi_chat, phi3-5-mini-instruct, llama2-7b_noanswerspace]
export model="qwen2.5-7b"  # [llama3.1-8b, phi_chat, phi3-5-mini-instruct, llama2-7b_noanswerspace, qwen2.5-7b]

#export model="phi3-5-mini-instruct"
#export model="llama2-7b_noanswerspace"
export batch_size=64

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export use_lora=False

#export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_llama3.1-8b_B32_G4_E5_lr2e-5_ComprehensiveQA/checkpoint-1650"
export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_qwen2.5-7b_B32_G4_E5_lr2e-5_ComprehensiveQA/checkpoint-1650"

# -------- Evaluate Model --------
python evaluate_PII.py --config-name=eval_pii_short.yaml \
    model_family=$model dataset=$dataset \
    split=$split batch_size=$batch_size \
    model_path=$save_dir forget_loss=$forget_loss \
    generation.max_length=200 \
    use_lora=$use_lora \
    save_dir=$save_dir/eval_results

# -------- Aggregate Evaluation --------
python aggregate_eval_stat.py \
    ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
    method_name=$forget_loss \
    save_file=$save_dir/eval_results/eval.csv \
    excel_file_path=$save_dir/eval_results/eval.xlsx \
    submitted_by=who

echo "--------------------------------------------"

