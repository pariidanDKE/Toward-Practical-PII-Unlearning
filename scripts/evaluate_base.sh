#!/bin/bash

### This script is used to evaluate the full model without any Unlearning on it.

export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model="llama3-8b"
export num_epochs=5
export batch_size=4
export gradaccum=4
export cache="$PWD/cache"
export forget_data_path="$PWD/data/${dataset}"
export retain_weight=1
export lr=2e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export use_lora=False
export use_quantization=False


export run_name="NoUnlearn_Llama3"
#export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/locuslab/tofu_ft_llama2-7b"
#export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_llama3-8b_B4_G4_E8_lr2e-5_answer_tagging"

# -------- Evaluate Model --------
python evaluate_PII.py --config-name=eval_pii.yaml \
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

echo "Finished run for full model with ${num_epochs} epochs"
echo "--------------------------------------------"

