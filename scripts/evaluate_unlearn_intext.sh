#!/bin/bash

### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=5
export batch_size=16 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=4
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="SyntheticPII"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"

## PerMU in-text params
export in_text=True
export token_replace_prob=1
export token_k_neighbours=1
export subject_noise_discrepancy_addition=True



echo "Running full model without LoRA"
export num_epochs=8
#export model="llama3-8b"
export model=llama2-7b;   # [phi, llama2-7b]

export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/experiment/PII/llama2-7b/forget10/_PermUIntext_Experiment1_llama2-7b_E8_B16_rp1.0_kn1"
echo "Running evaluate for run: $run_name"


# -------- Evaluate Model --------
# python evaluate_PII.py --config-name=eval_pii.yaml \
#     model_family=$model dataset=$dataset \
#     split=$split batch_size=$batch_size \
#     model_path=$save_dir forget_loss=$forget_loss \
#     generation.max_length=200 \
#     use_lora=$use_lora \
#     save_dir=$save_dir/eval_results2

# -------- Aggregate Evaluation --------
python aggregate_eval_stat.py \
    ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
    method_name=$forget_loss \
    save_file=$save_dir/eval_results/eval.csv \
    excel_file_path=$save_dir/eval_results/eval.xlsx \
    submitted_by=who

echo "Finished run for full model with ${num_epochs} epochs"
echo "--------------------------------------------"

echo "Finished all full model runs"
echo "============================================"

