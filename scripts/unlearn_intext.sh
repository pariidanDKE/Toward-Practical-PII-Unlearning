#!/bin/bash

### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=5
export batch_size=2 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=8
export cache="$PWD/cache"
export retain_weight=1
export lr=5e-5

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
export token_top_k=200

echo "Running full model without LoRA"
export num_epochs=8
#export model="llama3-8b"
export model=llama2-7b;   # [phi, llama2-7b]

#export run_name="FullFT_PII_${forget_loss}_${model}_E${num_epochs}_B${batch_size}_G${gradaccum}_lr${lr}_W${retain_weight}_intext${in_text}_replaceprob${token_replace_prob}_topk${token_top_k}_latestcompare"
export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
#export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/experiment/PII/llama3-8b/forget10/FullFT_PII_PerMU_llama3-8b_E8_B2_G8_lr1e-5_W1_intextTrue_replaceprob1_topk200_answertagging"
echo "Running model with intext=${in_text}"
# -------- Run Training --------
# python forget.py --config-name=forget_pii.yaml \
#     dataset=$dataset split=$split \
#     forget_data_path=$forget_data_path \
#     retain_data_path=$forget_data_path \
#     forget_loss=$forget_loss batch_size=$batch_size \
#     retain_weight=$retain_weight \
#     gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
#     save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
#     use_lora=$use_lora \
#     use_quantization=$use_quantization \
#     project_name=$project_name \
#     run_name=$run_name \
#     in_text=$in_text \
#     token_replace_prob=$token_replace_prob \
#     token_top_k=$token_top_k \
    
    
    # LoRA.r=$LoRA_r \
    # LoRA.alpha=$LoRA_alpha \

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


echo "Finished all full model runs"
echo "============================================"

