#!/bin/bash
export BNB_CUDA_VERSION=121
export dataset="TOFU"
export MASTER_PORT=18765
export model=phi3-5-mini-instruct  # [phi, llama2-7b,phi_chat,llama2-7b_nonchat,phi3-5-mini-instruct]
export num_epochs=5
export batch_size=4
export gradaccum=4
export cache="$PWD/cache"
export forget_data_path="$PWD/data/${dataset}"
export retain_weight=1
export lr=2e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget01"
export project_name="CompareFoundationInstruct"
export use_quantization=False

# === Full Model Run (No LoRA) ===
echo "Running full model without LoRA"

export use_lora=False
export run_name="${project_name}_EvaluateFulModel_${model}_WarmupConstantLR"
export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
export save_model=True


# export model=llama2-7b


# # -------- Run Training --------
# python forget_test.py --config-name=forget_pii.yaml \
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
#     save_model=$save_model

# -------- Evaluate Model --------
python evaluate_${dataset}.py \
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

echo "Finished run for full model"
echo "============================================"
