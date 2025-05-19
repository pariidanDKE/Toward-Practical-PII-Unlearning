#!/bin/bash
export BNB_CUDA_VERSION=121
export dataset="TOFU"
export MASTER_PORT=18765
export model=llama2-7b  # [phi, llama2-7b,phi_chat,llama2-7b_nonchat]
export num_epochs=5
export batch_size=4
export gradaccum=4
export cache="$PWD/cache"
export forget_data_path="$PWD/data/${dataset}"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="EvaluateLoRA"
export use_quantization=False

# === Full Model Run (No LoRA) ===
echo "Running full model without LoRA"

export use_lora=False
export run_name="${project_name}_EvaluateFulModel_${model}_WarmupConstantLR"
export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
export save_model=True
export model=llama2-7b_checkpointenable




# -------- Run Training --------
python forget.py --config-name=forget_pii.yaml \
    dataset=$dataset split=$split \
    forget_data_path=$forget_data_path \
    retain_data_path=$forget_data_path \
    forget_loss=$forget_loss batch_size=$batch_size \
    retain_weight=$retain_weight \
    gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
    save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
    use_lora=$use_lora \
    use_quantization=$use_quantization \
    project_name=$project_name \
    run_name=$run_name \
    save_model=$save_model

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


# === LoRA Config Runs ===
export model=llama2-7b
export use_lora=True
# LoRA rank configs
lora_ranks=(128 256 512)
alpha_multipliers=(1 2 3 4)  # alpha = multiplier * r

declare -A retain_splits
retain_splits["forget01"]="retain99"
retain_splits["forget05"]="retain95"
retain_splits["forget10"]="retain90"

for LoRA_r in "${lora_ranks[@]}"; do
  for multiplier in "${alpha_multipliers[@]}"; do

    if [[ "$LoRA_r" -eq 128 && ( "$multiplier" -eq 1 || "$multiplier" -eq 2 || "$multiplier" -ne 3 ) ]]; then
      echo "Skipping LoRA.r=${LoRA_r}, multiplier=${multiplier}"
      continue
    fi

    LoRA_alpha=$((LoRA_r * multiplier))

    echo "Running with LoRA.r=${LoRA_r}, LoRA.alpha=${LoRA_alpha}"

    export run_name="EvaluateLoRA_model${model}_r${LoRA_r}_a${LoRA_alpha}_WarmupConstantLR"
    export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"

    # -------- Run Training --------
    python forget.py --config-name=forget.yaml \
        dataset=$dataset split=$split \
        forget_data_path=$forget_data_path \
        retain_data_path=$forget_data_path \
        forget_loss=$forget_loss batch_size=$batch_size \
        retain_weight=$retain_weight \
        gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
        save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
        use_lora=$use_lora \
        use_quantization=$use_quantization \
        LoRA.r=$LoRA_r \
        LoRA.alpha=$LoRA_alpha \
        project_name=$project_name \
        run_name=$run_name  \
        save_model=$save_model

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

    echo "Finished run for r=${LoRA_r}, alpha=${LoRA_alpha}"
    echo "--------------------------------------------"
  done
done

