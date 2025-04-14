#!/bin/bash
export BNB_CUDA_VERSION=121
export dataset="TOFU"
export MASTER_PORT=18765
export model="llama2-7b"
export num_epochs=5
export batch_size=4 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=3
export cache="$PWD/cache"
export forget_data_path="$PWD/data/${dataset}"
export retain_weight=1
export lr=2e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget01"
export project_name="EvaluateLoRA"
export use_lora=True
export use_quantization=False

# LoRA rank configs
lora_ranks=(16 32 64 128 256 512)
alpha_multipliers=(1 2 3 4)  # alpha = multiplier * r


# lora_ranks=(16)
# alpha_multipliers=(1 2)  # alpha = multiplier * r


declare -A retain_splits
retain_splits["forget01"]="retain99"
retain_splits["forget05"]="retain95"
retain_splits["forget10"]="retain90"

# Loop through rank and multiplier combos
for LoRA_r in "${lora_ranks[@]}"; do
  for multiplier in "${alpha_multipliers[@]}"; do
    LoRA_alpha=$((LoRA_r * multiplier))

    # Skip specific combinations
    # if [["$LoRA_r" -eq 256 && ( "$multiplier" -eq 2) ]]; then
    #   echo "Skipping LoRA.r=${LoRA_r}, multiplier=${multiplier} (alpha=$((LoRA_r * multiplier)))"
    #   continue
    # fi


    echo "Running with LoRA.r=${LoRA_r}, LoRA.alpha=${LoRA_alpha}"

    export run_name="EvaluateLoRA_r${LoRA_r}_a${LoRA_alpha}_WarmupConstantLR"
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
        run_name=$run_name

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
