#!/bin/bash

### This script runs PerMU across different subject_noise_discrepancy_addition values to evaluate sensitivity to noise addition method.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b
export num_epochs=8
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="SubjectNoiseCompare"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"

## PerMU in-text params
export in_text=True
export token_replace_prob=1
export token_k_neighbours=2
export subject_key="subject"  # Fixed subject_key value
export remove_model_tensors=True

# Fixed training parameters
export batch_size=16
export gradaccum=2

# Calculate effective batch size for logging
effective_batch_size=$((batch_size * gradaccum))

# Loop through different subject_noise_discrepancy_addition values
for subject_noise_discrepancy_addition in "True" "False"
do
    export subject_noise_discrepancy_addition=$subject_noise_discrepancy_addition

    export run_name="_${project_name}_${model}_E${num_epochs}_B${batch_size}_G${gradaccum}_SNDA${subject_noise_discrepancy_addition}"
    export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"

    echo "Running model with subject_noise_discrepancy_addition=${subject_noise_discrepancy_addition} (batch_size=${batch_size}, grad_accum=${gradaccum}, effective_batch_size=${effective_batch_size})"

    #-------- Run Training --------
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
        in_text=$in_text \
        token_replace_prob=$token_replace_prob \
        token_k_neighbours=$token_k_neighbours \
        subject_key=$subject_key \
        subject_noise_discrepancy_addition=$subject_noise_discrepancy_addition

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
        submitted_by=who \
        remove_model_tensors=$remove_model_tensors

    echo "Finished run for subject_noise_discrepancy_addition=${subject_noise_discrepancy_addition}"
    echo "--------------------------------------------------------"
done

echo "Finished all subject_noise_discrepancy_addition sweeps"