#!/bin/bash

### This script runs PerMU across different batch sizes to evaluate sensitivity to batch-level perturbation dynamics.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b
export num_epochs=8
export cache="$PWD/cache"
export retain_weight=1x
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="BatchSizeCompare"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"

## PerMU in-text params
export in_text=True
export token_replace_prob=1
export token_k_neighbours=2
export subject_noise_discrepancy_addition=False
export subject_key='subject'
export sample_data_path="data/test/sample"

# Loop through different batch sizes
for batch_size in 1 2 4 8 16
do
    # Compute gradient accumulation to keep effective batch size constant (16)
    gradaccum=$((16 / batch_size))

    export batch_size=$batch_size
    export gradaccum=$gradaccum

    export run_name="_${project_name}_${model}_E${num_epochs}_B${batch_size}_G${gradaccum}"
    export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"

    echo "Running model with batch_size=${batch_size} and grad_accum=${gradaccum}"

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
        submitted_by=who

    echo "Finished run for batch_size=${batch_size}, grad_accum=${gradaccum}"
    echo "--------------------------------------------------------"
done

echo "Finished all batch size sweeps"
