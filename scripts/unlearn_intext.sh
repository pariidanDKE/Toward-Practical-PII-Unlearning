#!/bin/bash
### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
export dataset="PII"
export MASTER_PORT=18765
export num_epochs=8
export batch_size=8 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5
export forget_loss="PerMU"
export project_name="SyntheticPII"
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"
export remove_model_tensors=True
export split="forget10"
export logging_subject_token_len=False
export optimal_neighbours_generation=False

export model=phi_chat   
export in_text=True
export C=0.1
export P=0.4
export num_epochs=5
export batch_size=16



export BNB_CUDA_VERSION=121
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export run_name="FullFT_PII_${model}_E${num_epochs}_B${batch_size}_intext${in_text}"

export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
echo "Running model with intext=${in_text}"
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
    logging.subject_token_len=$logging_subject_token_len \
    optimal_neighbours_generation=$optimal_neighbours_generation \
    C=$C \
    P=$P \


# -------- Evaluate Model --------
python evaluate_PII.py --config-name=eval_pii_short.yaml \
    model_family=$model dataset=$dataset \
    split=$split batch_size=$batch_size \
    model_path=$save_dir forget_loss=$forget_loss \
    generation.max_length=200 \
    use_lora=$use_lora \
    save_dir=$save_dir/eval_results \
    #data_path=$sample_data_path \

# -------- Aggregate Evaluation --------
python aggregate_eval_stat.py \
    ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
    method_name=$forget_loss \
    save_file=$save_dir/eval_results/eval.csv \
    excel_file_path=$save_dir/eval_results/eval.xlsx \
    submitted_by=who \
    remove_model_tensors=$remove_model_tensors

echo "Finished run for full model with ${num_epochs} epochs"
echo "--------------------------------------------"

echo "Finished all full model runs"
echo "============================================"