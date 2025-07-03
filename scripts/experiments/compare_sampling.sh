#!/bin/bash
### This script is used to evaluate the PerMU in-text method with grid search over C and P parameters
export dataset="PII"
export MASTER_PORT=18765

### Hyperparameters
export batch_size=16 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1

export CUDA_VISIBLE_DEVICES=0
export use_deepspeed=False
export optimizer="paged_adamw_8bit"

### Unlearning Params
export forget_loss="PerMU"
export project_name="ModelSizeAblation"
export forget_data_path="$PWD/data/${dataset}"
export split="forget10"

### LogggingParams
export logging_subject_token_len=False
export remove_model_tensors=False

### Other Params
export BNB_CUDA_VERSION=121
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
model_list=("llama3-8b")
export in_text=False
export C=0.6
export P=1.2


export lr=1e-5
export num_epochs=10
export do_sample_list=(False True)

echo "Starting grid search with ${#C_values[@]} C values and ${#P_values[@]} P values"
echo "Total combinations: $((${#C_values[@]} * ${#P_values[@]}))"
echo "============================================"

for model in "${model_list[@]}"; do
    if [[ "$model" == "llama3-8b" ]]; then
        export batch_size=16
        export eval_batch_size=32
    else
        export batch_size=16
        export eval_batch_size=64
    fi
    export run_name="${model}_E${num_epochs}_B${batch_size}_P${P}_C${C}_lr${lr}"
    export save_dir="$PWD/experiment/${dataset}/UnlearnParamsExperiment/${split}/${model}/$run_name"

  # #-------- Run Training --------
    # python forget.py --config-name=forget_pii.yaml \
    # dataset=$dataset split=$split \
    # forget_data_path=$forget_data_path \
    # retain_data_path=$forget_data_path \
    # forget_loss=$forget_loss batch_size=$batch_size \
    # retain_weight=$retain_weight \
    # gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
    # save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
    # project_name=$project_name \
    # run_name=$run_name \
    # in_text=$in_text \
    # C=$C \
    # P=$P \
    # use_deepspeed=$use_deepspeed \
    # optimizer=$optimizer \
    # # Grid search over C and P values
    
    for do_sample in "${do_sample_list[@]}"; do
        echo "Running experiment with do_sample=${do_sample}"
        export sample_save_dir="$PWD/experiment/${dataset}/UnlearnParamsExperiment/${split}/${model}/$run_name/do_sample_${do_sample}"

        # -------- Evaluate Model --------
        python evaluate_PII.py --config-name=eval_pii_noextract.yaml \
            model_family=$model dataset=$dataset \
            split=$split batch_size=$eval_batch_size \
            model_path=$save_dir forget_loss=$forget_loss \
            generation.max_length=200 \
            save_dir=$sample_save_dir/eval_results \
            generation.do_sample=$do_sample \

        # Check if evaluation was successful
        if [ $? -ne 0 ]; then
            echo "Evaluation failed for C=${C}, P=${P}. Continuing to next combination..."
            continue
        fi

        # -------- Aggregate Evaluation --------
        python aggregate_eval_stat.py \
            ckpt_result=$sample_save_dir/eval_results/eval_log_aggregated.json \
            method_name=$forget_loss \
            save_file=$sample_save_dir/eval_results/eval.csv \
            excel_file_path=$sample_save_dir/eval_results/eval.xlsx \
            submitted_by=who \
            remove_model_tensors=$remove_model_tensors

        echo "Finished run for C=${C}, P=${P}"
        echo "--------------------------------------------"
    done
done

echo "Finished all grid search experiments"
echo "============================================"
