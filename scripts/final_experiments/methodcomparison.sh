#!/bin/bash

### This script is used to evaluate multiple forget loss methods with method-specific batch size handling
### Now runs 5 iterations for each method
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama3.1-8b;   # [phi, llama2-7b]
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5
export P=0.4
export C=0.1
export optimizer="paged_adamw_8bit"


export CUDA_VISIBLE_DEVICES=0
export remove_model_tensors=True
export split="forget10"
export project_name="CompareMethods_ExperimentSampling_Llama3.1"
export forget_data_path="$PWD/data/${dataset}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA debugging and logging parameters
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export generation_do_sample=False
export use_deepspeed=False

echo "Running Comparison Experiment with forget loss methods"
export num_epochs=8
export num_runs=5

# Array of forget loss methods to loop through
forget_losses=("grad_ascent" "dpo" "grad_ascent+kl" "grad_ascent+gd" "dpo+kl" "dpo+gd" "npo" "npo+kl" "npo+gd")



#forget_losses=("grad_ascent+gd" "dpo+gd" "npo" "npo+kl" "npo+gd" "dpo+kl")
#forget_losses=("ULD" "WHP")
#forget_losses=("ULD")

# Loop through each forget loss method
for forget_loss in "${forget_losses[@]}"; do
    echo "============================================"
    echo "Running experiment with forget_loss: $forget_loss"
    echo "============================================"
    
    # Loop through each run for the current method
    for run in $(seq 1 $num_runs); do
        echo "-------- Method: $forget_loss, Run: $run --------"
        
        # Wrap the entire method execution in error handling
        (
            set -e  # Exit on any error within this subshell
            
            # Set method-specific batch sizes
            if [ "$forget_loss" = "dpo+kl" ] || [ "$forget_loss" = "npo+kl" ] || [ "$forget_loss" = "dpo" ] || [ "$forget_loss" = "grad_ascent+kl" ]; then
                export train_batch_size=4
                export eval_batch_size=32
                echo "Setting memory-intensive method batch sizes: train_batch_size=$train_batch_size, eval_batch_size=$eval_batch_size"
            else
                export train_batch_size=16
                export eval_batch_size=16
                echo "Using standard batch sizes: train_batch_size=$train_batch_size, eval_batch_size=$eval_batch_size"
            fi

            export run_name="ModelComparison_${forget_loss}_E${num_epochs}_Run${run}"
            export save_dir="$PWD/experiment/${dataset}/${model}/${split}/_AllExperiments/MethodComparison/$run_name"
            
            echo "-------- Run Training --------"
            
            #Training command
            python forget.py --config-name=forget_pii.yaml \
                dataset=$dataset split=$split \
                forget_data_path=$forget_data_path \
                retain_data_path=$forget_data_path \
                forget_loss=$forget_loss batch_size=$train_batch_size \
                retain_weight=$retain_weight \
                gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
                save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
                project_name=$project_name \
                run_name=$run_name \
                use_deepspeed=$use_deepspeed \
                optimizer=$optimizer \
            
            echo "-------- Evaluate Model --------"
            python evaluate_PII.py --config-name=eval_pii.yaml \
                model_family=$model dataset=$dataset \
                split=$split batch_size=$eval_batch_size \
                model_path=$save_dir forget_loss=$forget_loss \
                generation.max_length=200 \
                use_lora=$use_lora \
                save_dir=$save_dir/eval_results \
                generation.do_sample=$generation_do_sample \

            echo "-------- Aggregate Evaluation --------"
            python aggregate_eval_stat.py \
                ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
                method_name=${forget_loss}_run${run} \
                save_file=$save_dir/eval_results/eval.csv \
                excel_file_path=$save_dir/eval_results/eval.xlsx \
                submitted_by=comparison_experiment \
                remove_model_tensors=$remove_model_tensors

            echo "✓ Successfully completed $forget_loss Run $run with ${num_epochs} epochs"
            echo "  Training batch size: $train_batch_size, Evaluation batch size: $eval_batch_size"
            
        ) && {
            # Success case
            ((method_successful_runs++))
            ((successful_runs++))
            echo "✓ $forget_loss Run $run completed successfully"
        } || {
            # Failure case
            ((method_failed_runs++))
            echo "✗ ERROR: $forget_loss Run $run failed. Continuing to next run..."
        }
        
        ((total_runs++))
        echo "Progress: Method $forget_loss - Run $run/$num_runs completed"
        echo "Method stats: $method_successful_runs successful, $method_failed_runs failed"
        echo "--------------------------------------------"
    done
    
    echo "--------------------------------------------"
done