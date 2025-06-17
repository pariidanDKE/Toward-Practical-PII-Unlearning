#!/bin/bash

### This script is used to evaluate multiple forget loss methods with method-specific batch size handling
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0

export time_stats=True
export remove_model_tensors=True
export split="forget10"
export project_name="CompareMethods_Experiment2"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"

# CUDA debugging and logging parameters
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

echo "Running Comparison Experiment with forget loss methods"
export num_epochs=5

# Array of forget loss methods to loop through
#forget_losses=("PerMU" "grad_ascent" "ULD" "WHP" "grad_ascent+kl" "grad_ascent+gd" "dpo" "dpo+kl" "dpo+gd" "npo" "npo+kl" "npo+gd")
forget_losses=("PerMU")


# Arrays to track results
successful_methods=()
failed_methods=()

# Loop through each forget loss method
for forget_loss in "${forget_losses[@]}"; do
    echo "============================================"
    echo "Running experiment with forget_loss: $forget_loss"
    echo "============================================"
    
    # Wrap the entire method execution in error handling
    (
        set -e  # Exit on any error within this subshell
        
        # Set method-specific batch sizes
        if [ "$forget_loss" = "dpo+kl" ] || [ "$forget_loss" = "dpo+gd" ]; then
            export train_batch_size=4
            export eval_batch_size=32
            echo "Setting memory-intensive method batch sizes: train_batch_size=$train_batch_size, eval_batch_size=$eval_batch_size"
        else
            export train_batch_size=16
            export eval_batch_size=64
            echo "Using standard batch sizes: train_batch_size=$train_batch_size, eval_batch_size=$eval_batch_size"
        fi

        export run_name="ModelComparison_${forget_loss}_E${num_epochs}"
        export save_dir="$PWD/experiment/${dataset}/${model}/${split}/_AllExperiments/Experiment_MethodComparison/$run_name"
        
        echo "-------- Run Training --------"
        
        # Training command
        python forget.py --config-name=forget_pii.yaml \
            dataset=$dataset split=$split \
            forget_data_path=$forget_data_path \
            retain_data_path=$forget_data_path \
            forget_loss=$forget_loss batch_size=$train_batch_size \
            retain_weight=$retain_weight \
            gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
            save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
            use_lora=$use_lora \
            use_quantization=$use_quantization \
            project_name=$project_name \
            run_name=$run_name
        
        echo "-------- Evaluate Model --------"
        python evaluate_PII.py --config-name=eval_pii.yaml \
            model_family=$model dataset=$dataset \
            split=$split batch_size=$eval_batch_size \
            model_path=$save_dir forget_loss=$forget_loss \
            generation.max_length=200 \
            use_lora=$use_lora \
            save_dir=$save_dir/eval_results

        echo "-------- Aggregate Evaluation --------"
        python aggregate_eval_stat.py \
            ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
            method_name=$forget_loss \
            save_file=$save_dir/eval_results/eval.csv \
            excel_file_path=$save_dir/eval_results/eval.xlsx \
            submitted_by=comparison_experiment \
            remove_model_tensors=$remove_model_tensors

        echo "✓ Successfully completed $forget_loss with ${num_epochs} epochs"
        echo "  Training batch size: $train_batch_size, Evaluation batch size: $eval_batch_size"
        
    ) && {
        # Success case
        successful_methods+=("$forget_loss")
        echo "✓ $forget_loss completed successfully"
    } || {
        # Failure case
        failed_methods+=("$forget_loss")
        echo "✗ ERROR: $forget_loss failed with exit code $?. Skipping to next method..."
        echo "Continuing with remaining methods..."
    }
    
    echo "--------------------------------------------"
done

# Print final summary
echo "============================================"
echo "FINAL SUMMARY"
echo "============================================"
echo "Total methods attempted: ${#forget_losses[@]}"
echo "Successful methods (${#successful_methods[@]}): ${successful_methods[*]}"
echo "Failed methods (${#failed_methods[@]}): ${failed_methods[*]}"
echo "============================================"

if [ ${#failed_methods[@]} -eq 0 ]; then
    echo "✓ All methods completed successfully!"
    exit 0
else
    echo "⚠ Some methods failed. Check logs above for details."
    exit 1
fi