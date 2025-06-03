#!/bin/bash

### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=5
export batch_size=4 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=4
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0

export split="forget10"
export project_name="SyntheticPII"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"

echo "Running full model without LoRA"
export num_epochs=1

# Array of forget loss methods to loop through
#forget_losses=("PerMU" "grad_ascent" "grad_ascent+kl" "grad_ascent+gd" "dpo" "dpo+kl" "dpo+gd" "npo" "npo+kl" "npo+gd" "task_vector" "ULD" "WHP" "icl"  "PerMUintext")
forget_losses=("grad_ascent")

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
        
        # Set PerMUintext-specific parameters only if forget_loss is PerMUintext
        if [ "$forget_loss" = "PerMUintext" ]; then
            export in_text=True
            export token_replace_prob=1
            export token_top_k=200
            echo "Setting PerMUintext parameters: in_text=$in_text, token_replace_prob=$token_replace_prob, token_top_k=$token_top_k"
        else
            # Unset or set to default values for other methods
            unset in_text
            unset token_replace_prob
            unset token_top_k
            echo "Using default parameters (no PerMUintext-specific settings)"
        fi

        export run_name="FullFT_PII_${forget_loss}_${model}_"
        export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
        
        echo "-------- Run Training --------"
        
        # Build the command conditionally
        train_cmd="python forget.py --config-name=forget_pii.yaml \
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
            run_name=$run_name"
        
        # Add PerMUintext-specific parameters if needed
        if [ "$forget_loss" = "PerMUintext" ]; then
            train_cmd="$train_cmd \
                in_text=$in_text \
                token_replace_prob=$token_replace_prob \
                token_top_k=$token_top_k"
        fi
        
        # Execute the training command
        eval $train_cmd
        
        echo "-------- Evaluate Model --------"
        python evaluate_PII.py --config-name=eval_pii.yaml \
            model_family=$model dataset=$dataset \
            split=$split batch_size=$batch_size \
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
            submitted_by=who

        echo "✓ Successfully completed $forget_loss with ${num_epochs} epochs"
        
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