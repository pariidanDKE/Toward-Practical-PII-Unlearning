#!/bin/bash

## This script runs both training and evaluation for PerMU in-text method
## Token Replace Probability Experiments: 4 probabilities x 5 runs each = 20 total runs
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama3.1-8b;   # [phi, llama2-7b]
export num_epochs=8
export train_batch_size=16  # Updated from 16 as per your note
export eval_batch_size=64  # Updated from 16 as per your note
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5
export optimizer="paged_adamw_8bit"


export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="LLama3.1_PerMuTok"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"
## PerMU in-text base params
export in_text=True
export logging_timestats=False
export remove_model_tensors=True 
export logging_corrupted_subjects=False
export logging_permu_contrast_stats=False

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Fixed parameters
export num_runs=5
export optimal_neighbours_generation=True

# Fixed K-Distance configuration (second one from original script)
export token_k_val="2"
export match_first_char="False"
export use_adaptive_k="False"
export config_name="k2_standard"

# Define token replace probability values to iterate through
token_replace_probs=(0.25 0.5 0.75 1.0)



# Counter for run ID
run_counter=1
total_runs=$((${#token_replace_probs[@]} * num_runs))

echo "Starting Token Replace Probability experiment suite..."
echo "K-Distance Configuration: $config_name (k=$token_k_val, match_first_char=$match_first_char, use_adaptive_k=$use_adaptive_k)"
echo "Token replace probabilities: ${token_replace_probs[*]}"
echo "Runs per probability: $num_runs"
echo "Total runs: $total_runs"
echo "============================================"

# Loop through each token replace probability
for token_replace_prob in "${token_replace_probs[@]}"; do
    
    echo ""
    echo "=========================================="
    echo "Starting Token Replace Probability: $token_replace_prob"
    echo "K-Distance Config: $config_name"
    echo "=========================================="
    
    # Run this probability num_runs times
    for run_num in $(seq 1 $num_runs); do
        # Set current token replace probability
        export token_replace_prob="$token_replace_prob"

        # Generate run name with all parameters
        export run_name="${project_name}_${model}_E${num_epochs}_B${train_batch_size}_${config_name}_prob${token_replace_prob}_run${run_num}"
        export save_dir="$PWD/experiment/${dataset}/${model}/${split}/_AllExperiments/ReplaceProb/$run_name"
        
        # Create individual log file for this run
        
        echo ""
        echo "----------------------------------------"
        echo "Starting Run $run_counter/$total_runs"
        echo "Token Replace Prob: $token_replace_prob (Run $run_num/$num_runs)"
        echo "Token K Neighbours: $token_k_val"
        echo "Match First Char: $match_first_char"
        echo "Use Adaptive K: $use_adaptive_k"
        echo "Run Name: $run_name"
        echo "Save Dir: $save_dir"
        
        # Record start time
        start_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        
        # Run the training and evaluation pipeline with logging
        {
            echo "=== TRAINING PHASE ==="
            
            #export batch_size=16  # Reset batch size for training
            
            # Run actual training
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
                run_name=$run_name \
                in_text=$in_text \
                token_replace_prob=$token_replace_prob \
                token_k_neighbours=$token_k_val \
                logging.time_stats=$logging_timestats \
                logging.corrupted_subjects=$logging_corrupted_subjects \
                match_first_char=$match_first_char \
                use_adaptive_k=$use_adaptive_k \
                optimal_neighbours_generation=$optimal_neighbours_generation \
                logging.permu_contrast_stats=$logging_permu_contrast_stats \
                optimizer=$optimizer \


            # Capture actual training exit code
            training_exit_code=$?
            
            if [ $training_exit_code -eq 0 ]; then
                echo ""
                echo "Training completed successfully!"
                echo "=== EVALUATION PHASE ==="
                
                # Check if model exists before evaluation
                if [ -d "$save_dir" ]; then
                    echo "Model directory found: $save_dir"
                    

                    echo "Changed batch size to 64 for evaluation"
                    #export batch_size=64
                    # Evaluation
                    python evaluate_PII.py --config-name=eval_pii.yaml \
                        model_family=$model dataset=$dataset \
                        split=$split batch_size=$eval_batch_size \
                        model_path=$save_dir forget_loss=$forget_loss \
                        generation.max_length=200 \
                        use_lora=$use_lora \
                        save_dir=$save_dir/eval_results \
                    
                    eval_exit_code=$?
                else
                    echo "ERROR: Model directory not found after training: $save_dir"
                    eval_exit_code=1
                fi
                
                if [ $eval_exit_code -eq 0 ]; then
                    echo ""
                    echo "Evaluation completed successfully!"
                    echo "=== AGGREGATION PHASE ==="
                    
                    # Aggregation
                    python aggregate_eval_stat.py \
                        ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
                        method_name=$forget_loss \
                        save_file=$save_dir/eval_results/eval.csv \
                        excel_file_path=$save_dir/eval_results/eval.xlsx \
                        submitted_by=token_replace_prob_experiment \
                        remove_model_tensors=$remove_model_tensors \

                    agg_exit_code=$?

                    if [ $agg_exit_code -eq 0 ]; then
                        echo "Aggregation completed successfully!"
                        final_status="SUCCESS"
                    else
                        echo "Aggregation failed!"
                        final_status="FAILED_AGGREGATION"
                    fi
                else
                    echo "Evaluation failed!"
                    final_status="FAILED_EVALUATION"
                fi
            else
                echo "Training failed with exit code: $training_exit_code"
                final_status="FAILED_TRAINING"
            fi
            
        } 2>&1 
        # Record end time and status
        end_time=$(date '+%Y-%m-%d %H:%M:%S')
        
        # Update master log with completion status
        
        echo "Run $run_counter completed with status: $final_status"
        echo "End time: $end_time"
        
        # Increment counter
        ((run_counter++))
        
        # Optional: Add a small delay between runs
        sleep 5
        
    done
    
    echo ""
    echo "Token Replace Probability $token_replace_prob completed (${num_runs} runs)"
    
done

