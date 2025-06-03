#!/bin/bash

### This script runs all 96 configurations for PerMU in-text method evaluation
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=5
export batch_size=16  # Updated from 16 as per your note
export gradaccum=4
export cache="$PWD/cache"
export retain_weight=1
export lr=2e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="SyntheticPII"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"

## PerMU in-text base params
export in_text=True

# Define parameter arrays
subject_keys=("subject" "subject_pii" "subject_person_pii")
noise_additions=("True" "False")
token_replace_probs=("0.25" "0.5" "0.75" "1.0")
token_k_neighbours=("1" "2" "3" "4")

# Create experiment log directory
experiment_log_dir="$PWD/experiment_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$experiment_log_dir"

# Create master log file
master_log="$experiment_log_dir/master_experiment_log.csv"
echo "run_id,subject_key,noise_addition,token_replace_prob,token_k_neighbours,run_name,save_dir,start_time,end_time,status" > "$master_log"

# Counter for run ID
run_counter=1

echo "Starting PerMU experiment suite with 96 configurations..."
echo "Experiment logs will be saved to: $experiment_log_dir"
echo "============================================"

# Nested loops for all parameter combinations
for subject_key in "${subject_keys[@]}"; do
    for noise_addition in "${noise_additions[@]}"; do
        for token_replace_prob in "${token_replace_probs[@]}"; do
            for token_k_neighbour in "${token_k_neighbours[@]}"; do
                
                # Set current parameters
                export subject_key_choice="$subject_key"
                export subject_noise_discrepancy_addition="$noise_addition"
                export token_replace_prob="$token_replace_prob"
                export token_k_neighbours="$token_k_neighbour"
                
                # Generate run name with all parameters
                export run_name="PerMU_${model}_E${num_epochs}_B${batch_size}_sk${subject_key}_na${noise_addition}_rp${token_replace_prob}_kn${token_k_neighbour}"
                export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"
                
                # Create individual log file for this run
                run_log="$experiment_log_dir/run_${run_counter}_${run_name}.log"
                
                echo "----------------------------------------"
                echo "Starting Run $run_counter/96"
                echo "Subject Key: $subject_key"
                echo "Noise Addition: $noise_addition"
                echo "Token Replace Prob: $token_replace_prob"
                echo "Token K Neighbours: $token_k_neighbour"
                echo "Run Name: $run_name"
                echo "Save Dir: $save_dir"
                echo "Log File: $run_log"
                
                # Record start time
                start_time=$(date '+%Y-%m-%d %H:%M:%S')
                
                # Log experiment start to master log
                echo "$run_counter,$subject_key,$noise_addition,$token_replace_prob,$token_k_neighbour,$run_name,$save_dir,$start_time,RUNNING,STARTED" >> "$master_log"
                
                # Run the experiment with logging
                {
                    echo "=== TRAINING PHASE ==="
                    echo "Started at: $start_time"
                    echo "Parameters:"
                    echo "  - subject_key: $subject_key"
                    echo "  - subject_noise_discrepancy_addition: $noise_addition"
                    echo "  - token_replace_prob: $token_replace_prob"
                    echo "  - token_k_neighbours: $token_k_neighbour"
                    echo ""
                    
                    # Training
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
                        subject_key=$subject_key_choice \
                        subject_noise_discrepancy_addition=$subject_noise_discrepancy_addition
                    
                    training_exit_code=$?
                    
                    if [ $training_exit_code -eq 0 ]; then
                        echo ""
                        echo "=== EVALUATION PHASE ==="
                        
                        # Evaluation
                        python evaluate_PII.py --config-name=eval_pii.yaml \
                            model_family=$model dataset=$dataset \
                            split=$split batch_size=$batch_size \
                            model_path=$save_dir forget_loss=$forget_loss \
                            generation.max_length=200 \
                            use_lora=$use_lora \
                            save_dir=$save_dir/eval_results
                        
                        eval_exit_code=$?
                        
                        if [ $eval_exit_code -eq 0 ]; then
                            echo ""
                            echo "=== AGGREGATION PHASE ==="
                            
                            # Aggregation
                            python aggregate_eval_stat.py \
                                ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
                                method_name=$forget_loss \
                                save_file=$save_dir/eval_results/eval.csv \
                                excel_file_path=$save_dir/eval_results/eval.xlsx \
                                submitted_by=experiment_suite
                            
                            agg_exit_code=$?
                            
                            if [ $agg_exit_code -eq 0 ]; then
                                final_status="SUCCESS"
                            else
                                final_status="FAILED_AGGREGATION"
                            fi
                        else
                            final_status="FAILED_EVALUATION"
                        fi
                    else
                        final_status="FAILED_TRAINING"
                    fi
                    
                } 2>&1 | tee "$run_log"
                
                # Record end time and status
                end_time=$(date '+%Y-%m-%d %H:%M:%S')
                
                # Update master log with completion status
                sed -i "${run_counter}s/RUNNING/$final_status/" "$master_log"
                sed -i "${run_counter}s/STARTED/$end_time/" "$master_log"
                
                echo "Run $run_counter completed with status: $final_status"
                echo "End time: $end_time"
                
                # Increment counter
                ((run_counter++))
                
                # Optional: Add a small delay between runs
                sleep 5
                
            done
        done
    done
done

echo "============================================"
echo "All 96 experiments completed!"
echo "Master log: $master_log"
echo "Individual logs: $experiment_log_dir/run_*.log"

# Generate summary report
echo ""
echo "=== EXPERIMENT SUMMARY ==="
echo "Total runs: 96"
echo "Success: $(grep -c "SUCCESS" "$master_log")"
echo "Failed Training: $(grep -c "FAILED_TRAINING" "$master_log")"
echo "Failed Evaluation: $(grep -c "FAILED_EVALUATION" "$master_log")"
echo "Failed Aggregation: $(grep -c "FAILED_AGGREGATION" "$master_log")"

# Create a summary CSV with just the results
summary_file="$experiment_log_dir/experiment_summary.csv"
echo "subject_key,noise_addition,token_replace_prob,token_k_neighbours,status,save_dir" > "$summary_file"
tail -n +2 "$master_log" | cut -d',' -f2,3,4,5,10,6 >> "$summary_file"

echo "Summary saved to: $summary_file"