#!/bin/bash

## This script runs both training and evaluation for PerMU in-text method
## Runs 1-5 (5 total runs) without configuration variations
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=8
export train_batch_size=16  # Updated from 16 as per your note
export eval_batch_size=64  # Updated from 16 as per your note
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="PerMU_10Runs"
export use_lora=False
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"
export num_runs=10  # Updated: now running 5 runs (1-5)
export logging_permu_contrast_stats=True

## PerMU in-text base params
export in_text=False
export logging_timestats=False
export remove_model_tensors=True 
export logging_corrupted_subjects=False

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

experiment_log_dir="$PWD/experiment_logs/permu_runs_1-5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$experiment_log_dir"


echo "Starting runs 1-5 experiment suite..."
echo "Experiment logs will be saved to: $experiment_log_dir"
echo "============================================"

# Run iterations 1-10
for run_num in $(seq 5 10); do
    final_status="UNKNOWN"

    # Generate run name with all parameters
    export run_name="${project_name}_${model}_E${num_epochs}_B${train_batch_size}_run${run_num}_Extraction"
    export save_dir="$PWD/experiment/${dataset}/${model}/${split}/_AllExperiments/ExtractionAttacks/PerMU/$run_name"
    
    # Create individual log file for this run
    run_log="$experiment_log_dir/run_${run_num}.log"
    
    echo ""
    echo "----------------------------------------"
    echo "Starting Run $run_num/5"
    echo "Run Name: $run_name"
    echo "Save Dir: $save_dir"
    echo "Log File: $run_log"
    
    # Record start time
    start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Run the training and evaluation pipeline with logging
    {
        echo "=== TRAINING PHASE ==="
        
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
            logging.time_stats=$logging_timestats \
            logging.corrupted_subjects=$logging_corrupted_subjects \
            logging.permu_contrast_stats=$logging_permu_contrast_stats 


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
                # Evaluation
                python evaluate_PII.py --config-name=eval_pii_short.yaml \
                    model_family=$model dataset=$dataset \
                    split=$split batch_size=$eval_batch_size \
                    model_path=$save_dir forget_loss=$forget_loss \
                    generation.max_length=200 \
                    use_lora=$use_lora \
                    save_dir=$save_dir/eval_results
                
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
                    submitted_by=10_run_experiment \
                    remove_model_tensors=$remove_model_tensors

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
        
    } 2>&1 | tee "$run_log"
    
    # Record end time and status
    end_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Update master log with completion status
    echo "Run $run_num completed with status: $final_status"
    echo "End time: $end_time"
    
    # Optional: Add a small delay between runs
    sleep 20
done

echo "============================================"
echo "Runs 1-5 completed!"
echo "Total runs executed: $num_runs"