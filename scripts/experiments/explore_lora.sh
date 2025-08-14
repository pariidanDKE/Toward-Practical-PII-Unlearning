#!/bin/bash
### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
### Modified to loop through different LoRA ranks with alpha = 2x rank
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765

export model=llama3.1-8b;   # [phi, llama2-7b]
export num_epochs=8
export train_batch_size=16 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export eval_batch_size=64
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5
export seed="None"
export cache_path="/projects/0/hpmlprjs/LLM/danp/UGBench/models/neighbourhood_cache/${model}/${model}.pkl"

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export project_name="CompareLoRA_PaperConfig"
export use_lora=True  # Enable LoRA
export use_quantization=True
export forget_data_path="$PWD/data/${dataset}"
    
export remove_model_tensors=True
export token_level=True
export optimal_neighbours_generation=True
export optimizer="paged_adamw_8bit"

# Define LoRA ranks to loop through

lora_ranks=(32 64 128 256 512 1024)

### new setup
export use_quantization=False

# Loop through different LoRA ranks
for use_quantization in False; do
    for rank in "${lora_ranks[@]}"; do
        export LoRA_r=$rank
        export LoRA_alpha=$((rank * 2))  # Alpha = 2x rank
        #run_range="{1..5}"
        run_range="1"

        
        # # Set run range based on quantization setting
        # if [ "$use_quantization" = "True" ]; then
        # else
        #     run_range="1"
        # fi
        
        for run_id in $(eval echo $run_range); do
            echo "============================================"
            echo "Starting training with LoRA rank: $LoRA_r, alpha: $LoRA_alpha, run: $run_id"
            echo "============================================"

            if [ "$use_quantization" = "True" ]; then
                export run_name="QLoRA_PII_${model}_E${num_epochs}_B${train_batch_size}_r${LoRA_r}_a${LoRA_alpha}_token_level${token_level}_run${run_id}"
            else
                export run_name="LoRA_PII_${model}_E${num_epochs}_B${train_batch_size}_r${LoRA_r}_a${LoRA_alpha}_token_level${token_level}_run${run_id}"
            fi
            export save_dir="$PWD/experiment/${dataset}/${model}/${split}/_AllExperiments/LoRA/$run_name"

            echo "Running model with token_level=${token_level}, LoRA rank=${LoRA_r}, LoRA alpha=${LoRA_alpha}"
            # Set optimal_neighbours_generation based on token_level value
            if [[ "$token_level" == "True" ]]; then
                export optimal_neighbours_generation=True
            else
                export optimal_neighbours_generation=False
            fi
            #-------- Run Training --------
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
                token_level=$token_level \
                optimizer=$optimizer \
                optimal_neighbours_generation=$optimal_neighbours_generation \
                LoRA.r=$LoRA_r \
                LoRA.alpha=$LoRA_alpha \
                cache_path=$cache_path \
                seed=$seed \

            # # -------- Evaluate Model --------
            # echo "Starting evaluation for LoRA rank: $LoRA_r, alpha: $LoRA_alpha"
            # python evaluate_PII.py --config-name=eval_pii.yaml \
            #     model_family=$model dataset=$dataset \
            #     split=$split batch_size=$eval_batch_size \
            #     model_path=$save_dir forget_loss=$forget_loss \
            #     generation.max_length=200 \
            #     use_lora=$use_lora \
            #     save_dir=$save_dir/eval_results

            # # -------- Aggregate Evaluation --------
            # python aggregate_eval_stat.py \
            #     ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
            #     method_name=$forget_loss \
            #     save_file=$save_dir/eval_results/eval.csv \
            #     excel_file_path=$save_dir/eval_results/eval.xlsx \
            #     submitted_by=who \
            #     remove_model_tensors=$remove_model_tensors

            echo "Finished run $run_id for LoRA rank: $LoRA_r, alpha: $LoRA_alpha with ${num_epochs} epochs"
            echo "--------------------------------------------"
        done  # Close run_id loop
    done      # Close rank loop
done          # Close use_quantization loop

echo "Finished all LoRA rank experiments"
echo "============================================"