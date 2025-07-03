#!/bin/bash
### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
### Modified to loop through different splits and random_splits
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]

export num_epochs=8
export batch_size=8 ## Should increase the batch size to 16 (Would just make it all faster, no other difference)
export eval_batch_size=64
export gradaccum=2

export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export project_name="CompareSplits1_"
export use_lora=False  # Disable LoRA
export use_quantization=False
export remove_model_tensors=True

## PerMU in-text params
export in_text=False
export token_replace_prob=1
export token_k_neighbours=1
export subject_key='subject'
export use_adaptive_k=False
export match_first_char=True
export logging_corrupted_subjects=False
export optimal_neighbours_generation=True
export logging_timestats=False

# Define arrays for splits and random_splits
splits=("forget1" "forget10"  "forget5" "forget25" "forget50")
random_splits=(3 4 5 6 7 8 9)

# Loop through different random_splits
for random_split in "${random_splits[@]}"; do
    echo "============================================"
    echo "Starting experiments for Random Split: $random_split"
    echo "============================================"
    
    # Loop through different splits
    for split in "${splits[@]}"; do
        # Skip random_split 2 for forget1, forget5, and forget10
        # if [[ $random_split -eq 3 && ($split == "forget1" || $split == "forget5" || $split == "forget10") ]]; then
        #     echo "Skipping Split: $split for Random Split: $random_split (rule applied)"
        #     continue
        # fi

        if [[ $random_split -eq 3 && $split == "forget1" ]]; then
            echo "Skipping Split: $split for Random Split: $random_split (rule applied)"
            continue
        fi

        echo "--------------------------------------------"
        echo "Starting training with Split: $split, Random Split: $random_split"
        echo "--------------------------------------------"
        
        if [ "$in_text" = "True" ]; then
            export run_name="Splits_${model}_E${num_epochs}_B${batch_size}_intext${in_text}_v${random_split}"
        else
            export run_name="Splits_${model}_E${num_epochs}_B${batch_size}_intext${in_text}_v${random_split}"
        fi

        export save_dir="$PWD/experiment/${dataset}/${model}/SplitExperiment/${split}/$run_name"
        
        echo "Running model with intext=${in_text}"
        echo "Save directory: $save_dir"

        export forget_data_path="$PWD/data/PII/SplitExperiment/Experiment$random_split"

        #-------- Run Training --------
        python forget.py --config-name=forget_pii_splitexperiment.yaml \
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
            use_adaptive_k=$use_adaptive_k \
            logging.corrupted_subjects=$logging_corrupted_subjects \
            match_first_char=$match_first_char \
            optimal_neighbours_generation=$optimal_neighbours_generation \
            logging.time_stats=$logging_timestats \
            random_split=$random_split \
            # LoRA.r=$LoRA_r \
            # LoRA.alpha=$LoRA_alpha \

        # -------- Evaluate Model --------
        echo "Starting evaluation for Split: $split, Random Split: $random_split"
        python evaluate_PII.py --config-name=eval_pii_splitexperiment.yaml \
            model_family=$model dataset=$dataset \
            split=$split batch_size=$eval_batch_size \
            model_path=$save_dir forget_loss=$forget_loss \
            generation.max_length=200 \
            use_lora=$use_lora \
            save_dir=$save_dir/eval_results \
            random_split=$random_split \

        # -------- Aggregate Evaluation --------
        python aggregate_eval_stat.py \
            ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
            method_name=$forget_loss \
            save_file=$save_dir/eval_results/eval.csv \
            excel_file_path=$save_dir/eval_results/eval.xlsx \
            submitted_by=who \
            remove_model_tensors=$remove_model_tensors

        echo "Finished run for Split: $split, Random Split: $random_split with ${num_epochs} epochs"
        echo "--------------------------------------------"
    done
    
    echo "Finished all splits for Random Split: $random_split"
    echo "============================================"
done

echo "Finished all experiments for all Random Splits"
echo "============================================"