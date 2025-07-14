# #!/bin/bash
#!/bin/bash
### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
### Modified to loop through different splits with 5 runs per configuration
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama3.1-8b;   # [phi, llama2-7b]
export optimizer="paged_adamw_8bit"
export forget_data_path="$PWD/data/${dataset}"

export num_epochs=8
export batch_size=16
export eval_batch_size=64
export gradaccum=2
export P=1.2

export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export project_name="CompareSplits_LLm3.1"
export remove_model_tensors=True

## PerMU in-text params
export in_text=False
export permu_contrast_stats=True
export optimal_neighbours_generation=True

# Define arrays for splits and runs
splits=("forget1" "forget10"  "forget5" "forget25" "forget50")
runs=(1 2 3 4 5)

# Loop through different splits
for split in "${splits[@]}"; do
    echo "============================================"
    echo "Starting experiments for Split: $split"
    echo "============================================"
    
    # Loop through 5 runs per split
    for run in "${runs[@]}"; do
        echo "--------------------------------------------"
        echo "Starting run $run for Split: $split"
        echo "--------------------------------------------"
        
        export run_name="Splits_${model}_E${num_epochs}_B${batch_size}_intext${in_text}_${split}_run${run}"

        if [ "$in_text" = "True" ]; then
            export optimal_neighbours_generation=True
        else
            export optimal_neighbours_generation=False
        fi

        export save_dir="$PWD/experiment/${dataset}/${model}/SplitExperiment/OriginalSplit/${split}/$run_name"
        export cache_path="/projects/0/hpmlprjs/LLM/danp/UGBench/models/neighbourhood_cache/${model}/${model}.pkl"

        echo "Running model with intext=${in_text}"
        echo "Save directory: $save_dir"

        #-------- Run Training --------
        python forget.py --config-name=forget_pii.yaml \
            dataset=$dataset split=$split \
            forget_data_path=$forget_data_path \
            retain_data_path=$forget_data_path \
            forget_loss=$forget_loss batch_size=$batch_size \
            retain_weight=$retain_weight \
            gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
            save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
            project_name=$project_name \
            run_name=$run_name \
            in_text=$in_text \
            optimal_neighbours_generation=$optimal_neighbours_generation \
            logging.permu_contrast_stats=$permu_contrast_stats \
            cache_path=$cache_path \
            optimizer=$optimizer \
            P=$P \

        # -------- Evaluate Model --------
        python evaluate_PII.py --config-name=eval_pii.yaml \
            model_family=$model dataset=$dataset \
            split=$split batch_size=$eval_batch_size \
            model_path=$save_dir forget_loss=$forget_loss \
            generation.max_length=200 \
            save_dir=$save_dir/eval_results \

        # -------- Aggregate Evaluation --------
        python aggregate_eval_stat.py \
            ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
            method_name=$forget_loss \
            save_file=$save_dir/eval_results/eval.csv \
            excel_file_path=$save_dir/eval_results/eval.xlsx \
            submitted_by=who \
            remove_model_tensors=$remove_model_tensors

        echo "Finished run $run for Split: $split with ${num_epochs} epochs"
        echo "--------------------------------------------"
    done
    
    echo "Finished all 5 runs for Split: $split"
    echo "============================================"
done

echo "Finished all experiments for all splits"
echo "============================================"


# ### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
# ### Modified to loop through different splits and random_splits
# export BNB_CUDA_VERSION=121
# export dataset="PII"
# export MASTER_PORT=18765
# export model=llama3.1-8b;   # [phi, llama2-7b]
# export optimizer="paged_adamw_8bit"

# export num_epochs=8
# export batch_size=16
# export eval_batch_size=64
# export gradaccum=2

# export cache="$PWD/cache"
# export retain_weight=1
# export lr=1e-5

# export CUDA_VISIBLE_DEVICES=0
# export forget_loss="PerMU"
# export project_name="CompareSplits_LLm3.1"
# export remove_model_tensors=True

# ## PerMU in-text params
# export in_text=True
# export permu_contrast_stats=True
# export optimal_neighbours_generation=True

# # Define arrays for splits and random_splits
# splits=("forget1" "forget10"  "forget5" "forget25" "forget50")
# random_splits=(1 2 3 4 5 6 7 8 9)

# # Loop through different random_splits
# for random_split in "${random_splits[@]}"; do
#     echo "============================================"
#     echo "Starting experiments for Random Split: $random_split"
#     echo "============================================"
    
#     # Loop through different splits
#     for split in "${splits[@]}"; do

#         echo "--------------------------------------------"
#         echo "Starting training with Split: $split, Random Split: $random_split"
#         echo "--------------------------------------------"
#         export run_name="Splits_${model}_E${num_epochs}_B${batch_size}_intext${in_text}_v${random_split}"


#         if [ "$in_text" = "True" ]; then
#             export optimal_neighbours_generation=True
#         else
#             export optimal_neighbours_generation=False
#         fi

#         export save_dir="$PWD/experiment/${dataset}/${model}/SplitExperiment/${split}/$run_name"
#         export cache_path="/projects/0/hpmlprjs/LLM/danp/UGBench/models/neighbourhood_cache/${model}/${model}.pkl"

        
#         echo "Running model with intext=${in_text}"
#         echo "Save directory: $save_dir"

#         export forget_data_path="$PWD/data/PII/SplitExperiment/Experiment$random_split"

#         #-------- Run Training --------
#         python forget.py --config-name=forget_pii_splitexperiment.yaml \
#             dataset=$dataset split=$split \
#             forget_data_path=$forget_data_path \
#             retain_data_path=$forget_data_path \
#             forget_loss=$forget_loss batch_size=$batch_size \
#             retain_weight=$retain_weight \
#             gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
#             save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
#             project_name=$project_name \
#             run_name=$run_name \
#             in_text=$in_text \
#             optimal_neighbours_generation=$optimal_neighbours_generation \
#             logging.permu_contrast_stats=$permu_contrast_stats \
#             random_split=$random_split \
#             cache_path=$cache_path \
#             optimizer=$optimizer \

#         # -------- Evaluate Model --------
#         echo "Starting evaluation for Split: $split, Random Split: $random_split"
#         python evaluate_PII.py --config-name=eval_pii.yaml \
#             model_family=$model dataset=$dataset \
#             split=$split batch_size=$eval_batch_size \
#             model_path=$save_dir forget_loss=$forget_loss \
#             generation.max_length=200 \
#             save_dir=$save_dir/eval_results \
#             random_split=$random_split \

#         # -------- Aggregate Evaluation --------
#         python aggregate_eval_stat.py \
#             ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
#             method_name=$forget_loss \
#             save_file=$save_dir/eval_results/eval.csv \
#             excel_file_path=$save_dir/eval_results/eval.xlsx \
#             submitted_by=who \
#             remove_model_tensors=$remove_model_tensors

#         echo "Finished run for Split: $split, Random Split: $random_split with ${num_epochs} epochs"
#         echo "--------------------------------------------"
#     done
    
#     echo "Finished all splits for Random Split: $random_split"
#     echo "============================================"
# done

# echo "Finished all experiments for all Random Splits"
# echo "============================================"