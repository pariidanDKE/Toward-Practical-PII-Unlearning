#!/bin/bash
### This script is used to evaluate the PerMU in-text method, as described in the "7.5 Discrete-Token Level Perturbation" section of the PerMU paper.
export dataset="PII"
export MASTER_PORT=18765

### Hyperparameters
export num_epochs=8
export batch_size=16 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=2
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5
export neftune_noise_alpha=False
export CUDA_VISIBLE_DEVICES=0

### Unlearning Params
export forget_loss="PerMU"
export project_name="SyntheticPII"
export use_quantization=False
export forget_data_path="$PWD/data/${dataset}"
export split="forget1"
export C=0.2
export P=0.8
export use_lora=False


### PerMutok Params
export in_text=False
export token_replace_prob=1
export token_k_neighbours=1
export subject_key='subject'
export use_adaptive_k=False
export match_first_char=True 
export perturb_logits_further=False
export optimal_neighbours_generation=True

### LogggingParams
export logging_subject_token_len=False
export remove_model_tensors=True

### Other Params
export BNB_CUDA_VERSION=121
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591

#model_list=("llama2-7b" "llama3-8b")
#model_list=("llama3-8b")
#model_list=("llama2-7b_ansspace")

model_list=("phi3-5-mini-instruct")

for model in "${model_list[@]}"; do
    if [[ "$model" == "llama3-8b" ]]; then
        export batch_size=8
        export eval_batch_size=32
    else
        export batch_size=8
        export eval_batch_size=64
    fi

    export run_name="FullFT_PII_${model}_E${num_epochs}_B${batch_size}_intext${in_text}_P${P}_C${C}_lr${lr}"
    export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"

    # echo "Running model with intext=${in_text}"
    #-------- Run Training --------
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
    subject_key=$subject_key \
    logging.subject_token_len=$logging_subject_token_len \
    use_adaptive_k=$use_adaptive_k \
    logging.corrupted_subjects=True \
    match_first_char=$match_first_char \
    optimal_neighbours_generation=$optimal_neighbours_generation \
    neftune_noise_alpha=$neftune_noise_alpha \
    perturb_logits_further=$perturb_logits_further \
    C=$C \
    P=$P \


    # -------- Evaluate Model --------
    python evaluate_PII.py --config-name=eval_pii_short.yaml \
        model_family=$model dataset=$dataset \
        split=$split batch_size=$eval_batch_size \
        model_path=$save_dir forget_loss=$forget_loss \
        generation.max_length=200 \
        use_lora=$use_lora \
        save_dir=$save_dir/eval_results \
        #data_path=$sample_data_path \

    # -------- Aggregate Evaluation --------
    python aggregate_eval_stat.py \
        ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
        method_name=$forget_loss \
        save_file=$save_dir/eval_results/eval.csv \
        excel_file_path=$save_dir/eval_results/eval.xlsx \
        submitted_by=who \
        remove_model_tensors=$remove_model_tensors

    echo "Finished run for full model with ${num_epochs} epochs"
    echo "--------------------------------------------"
done
echo "Finished all full model runs"
echo "============================================"