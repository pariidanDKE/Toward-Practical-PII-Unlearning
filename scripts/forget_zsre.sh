#!/bin/sh
export dataset="ZSRE";  
export MASTER_PORT=18765;  
export split=forget;    
export model=phi;   # [phi, llama2-7b]
export num_epochs=5;
export batch_size=4;
export gradaccum=16;
export cache=$PWD/cache;
export bf16=false;
export CUDA_VISIBLE_DEVICES=1;
export retain_weight=1;
export lr=2e-5;
# "grad_ascent" "grad_ascent+kl" "grad_ascent+gd" 
# "dpo" "dpo+kl" "dpo+gd" "npo" "npo+kl" "npo+gd"
# "task_vector" "ULD" "WHP" "icl" "ours"
export Forget_Loss=("ours");  
export Types=(inverse);  # inverse; onehop; subject_replace
for type in "${Types[@]}"
do
    export forget_data_path=$PWD/data/${dataset}/${type};
    for forget_loss in "${Forget_Loss[@]}"
    do
        export save_dir=$PWD/our_experiment/${dataset}/${type}/${model}/${forget_loss}_E${num_epochs}_B${batch_size}_G${gradaccum}_lr${lr}_W${retain_weight};
        if [[ ${forget_loss} != "icl" ]];
        then
        python forget.py --config-name=forget.yaml \
            dataset=$dataset split=${split} \
            forget_data_path=${forget_data_path} \
            retain_data_path=${forget_data_path} \
            forget_loss=${forget_loss} batch_size=${batch_size} \
            retain_weight=${retain_weight} bf16=${bf16} \
            gradient_accumulation_steps=${gradaccum} model_family=${model} lr=${lr} \
            save_dir=$save_dir cache_dir=$cache num_epochs=${num_epochs};
        fi
        python evaluate_${dataset}.py --config-name eval_zsre_${type}.yaml \
            model_family=$model dataset=${dataset} \
            split=${split} batch_size=4 \
            model_path=$save_dir bf16=${bf16}  \
            generation.max_length=200 \
            save_dir=$save_dir/eval_${type}_results;

        python aggregate_eval_stat.py \
            retain_result=$PWD/data/retain_reference/${dataset}/all_retain/${model}_eval_results/${type}_eval_log_aggregated.json \
            ckpt_result=$save_dir/eval_${type}_results/eval_log_aggregated.json \
            method_name=$forget_loss \
            save_file=$save_dir/eval_${type}_results/eval.csv \
            excel_file_path=$save_dir/eval_${type}_results/eval.xlsx \
            submitted_by=who;
    done
done