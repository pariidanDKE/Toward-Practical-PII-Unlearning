#!/bin/sh
export dataset="TOFU";  
export MASTER_PORT=18765;  
export splits=(forget01);  # [forget01, forget05, forget10]  
export model=phi;   # [phi, llama2-7b]
export num_epochs=5;
export batch_size=8;
export gradaccum=8;
export cache=$PWD/cache;
export forget_data_path=$PWD/data/${dataset};
export CUDA_VISIBLE_DEVICES=1;
export retain_weight=1;
export lr=2e-5;
# unlearning methods include: ["grad_ascent" "grad_ascent+kl" "grad_ascent+gd" 
# "dpo" "dpo+kl" "dpo+gd" "npo" "npo+kl" "npo+gd"
# "task_vector" "ULD" "WHP" "icl" "PerMU"]
export Forget_Loss=("grad_ascent");  
for split in "${splits[@]}"
do
    for forget_loss in "${Forget_Loss[@]}"
    do
        export save_dir=$PWD/experiment/${dataset}/${model}/${split}/${forget_loss}_E${num_epochs}_B${batch_size}_G${gradaccum}_lr${lr}_W${retain_weight};
        
        if [[ ${forget_loss} != "icl" ]];
        then
        python forget.py --config-name=forget.yaml \
            dataset=$dataset split=${split} \
            forget_data_path=${forget_data_path} \
            retain_data_path=${forget_data_path} \
            forget_loss=${forget_loss} batch_size=${batch_size} \
            retain_weight=${retain_weight} \
            gradient_accumulation_steps=${gradaccum} model_family=${model} lr=${lr} \
            save_dir=$save_dir cache_dir=$cache num_epochs=${num_epochs};
        fi
        python evaluate_${dataset}.py \
            model_family=$model dataset=${dataset} \
            split=${split} batch_size=4 \
            model_path=$save_dir forget_loss=${forget_loss} \
            generation.max_length=200 \
            save_dir=$save_dir/eval_results;
        declare -A retain_splits;
        retain_splits["forget01"]="retain99";
        retain_splits["forget05"]="retain95";
        retain_splits["forget10"]="retain90";

        python aggregate_eval_stat.py \
            ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
            method_name=$forget_loss \
            save_file=$save_dir/eval_results/eval.csv \
            excel_file_path=$save_dir/eval_results/eval.xlsx \
            submitted_by=who;
    done
done
