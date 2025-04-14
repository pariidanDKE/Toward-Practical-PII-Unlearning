#!/bin/sh
export CUDA_VISIBLE_DEVICES=0;

export dataset="TOFU";  
export MASTER_PORT=18765;  
export splits=(forget01);  # [forget01, forget05, forget10]  
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=5;
export batch_size=8;
export gradaccum=8;
export cache=$PWD/cache;
export forget_data_path=$PWD/data/${dataset};
export retain_weight=1;
export lr=2e-5;

export use_lora=False;

# unlearning methods include: ["grad_ascent" "grad_ascent+kl" "grad_ascent+gd" 
# "dpo" "dpo+kl" "dpo+gd" "npo" "npo+kl" "npo+gd"
# "task_vector" "ULD" "WHP" "icl" "PerMU"]
export Forget_Loss=("PerMU");  
for split in "${splits[@]}"
do
    for forget_loss in "${Forget_Loss[@]}"
    do
        export save_dir="${save_dir:-MustDeclareModelPathWith:save_dir}"

        python evaluate_${dataset}.py \
            model_family=$model dataset=${dataset} \
            split=${split} batch_size=${batch_size} \
            model_path=$save_dir forget_loss=${forget_loss} \
            generation.max_length=200 \
            use_lora=${use_lora} \
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
