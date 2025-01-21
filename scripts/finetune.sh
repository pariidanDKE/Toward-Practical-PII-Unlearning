#!/bin/bash
export dataset="TOFU";   #[TOFU, Harry, ZSRE]
export master_port=18765;
export model=phi;   # [phi, llama2-7b]
export split=retain95;   
export data_path=$PWD/data/${dataset}/${split}.json;
export lr=3e-5;
export batch_size=4;
export GA=8;
export epoch=10;
export bf16=true;
export save_file=$PWD/save_model/${dataset}/${split}_${model}_B${batch_size}_G${GA}_E${epoch}_lr${lr}_2;
export CUDA_VISIBLE_DEVICES=1;

python finetune.py --config-name=finetune.yaml \
    batch_size=${batch_size} gradient_accumulation_steps=${GA} \
    dataset=${dataset} data_path=${data_path} lr=${lr} num_epochs=${epoch}\
    model_family=${model} save_dir=${save_file} bf16=${bf16};

if [[ ${dataset} = "TOFU" ]];
then
declare -A retain_splits;
retain_splits["retain99"]="forget01";
retain_splits["retain95"]="forget05";
retain_splits["retain90"]="forget10";

python evaluate_${dataset}.py \
    model_family=$model dataset=${dataset} \
    split=${retain_splits[${split}]} \
    model_path=$save_file batch_size=${batch_size} \
    generation.max_length=200 bf16=${bf16} \
    save_dir=$save_file/eval_results;

python aggregate_eval_stat.py \
    ckpt_result=$save_file/eval_results/eval_log_aggregated.json \
    method_name="finetune" \
    save_file=$save_file/eval_results/eval.csv \
    excel_file_path=$save_file/eval_results/eval.xlsx \
    submitted_by=who;

elif [[ ${dataset} = "Harry" ]];
then 
python evaluate_${dataset}.py \
    model_family=$model dataset=${dataset} \
    split=forget_all_subject batch_size=${batch_size}\
    model_path=$save_file \
    generation.max_length=200 bf16=${bf16} \
    save_dir=$save_file/eval_results;

python aggregate_eval_stat.py \
    ckpt_result=$save_file/eval_results/eval_log_aggregated.json \
    method_name="finetune" \
    save_file=$save_file/eval_results/eval.csv \
    excel_file_path=$save_file/eval_results/eval.xlsx \
    submitted_by=who;

elif [[ ${dataset} = "ZSRE" ]];
then 
export Types=(inverse onehop subject_replace);  # onehop subject_replace
for type in "${Types[@]}"
do
    python evaluate_${dataset}.py --config-name eval_${type}.yaml \
        model_family=$model dataset=${dataset} \
        split=forget batch_size=${batch_size} \
        model_path=$save_file bf16=${bf16} \
        generation.max_length=200 \
        save_dir=$save_file/eval_${type}_results; \

    python aggregate_eval_stat.py \
        ckpt_result=$save_file/eval_${type}_results/eval_log_aggregated.json \
        method_name="finetune" \
        save_file=$save_file/eval_${type}_results/eval.csv \
        excel_file_path=$save_file/eval_${type}_results/eval.xlsx \
        submitted_by=who;
done
fi

