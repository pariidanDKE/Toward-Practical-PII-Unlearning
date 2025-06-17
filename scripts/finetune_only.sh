#!/bin/bash

# This script simply finetunes the model on a given dataset.
export dataset="PII";   # [TOFU, Harry, ZSRE]
export master_port=18765;
export model=llama2-7b_noanswerspace;   # [[phi, llama2-7b,phi_chat,llama2-7b_nonchat]
#export model=phi;   # [phi, llama2-7b]

export split=retain_and_test_retain;    
export lr=2e-5;
export batch_size=32;
export GA=4;
export epoch=5;

export CUDA_VISIBLE_DEVICES=0;
export data_path=$PWD/data/${dataset}/${split}.json;
export save_file=$PWD/save_model/${dataset}/${split}_${model}_B${batch_size}_G${GA}_E${epoch}_lr${lr}_ComprehensiveQA;


export data_path="$PWD/data/${dataset}/${split}.json"
export run_name="FineTune_${dataset}_${model}_B${batch_size}_G${GA}_E${epoch}_lr${lr}"

export project_name="FT_Generalizable"
export file_path="/projects/0/hpmlprjs/LLM/danp/UGBench/finetune.py"

python ${file_path} --config-name=finetune.yaml \
    batch_size=${batch_size} gradient_accumulation_steps=${GA} \
    dataset=${dataset} data_path=${data_path} lr=${lr} num_epochs=${epoch}\
    model_family=${model} save_dir=${save_file} \
    project_name=${project_name} \
    run_name=$run_name