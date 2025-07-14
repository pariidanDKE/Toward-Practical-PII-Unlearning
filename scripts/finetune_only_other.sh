# #!/bin/bash

# This script simply finetunes the model on a given dataset.
export dataset="PII";   # [TOFU, Harry, ZSRE]
export master_port=18765;
export model=qwen2.5-3b;  # [llama3.1-8b, qwen2.5-1.5b, phi_chat, phi3-5-mini-instruct, llama2-7b_noanswerspace]
export debug=false;

export split=full_with_qa;    
export lr=2e-5;
export batch_size=64;
export GA=2;
export epoch=7;
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export use_deepspeed=false;

export CUDA_VISIBLE_DEVICES=0
export BNB_CUDA_VERSION=126
export data_path=$PWD/data/${dataset}/${split}.json;
export save_file=$PWD/save_model/${dataset}/${split}_${model}_B${batch_size}_G${GA}_E${epoch}_lr${lr}_ComprehensiveQA;

export data_path="$PWD/data/${dataset}/${split}.json"
export run_name="FineTune_${dataset}_${model}_B${batch_size}_G${GA}_E${epoch}_lr${lr}"

export project_name="Qwen-3B"
export file_path="/projects/0/hpmlprjs/LLM/danp/UGBench/finetune.py"

python ${file_path} --config-name=finetune.yaml \
    batch_size=${batch_size} gradient_accumulation_steps=${GA} \
    dataset=${dataset} data_path=${data_path} lr=${lr} num_epochs=${epoch}\
    model_family=${model} save_dir=${save_file} \
    project_name=${project_name} \
    run_name=$run_name \
    use_deepspeed=${use_deepspeed} \
    debug=${debug} \
