#!/bin/bash

#### Failed attampt to extract perturbed subjects from model logits


export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=8
export batch_size=4 ## Should increase the batch size to 8 (Would just make it all faster, no other difference)
export gradaccum=4
export cache="$PWD/cache"
export retain_weight=1
export lr=1e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
#export forget_data_path="$PWD/data/PII"  
#export retain_data_path="$PWD/data/PII"
export forget_data_path="$PWD/data/${dataset}"

retain_splits["forget01"]="retain99"
retain_splits["forget05"]="retain95"
retain_splits["forget10"]="retain90"

export save_dir="$PWD/data/${dataset}/"

python extract_perturbed_subjects.py --config-name=forget_pii.yaml \
    dataset=$dataset split=$split \
    forget_data_path=$forget_data_path \
    retain_data_path=$forget_data_path \
    forget_loss=$forget_loss batch_size=$batch_size \
    retain_weight=$retain_weight \
    gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
    save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \


