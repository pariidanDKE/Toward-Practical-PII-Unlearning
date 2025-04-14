#!/bin/sh
export BNB_CUDA_VERSION=121;
export dataset="TOFU";  
export MASTER_PORT=18765;  
export splits=(forget01);  # [forget01, forget05, forget10]  
export model=llama2-7b;   # [phi, llama2-7b]
export num_epochs=5;
export batch_size=4;
export gradaccum=3;
export cache=$PWD/cache;
export forget_data_path=$PWD/data/${dataset};
export retain_weight=1;
export lr=2e-5;

export CUDA_VISIBLE_DEVICES=0; # snivdia-smi shows my gpu is at 0
export forget_loss="PerMU";
export save_dir=$PWD/experiment/${dataset}/${model}/${split}/${forget_loss}_E${num_epochs}_B${batch_size}_G${gradaccum}_lr${lr}_W${retain_weight};
export split='forget01';

# export LoRA.r=256
# export LoRA.alpha=512


#export run_name="EvaluateLoRA_r${LoRA_r}_a${LoRA_alpha}_ConstantLRWithWarmup"

export run_name="FullPerMU_WarmupConstantLR"
export project_name='EvaluateLoRA'
export use_lora=False;
export use_quantization=False;
       
python forget.py --config-name=forget.yaml \
    dataset=$dataset split=${split} \
    forget_data_path=${forget_data_path} \
    retain_data_path=${forget_data_path} \
    forget_loss=${forget_loss} batch_size=${batch_size} \
    retain_weight=${retain_weight} \
    gradient_accumulation_steps=${gradaccum} model_family=${model} lr=${lr} \
    save_dir=$save_dir cache_dir=$cache num_epochs=${num_epochs} \
    use_lora=${use_lora} \
    use_quantization=${use_quantization} \
    project_name=${project_name} \
    run_name=$run_name





