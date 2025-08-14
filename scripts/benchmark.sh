# #!/bin/bash
# ################### ARC Base Model #########################

# lm_eval --model hf \
#  --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct" \
#  --tasks arc_challenge \
#  --num_fewshot 25 \
#  --device auto \
#  --batch_size 64

# ################### ARC Fine-tune Model #########################

# ### benchmark unlearned
# export MODEL_PATH="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_llama3.1-8b_B32_G4_E5_lr2e-5_ComprehensiveQA"

# lm_eval --model hf \
#  --model_args "pretrained=$MODEL_PATH" \
#  --tasks arc_challenge \
#  --num_fewshot 25 \
#  --device auto \
#  --batch_size 64

################### ARC Unlearned Model #########################

# export dataset="PII"
# export MASTER_PORT=18765
# export forget_data_path="$PWD/data/${dataset}"

# ### Hyperparameters
# export cache="$PWD/cache"
# export retain_weight=1
# export num_epochs=8

# ### MULTI GPU PARAMS
# export CUDA_VISIBLE_DEVICES=0
# export use_deepspeed=False
# export batch_size=16
# export gradaccum=2
# export eval_batch_size=64

# export optimizer="paged_adamw_8bit"
# export seed="None"

# ### Unlearning Params
# export forget_loss="PerMU"
# export project_name="CompareLoRA_PaperConfig"
# export use_quantization=False
# export split="forget10"
# export remove_model_tensors=True

# ### Fixed Parameters
# C=0.1
# P=1.0
# model="llama3.1-8b"
# token_level=False
# export lr=1e-5
# run=1

# export cache_path="/projects/0/hpmlprjs/LLM/danp/UGBench/models/neighbourhood_cache/${model}/${model}.pkl"

# # Set optimal_neighbours_generation based on token_level value
# if [[ "$token_level" == "True" ]]; then
#     export optimal_neighbours_generation=True
#     P=1  # Use default P value when token_level=True
# else
#     export optimal_neighbours_generation=False
# fi

# export run_name="${model}_E${num_epochs}_B${batch_size}_C${C}_P_${P}_token_level${token_level}_run${run}_test"
# export save_dir="$PWD/experiment/${dataset}/${model}/${split}/_AllExperiments/PIIAnalysis/Reruns/$run_name"

# echo "Running experiment with C=${C}, P=${P}, token_level=${token_level}, run=${run}"

# #-------- Run Training --------
# python forget.py --config-name=forget_pii.yaml \
# dataset=$dataset split=$split \
# forget_data_path=$forget_data_path \
# retain_data_path=$forget_data_path \
# forget_loss=$forget_loss batch_size=$batch_size \
# retain_weight=$retain_weight \
# gradient_accumulation_steps=$gradaccum model_family=$model lr=$lr \
# save_dir=$save_dir cache_dir=$cache num_epochs=$num_epochs \
# use_quantization=$use_quantization \
# project_name=$project_name \
# run_name=$run_name \
# token_level=$token_level \
# optimal_neighbours_generation=$optimal_neighbours_generation \
# C=$C \
# P=$P \
# use_deepspeed=$use_deepspeed \
# optimizer=$optimizer \
# cache_path=$cache_path \
# seed=$seed

# # Check if training was successful
# if [ $? -ne 0 ]; then
#     echo "Training failed. Exiting..."
#     exit 1
# fi


export MODEL_PATH="/projects/0/hpmlprjs/LLM/danp/UGBench/experiment/PII/llama3.1-8b/forget10/_AllExperiments/PIIAnalysis/Reruns/llama3.1-8b_E8_B16_C0.1_P_1.0_token_levelFalse_run1_test"

# Run evaluation
lm_eval --model hf \
    --model_args "pretrained=$MODEL_PATH" \
    --tasks mmlu_pro \
    --num_fewshot 0 \
    --device auto \
    --batch_size 128