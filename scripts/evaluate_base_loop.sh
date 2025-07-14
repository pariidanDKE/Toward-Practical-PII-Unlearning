#!/bin/bash

### This script is used to evaluate the full model without any Unlearning on it.
export BNB_CUDA_VERSION=121
export dataset="PII"
export MASTER_PORT=18765
# export model="llama3.1-8b"  # [llama3.1-8b, phi_chat, phi3-5-mini-instruct, llama2-7b_noanswerspace]
export batch_size=64

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget10"
export use_lora=False
model_list=("qwen2.5-3b" "qwen2.5-1.5b")

for model in "${model_list[@]}"; do
   export model_family=$model

   if [[ $model == "qwen2.5-3b" ]]; then
        export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_${model}_B32_G4_E5_lr2e-5_ComprehensiveQA_PadToken"
   elif [[ $model == "qwen2.5-1.5b" ]]; then
       export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_${model}_B16_G8_E5_lr2e-5_ComprehensiveQA_PadToken"
   else
       echo "Unsupported model: $model"
       continue
   fi
   echo "Evaluating model: $model"

   
   # -------- Evaluate Model --------
   python evaluate_PII.py --config-name=eval_pii.yaml \
       model_family=$model dataset=$dataset \
       split=$split batch_size=$batch_size \
       model_path=$save_dir forget_loss=$forget_loss \
       generation.max_length=200 \
       use_lora=$use_lora \
       save_dir=$save_dir/eval_results

   # -------- Aggregate Evaluation --------
   python aggregate_eval_stat.py \
       ckpt_result=$save_dir/eval_results/eval_log_aggregated.json \
       method_name=$forget_loss \
       save_file=$save_dir/eval_results/eval.csv \
       excel_file_path=$save_dir/eval_results/eval.xlsx \
       submitted_by=who

   echo "--------------------------------------------"
done

echo "All evaluations completed!"