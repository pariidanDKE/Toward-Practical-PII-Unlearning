#!/bin/bash
export BNB_CUDA_VERSION=121
export dataset="TOFU"
export MASTER_PORT=18765
export model="llama2-7b"
export num_epochs=5
export batch_size=8
export gradaccum=2
export cache="$PWD/cache"
export forget_data_path="$PWD/data/${dataset}"
export retain_weight=1
export lr=2e-5

export CUDA_VISIBLE_DEVICES=0
export forget_loss="PerMU"
export split="forget01"
export project_name="EvaluateLoRA"
export use_lora=False
export use_quantization=False


export run_name="EvaluateLoRA_NoUnlearn_WarmupConstantLR"
export save_dir="/projects/0/hpmlprjs/LLM/danp/UGBench/locuslab/tofu_ft_llama2-7b"
#export save_dir="$PWD/experiment/${dataset}/${model}/${split}/$run_name"


# -------- Evaluate Model --------
python evaluate_${dataset}.py \
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

echo "Finished run for r=${LoRA_r}, alpha=${LoRA_alpha}"
echo "--------------------------------------------"

