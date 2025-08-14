# Standard library imports
import argparse
import copy
import json
import logging
import os
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Third-party imports
import hydra
import psutil
import torch
import transformers
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    set_seed
)

# Local modeling imports
import modelling.modelling_llama3
from modelling.modeling_llama import LlamaForCausalLM
from modelling.modeling_phi import PhiForCausalLM
from modelling.modeling_qwen2 import Qwen2ForCausalLM
from modelling.modelling_phi3 import Phi3ForCausalLM

# Local utility imports
from data_module import CommonForgetQA
from dataloader import CustomTrainerForgetting
from logging_utils import (
    get_config,
    init_config,
    init_logger,
    save_permu_metrics_to_json,
    should_log_stats,
    write_subject_corruption_info,
    write_subject_lengths
)
from permu_tok.corrupt_neighbourhood_generate import setup_optimized_tokenizer
from utils import get_model_identifiers_from_yaml


# ========================= UTILITY FUNCTIONS =========================

def check_memory(stage):
    """Monitor GPU and CPU memory usage at different stages."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"{stage}: GPU {allocated:.2f}GB")
    
    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1024**3
    print(f"{stage}: CPU {ram_gb:.2f}GB")


def find_all_linear_names(model):
    """Find all linear layer names for LoRA configuration."""
    print('Find linear layers for LoRA..')
    cls = torch.nn.Linear
    lora_module_names = set()
    
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    # Remove lm_head for 16-bit compatibility
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_params} || "
        f"trainable%: {100 * trainable_params / all_params:.2f}"
    )


def save_training_info(cfg, model, training_args, model_size, trainable_params, lora_target_modules, save_dir):
    """Save training configuration and model information to JSON."""
    info = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'model_size': model_size,
        'trainable_params': trainable_params,
        'lora_target_modules': lora_target_modules,
        'training_args': {
            'per_device_train_batch_size': training_args.per_device_train_batch_size,
            'per_device_eval_batch_size': training_args.per_device_eval_batch_size,
            'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
            'warmup_steps': training_args.warmup_steps,
            'max_steps': training_args.max_steps,
            'learning_rate': training_args.learning_rate,
            'bf16': training_args.bf16,
            'logging_steps': training_args.logging_steps,
            'logging_dir': training_args.logging_dir,
            'output_dir': training_args.output_dir,
            'optim': training_args.optim,
            'save_strategy': training_args.save_strategy,
            'save_steps': training_args.save_steps,
            'weight_decay': training_args.weight_decay,
            'eval_steps': training_args.eval_steps,
            'seed': training_args.seed
        }
    }
    
    save_path = os.path.join(save_dir, 'training_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"Training information saved to {save_path}")


# ========================= MODEL CREATION FUNCTIONS =========================

def get_model_class_from_name(model_name):
    """Determine the appropriate model class based on model name."""
    model_name = model_name.lower()
    
    if "phi-3.5" in model_name:
        print(f'Using Phi3ForCausalLM for {model_name}')
        return Phi3ForCausalLM
    elif "phi" in model_name:
        print(f'Using PhiForCausalLM for {model_name}')
        return PhiForCausalLM
    elif "llama-2" in model_name:
        print(f'Using LlamaForCausalLM for {model_name}')
        return LlamaForCausalLM
    elif "llama-3" in model_name:
        print(f'Using Llama3ForCausalLM for {model_name}')
        return modelling.modelling_llama3.LlamaForCausalLM
    elif "qwen" in model_name:
        print(f'Using Qwen2ForCausalLM for {model_name}')
        return Qwen2ForCausalLM
    else:
        return AutoModelForCausalLM


def create_quantization_config(use_quantization):
    """Create quantization configuration if needed."""
    if use_quantization:
        print('Adding quantization..')
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    return None


def setup_lora_model(model, cfg):
    """Setup LoRA configuration for the model."""
    print('Attaching LoRA...')
    target_modules = find_all_linear_names(model)
    
    peft_config = LoraConfig(
        r=cfg.LoRA.r,
        lora_alpha=cfg.LoRA.alpha,
        lora_dropout=cfg.LoRA.dropout,
        task_type=cfg.LoRA.task_type,
        target_modules=target_modules
    )
    
    model.enable_input_require_grads()
    model = get_peft_model(model, peft_config)
    
    return model, target_modules


def create_model(cfg, config, model_cfg, torch_dtype, causalLM):
    """Create and configure the main model."""
    if cfg.use_lora:
        quantization_config = create_quantization_config(cfg.use_quantization)
        
        model = causalLM.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=model_cfg["flash_attention2"] == "true",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            quantization_config=quantization_config
        )
        
        model, target_modules = setup_lora_model(model, cfg)
        
    elif "ULD" not in cfg.forget_loss:
        print(f'Config: {config}')
        print(f'Model Path: {cfg.model_path}')
        print(f'Torch dtype: {torch_dtype}')
        
        model = causalLM.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=model_cfg["flash_attention2"] == "true",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        if model.generation_config is None:
            model.generation_config = GenerationConfig.from_pretrained(cfg.model_path)
        
        target_modules = None
        model.generation_config.do_sample = True
    
    return model, target_modules


def create_oracle_model(cfg, config, model_cfg, torch_dtype):
    """Create oracle model if needed for certain loss functions."""
    if any(loss_type in cfg.forget_loss for loss_type in ["kl", "npo", "dpo"]):
        return AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=model_cfg["flash_attention2"] == "true",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
    return None


# ========================= SETUP FUNCTIONS =========================

def setup_environment_and_config(cfg):
    """Setup environment variables and configuration."""
    logger = init_logger(cfg)
    init_config(cfg)
    logger.info("Starting Forgetting Training...")

    # Get device information
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = 0
    
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        print(f'Local rank: {local_rank}')
    
    print(f"num_devices: {num_devices}")
    
    #env_seed =  cfg.seed if cfg.seed != "None" else None
    if cfg.seed=="None":
        print("No seed set, using random seed.")
    else:
        print(f"Setting seed: {cfg.seed}")
        set_seed(cfg.seed)

    return logger, num_devices, local_rank


def setup_model_config(cfg):
    """Setup model configuration and paths."""
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    
    # Set model path based on dataset if not provided
    if cfg.model_path is None:
        dataset_paths = {
        "TOFU": "tofu_target_model_path",
        "PII": "pii_target_model_path",
        "Harry": "harry_target_model_path",
        "ZSRE": "zsre_target_model_path"
    }
            
        required_path_key = dataset_paths.get(cfg.dataset)
        if required_path_key is None:
            raise ValueError(f"Unknown dataset: {cfg.dataset}")
        if required_path_key not in model_cfg:
            raise KeyError(f"Model config missing required path '{required_path_key}' for dataset '{cfg.dataset}'")
        
        cfg.model_path = model_cfg[required_path_key]
    return model_cfg, model_id


def setup_save_directory(cfg, local_rank):
    """Setup and validate save directory."""
    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    
    if local_rank == 0:
        if os.path.exists(cfg.save_dir) and not cfg.overwrite_dir:
            print("Directory already exists")
            exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
        
        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)


def setup_tokenizer(cfg, model_id, logger):
    """Setup and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    if cfg.optimal_neighbours_generation:
        logger.info("Setting up optimized tokenizer..")
        setup_optimized_tokenizer(
            tokenizer=tokenizer,
            memory_mode="unlimited",
            cache_path=cfg.cache_path
        )
        logger.info("Optimizer Successfully set up!")
    
    return tokenizer


def setup_dataset(cfg, tokenizer):
    """Setup training dataset."""
    max_length = 500
    
    # Determine retain split based on dataset
    if cfg.dataset in ["Harry", "ZSRE"]:
        retain_split = "retain"
    elif cfg.dataset in ["TOFU", "PII"]:
        retain_split = "retain" + str(100 - int(cfg.split.replace("forget", ""))).zfill(2)
    
    return CommonForgetQA(
        cfg.forget_data_path,
        cfg.retain_data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=max_length,
        split=cfg.split,
        retain_split=retain_split,
        loss_type=cfg.forget_loss,
        token_level=cfg.token_level,
        token_replace_prob=cfg.token_replace_prob,
        token_k_neighbours=cfg.token_k_neighbours,
        subject_key=cfg.subject_key,
        subject_noise_discrepancy_addition=cfg.subject_noise_discrepancy_addition
    )


def calculate_training_steps(cfg, torch_format_dataset, num_devices):
    """Calculate training steps and epochs."""
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    
    steps_per_epoch = len(torch_format_dataset) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    
    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    
    return batch_size, gradient_accumulation_steps, steps_per_epoch, max_steps


def create_training_arguments(cfg, batch_size, gradient_accumulation_steps, steps_per_epoch, max_steps):
    """Create training arguments configuration."""
    deepspeed_config = "config/ds_config.json" if cfg.use_deepspeed else None

    # Create the base arguments dictionary
    training_args = {
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "warmup_steps": max(0, steps_per_epoch),
        "max_steps": max_steps,
        "learning_rate": cfg.lr,
        "bf16": cfg.bf16,
        "bf16_full_eval": cfg.bf16,
        "logging_steps": 1,
        "logging_dir": f'{cfg.save_dir}/logs',
        "output_dir": cfg.save_dir,
        "optim": cfg.optimizer,
        "save_strategy": "no",
        "save_steps": steps_per_epoch,
        "deepspeed": deepspeed_config,
        "weight_decay": cfg.weight_decay,
        "eval_steps": steps_per_epoch,
        "disable_tqdm": False,
        "report_to": 'wandb',
        "lr_scheduler_type": 'constant_with_warmup',
        "ddp_find_unused_parameters": True,
    }

    # Only add seed if it's not None and not "None"
    if cfg.seed is not None and cfg.seed != "None":
        training_args["seed"] = cfg.seed

    return transformers.TrainingArguments(**training_args)
    
    # return transformers.TrainingArguments(
    #     per_device_train_batch_size=batch_size,
    #     per_device_eval_batch_size=batch_size,
    #     gradient_accumulation_steps=gradient_accumulation_steps,
    #     warmup_steps=max(0, steps_per_epoch),
    #     max_steps=max_steps,
    #     learning_rate=cfg.lr,
    #     bf16=cfg.bf16,
    #     bf16_full_eval=cfg.bf16,
    #     logging_steps=1,
    #     logging_dir=f'{cfg.save_dir}/logs',
    #     output_dir=cfg.save_dir,
    #     optim=cfg.optimizer,
    #     save_strategy="no",
    #     save_steps=steps_per_epoch,
    #     deepspeed=deepspeed_config,
    #     weight_decay=cfg.weight_decay,
    #     eval_steps=steps_per_epoch,
    #     seed=cfg.seed,
    #     disable_tqdm=False,
    #     report_to='wandb',
    #     lr_scheduler_type='constant_with_warmup',
    #     ddp_find_unused_parameters=True,  # DDP fix
    # )

# ========================= CLEANUP FUNCTIONS =========================

def cleanup_checkpoints(cfg, local_rank):
    """Clean up checkpoint directories to save space."""
    if local_rank == 0:
        save_path = Path(cfg.save_dir)
        
        # Remove global_step directories
        for checkpoint_dir in save_path.glob("checkpoint-*"):
            for global_step_dir in checkpoint_dir.glob("global_step*"):
                import shutil
                shutil.rmtree(global_step_dir)
        
        # Remove checkpoint directories
        for checkpoint_dir in save_path.glob("checkpoint-*"):
            import shutil
            shutil.rmtree(checkpoint_dir)


def save_final_model(cfg, model, tokenizer, trainer):
    """Save the final trained model and tokenizer."""
    if cfg.save_model and not cfg.eval_only:
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        
        if cfg.use_deepspeed:
            trainer.save_model(cfg.save_dir)
            # Clean up safetensors file if it exists
            safetensors_path = os.path.join(cfg.save_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                os.remove(safetensors_path)


# ========================= ARGUMENT PARSING =========================

def setup_argument_parsing():
    """Setup argument parsing to handle DeepSpeed's --local_rank argument."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--local_rank', type=int, default=0)
    args, remaining_args = parser.parse_known_args()
    
    # Remove --local_rank from sys.argv so Hydra doesn't see it
    sys.argv = [sys.argv[0]] + remaining_args
    
    return args

# ========================= MAIN FUNCTION =========================

@hydra.main(version_base=None, config_path="./config", config_name="forget")
def main(cfg):
    """Main training function."""
    
    # Setup environment and configuration
    logger, num_devices, local_rank = setup_environment_and_config(cfg)
    model_cfg, model_id = setup_model_config(cfg)
    setup_save_directory(cfg, local_rank)
    
    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
    )

    # Setup tokenizer and dataset
    tokenizer = setup_tokenizer(cfg, model_id, logger)
    torch_format_dataset = setup_dataset(cfg, tokenizer)
    
    # Calculate training parameters
    batch_size, gradient_accumulation_steps, steps_per_epoch, max_steps = calculate_training_steps(
        cfg, torch_format_dataset, num_devices
    )
    
    # Setup torch dtype
    torch_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    
    # Create training arguments
    training_args = create_training_arguments(
        cfg, batch_size, gradient_accumulation_steps, steps_per_epoch, max_steps
    )
    print(f"Trainer will use DeepSpeed: {training_args.deepspeed is not None}")
    print(f'Optimizer {training_args.optim}')
    
    # Setup model configuration
    try:
        config = AutoConfig.from_pretrained(model_id)
    except ValueError as e:
        # Handle old transformers version for Llama3.1 config
        if "rope_scaling" in str(e):
            config_path = os.path.join(model_cfg["t440_config"])
            config = AutoConfig.from_pretrained(config_path)
        else:
            raise e
    
    model_name = config._name_or_path.lower()
    print(f'Model Name: {model_name}')
    
    # Create models
    assistant_model = None
    oracle_model = None
    target_modules = None
    
    if "ULD" in cfg.forget_loss:
        logger.info("Initializing model...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            config=config,
            torch_dtype=torch_dtype,
            use_flash_attention_2=model_cfg["flash_attention2"] == "true",
            trust_remote_code=True
        )
    else:
        causalLM = get_model_class_from_name(model_name)
        model, target_modules = create_model(cfg, config, model_cfg, torch_dtype, causalLM)
        oracle_model = create_oracle_model(cfg, config, model_cfg, torch_dtype)
    
    # Enable gradient checkpointing if configured
    if model_cfg["gradient_checkpointing"] == "true":
        logger.info('Enable Gradient Checkpoint..')
        model.gradient_checkpointing_enable()
    
    # Setup trainer
    model.config.use_cache = False
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=torch_format_dataset.custom_data_collator_forget,
        oracle_model=oracle_model,
        assistant_model=assistant_model,
        forget_loss=cfg.forget_loss,
        eval_cfg=cfg.eval,
        retain_weight=cfg.retain_weight,
        C=cfg.C,
        P=cfg.P,
        token_level=cfg.token_level,
    )
    
    # Calculate model statistics and save training info
    model_size = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    save_training_info(cfg, model, training_args, model_size, trainable_params, target_modules, cfg.save_dir)
    
    # Run training or evaluation
    if cfg.eval_only:
        trainer.evaluate()
    else:
        logger.info("Training the model...")
        trainer.train()
    
    # Save metrics if configured
    if should_log_stats('permu_contrast_stats'):
        save_permu_metrics_to_json(save_dir=cfg.save_dir)
    
    # Save final model and clean up
    save_final_model(cfg, model, tokenizer, trainer)
    cleanup_checkpoints(cfg, local_rank)


if __name__ == "__main__":
    # Setup argument parsing for DeepSpeed compatibility
    setup_argument_parsing()
    main()