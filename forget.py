# Standard library imports
import copy
import json
import os
import psutil
from datetime import datetime
from pathlib import Path

# Third-party imports
import torch
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
import hydra
import transformers

# Local modeling imports
import modelling.modelling_llama3
from modelling.modeling_llama import LlamaForCausalLM
from modelling.modeling_phi import PhiForCausalLM
from modelling.modeling_qwen2 import Qwen2ForCausalLM
from modelling.modelling_phi3 import Phi3ForCausalLM

# Local utility imports
from permu_tok.corrupt_neighbourhood_generate import setup_optimized_tokenizer
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
from utils import get_model_identifiers_from_yaml


# def do_something(tensor, perturb_function):
#     # do stuff here
#     # some more stuff

#     # and then perturn
#     out = perturn_function(tensor)
#     return
# from functools import partial
# def perturb_randomly(tensor, mean_of_noise, std_of_noise):
#     dsadsada
#     return
# perturb_f = partial(perturb_randomly, mean=0, std=1)
# do_something(perturb_function=perturb_funtion)



def check_memory(stage):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"{stage}: GPU {allocated:.2f}GB")
    
    process = psutil.Process()
    ram_gb = process.memory_info().rss / 1024**3
    print(f"{stage}: CPU {ram_gb:.2f}GB")


def find_all_linear_names(model):
    print('Find linear layers for LoRA..')
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def save_training_info(cfg, model, training_args, model_size, trainable_params,lora_target_modules, save_dir):
# Create a dictionary with the details to be saved
    info = {
        'config': OmegaConf.to_container(cfg, resolve=True),
        'model_size': model_size,
        'trainable_params': trainable_params,
        'lora_target_modules' : lora_target_modules,
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
#            'evaluation_strategy': training_args.evaluation_strategy,
            'seed': training_args.seed
            }
        }
    
    # Save the information to a JSON file
    save_path = os.path.join(save_dir, 'training_info.json')
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"Training information saved to {save_path}")


import signal
import sys
import os
import time



import logging
import argparse
import sys

# Handle DeepSpeed's --local_rank argument
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--local_rank', type=int, default=0)
args, remaining_args = parser.parse_known_args()

# Remove --local_rank from sys.argv so Hydra doesn't see it
sys.argv = [sys.argv[0]] + remaining_args


@hydra.main(version_base=None, config_path="./config", config_name="forget")
def main(cfg):

    logger = init_logger(cfg)
    init_config(cfg)
    logger.info("Starting Forgetting Training...")


    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    local_rank = 0
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        print(f'Local rank: {local_rank}')
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    #os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        if cfg.dataset == "TOFU":
            cfg.model_path = model_cfg["tofu_target_model_path"]
        elif cfg.dataset == "PII": ## DP Addition
            cfg.model_path = model_cfg["pii_target_model_path"]
        elif cfg.dataset == "Harry":
            cfg.model_path = model_cfg["harry_target_model_path"]
        elif cfg.dataset == "ZSRE":
            cfg.model_path = model_cfg["zsre_target_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")
    # save cfg in cfg.save_dir
    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    
    if cfg.optimal_neighbours_generation:
        logger.info(f"Setting up optimized tokenizer..")
        setup_optimized_tokenizer(tokenizer=tokenizer,memory_mode="unlimited",cache_path=cfg.cache_path)
        logger.info(f"Optimizer Successfully set up!")


    max_length = 500
    if cfg.dataset == "Harry" or cfg.dataset == "ZSRE":
        retain_split = "retain"
    elif cfg.dataset == "TOFU" or cfg.dataset=='PII':
        retain_split = "retain" + str(100 - int(cfg.split.replace("forget", ""))).zfill(2)        
    torch_format_dataset = CommonForgetQA(cfg.forget_data_path, cfg.retain_data_path, tokenizer=tokenizer, model_family = cfg.model_family, \
                                          max_length=max_length, split=cfg.split, retain_split=retain_split, loss_type=cfg.forget_loss,in_text = cfg.in_text, \
                                          token_replace_prob=cfg.token_replace_prob,token_k_neighbours=cfg.token_k_neighbours,subject_key=cfg.subject_key,subject_noise_discrepancy_addition=cfg.subject_noise_discrepancy_addition)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    #### print all parameters for max_steps

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    
    if cfg.bf16 is True:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    # os.environ["WANDB_PROJECT"] = cfg.project_name
    # os.environ["WANDB_DIR"] = cfg.log_dir
    # wandb.init(name=cfg.run_name)

    if cfg.use_deepspeed:
        deepspeed_config = "config/ds_config.json"
    else:
        deepspeed_config = None

    optimizer = cfg.optimizer
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(0, steps_per_epoch),  
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=cfg.bf16,  
            bf16_full_eval=cfg.bf16, 
            logging_steps= 1, # max(1,max_steps//50),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim=optimizer,
            #save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_strategy = "no",
            save_steps=steps_per_epoch,
            deepspeed=deepspeed_config,
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            seed=cfg.seed,
            disable_tqdm=False, 
            report_to='wandb',
            lr_scheduler_type='constant_with_warmup' ,
            # ADD THESE DDP FIXES:
            ddp_find_unused_parameters=True,  # This is important
      
    )
    print(f"Trainer will use DeepSpeed: {training_args.deepspeed is not None}")




    #first get the base model architecture
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search(r"pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search(r"model-*\.safetensors", file):
            path_found = True
            break

    assistant_model = None
    oracle_model = None
    try:
        config = AutoConfig.from_pretrained(model_id)
    except ValueError as e:
        ## Account for old transformers version for Llama3.1 config
        if "rope_scaling" in str(e):
            config_path = os.path.join(model_cfg["t440_config"])
            config = AutoConfig.from_pretrained(config_path)
        else:
            raise e


    model_name = config._name_or_path.lower()

    if "ULD" in cfg.forget_loss:
        logger.info("Initializing model...")
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, \
            torch_dtype=torch_dtype, use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
            trust_remote_code=True)
        target_modules = None
        causalLM = AutoModelForCausalLM
    
    else:
        config = AutoConfig.from_pretrained(model_id)
        print("Loading from checkpoint")

        if "phi-3.5" in model_name:
            print(f'Using Phi3ForCausalLM for {model_name}')
            causalLM = Phi3ForCausalLM
        elif "phi" in model_name:
            print(f'Using PhiForCausalLM for {model_name}')
            causalLM = PhiForCausalLM
        elif "llama-2" in model_name:
            print(f'Using LlamaForCausalLM for {model_name}')
            causalLM = LlamaForCausalLM
        elif "llama-3" in model_name:
            print(f'Using Llama3ForCausalLM for {model_name}')
            causalLM = modelling_llama3.LlamaForCausalLM
        elif "qwen" in model_name:
            print(f'Using Qwen2ForCausalLM for {model_name}')
            causalLM = Qwen2ForCausalLM
        else:
            causalLM = AutoModelForCausalLM

    print(f'Model Name : {model_name}')
    if cfg.use_lora:
        if cfg.use_quantization:
            # DP : Add quantization
            print('Adding quantization..')
            quantization_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16) 
        else:
            quantization_config=None  

        model = causalLM.from_pretrained(cfg.model_path, config=config, \
        use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch_dtype, \
        trust_remote_code = True, \
        quantization_config = quantization_config  # DP : Add quantization
        #,device_map="auto"  # This handles device placement automatically
        )
        print('Attaching LoRA...')
        target_modules = find_all_linear_names(model)
        peft_config = LoraConfig(r=cfg.LoRA.r,lora_alpha=cfg.LoRA.alpha,lora_dropout=cfg.LoRA.dropout,task_type = cfg.LoRA.task_type,target_modules=target_modules)
        model.enable_input_require_grads()
        model = get_peft_model(model,peft_config)
        
    elif "ULD" not in cfg.forget_loss:
            print(f'Config : {config}')
            print(f'Model Path: {cfg.model_path}')
            logger.info('Initializing Model Again..')
            print(f'Torch dtype: {torch_dtype}')
            model = causalLM.from_pretrained(cfg.model_path, config=config, \
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", 
            torch_dtype=torch_dtype, \
            trust_remote_code = True, \
            #device_map=device_map  # This handles device placement automatically
            )
            if model.generation_config is None:
                model.generation_config = GenerationConfig.from_pretrained(cfg.model_path)
            target_modules=None
            model.generation_config.do_sample = True

    if "kl" in cfg.forget_loss or "npo" in cfg.forget_loss or "dpo" in cfg.forget_loss: 
        oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, \
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
            torch_dtype=torch_dtype, trust_remote_code = True)
        

    if model_cfg["gradient_checkpointing"] == "true":
        logger.info('Enable Gradient Checkpoint..')
        model.gradient_checkpointing_enable()
        #model.gradient_checkpointing_enable()


    #model = model.to(device)
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,               
        args=training_args,
        data_collator=torch_format_dataset.custom_data_collator_forget,
        oracle_model = oracle_model,
        assistant_model = assistant_model,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
        retain_weight = cfg.retain_weight,
        C = cfg.C,
        P = cfg.P,
        in_text = cfg.in_text,
    )
    # Before creating the trainer
    print(f'Optimizer {training_args.optim}')

    model.config.use_cache = False

    model_size = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Save the training info to a json file

    #print_trainable_parameters(model)
    save_training_info(cfg, model, training_args, model_size, trainable_params,target_modules, cfg.save_dir)
    if cfg.eval_only:
        trainer.evaluate()
    else:
        logger.info("Training the model...")
        trainer.train()
    
    if should_log_stats('permu_contrast_stats'):
        save_permu_metrics_to_json(save_dir=cfg.save_dir)
    
    # save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        
        if cfg.use_deepspeed:
            trainer.save_model(cfg.save_dir) ##### deepspeed messes up
            safetensors_path = os.path.join(cfg.save_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                os.remove(safetensors_path)

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                # delete the directory
                import shutil
                shutil.rmtree(global_step_dir)
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            # delete the directory
            import shutil
            shutil.rmtree(file)

def init_small_llm(origin_config, num_layer, device, hparams=None, base_llm=None, saved_path=None):
    config = copy.deepcopy(origin_config)
    config.num_hidden_layers = num_layer
    model = AutoModelForCausalLM.from_config(
        config,
        use_flash_attention_2=False, 
        torch_dtype=torch.bfloat16, 
    ).to('cpu')

    if base_llm is not None:
        copy_weights(base_llm, model)
        
    if saved_path is not None:
        os.makedirs(saved_path, exist_ok=True)
        model.save_pretrained(saved_path)

    return model

def copy_weights(base_llm, model):
    config = model.config
    name = model.config._name_or_path.lower()
    if ('llama' in name) or ('zephyr' in name) or ('mistral' in name):
        print(f"Copying {name} first layer: {config.num_hidden_layers}")
        model.model.embed_tokens.load_state_dict(
            base_llm.model.embed_tokens.state_dict()
        )
        model.model.norm.load_state_dict(
            base_llm.model.norm.state_dict()
        )
        for layer_num in range(config.num_hidden_layers):
            model.model.layers[layer_num].load_state_dict(
                base_llm.model.layers[layer_num].state_dict()
            )
        model.lm_head.load_state_dict(
            base_llm.lm_head.state_dict()
        )
        return model
    else:
        raise ValueError(f"Unsupported model: {name}")


if __name__ == "__main__":
    main()