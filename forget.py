from data_module import CommonForgetQA
from dataloader import CustomTrainerForgetting
import torch
from transformers import AutoTokenizer, AutoConfig, set_seed
from transformers import AutoModelForCausalLM
import copy 
import hydra 
import transformers
import os
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
from utils import get_model_identifiers_from_yaml
from omegaconf import OmegaConf
from modeling_phi import PhiForCausalLM
from modeling_llama import LlamaForCausalLM

def find_all_linear_names(model):
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

@hydra.main(version_base=None, config_path="./config", config_name="forget")
def main(cfg):
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    local_rank = 0
    if os.environ.get('LOCAL_RANK') is not None:
        print("here")
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    set_seed(cfg.seed)

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        if cfg.dataset == "TOFU":
            cfg.model_path = model_cfg["tofu_target_model_path"]
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

    max_length = 500
    if cfg.dataset == "Harry" or cfg.dataset == "ZSRE":
        retain_split = "retain"
    elif cfg.dataset == "TOFU":
        retain_split = "retain" + str(100 - int(cfg.split.replace("forget", ""))).zfill(2)        
    torch_format_dataset = CommonForgetQA(cfg.forget_data_path, cfg.retain_data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, retain_split=retain_split, loss_type=cfg.forget_loss)
    
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    print(f"max_steps: {max_steps}")
    
    if cfg.bf16 is True:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(0, steps_per_epoch),  
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=cfg.bf16,  
            bf16_full_eval=cfg.bf16, 
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            evaluation_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed

        )
    
    #first get the base model architectur2e
    #if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    import re
    path_found = False
    for file in os.listdir(cfg.model_path):
        if re.search("pytorch.*\.bin", file):
            path_found = True
            break
        
        if re.search("model-*\.safetensors", file):
            path_found = True
            break

    oracle_model = None
    assistant_model = None
    config = AutoConfig.from_pretrained(model_id)
    model_name = config._name_or_path.lower()
    
    if "ULD" in cfg.forget_loss:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, \
            torch_dtype=torch_dtype, use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
            trust_remote_code=True)
    else:
        config = AutoConfig.from_pretrained(model_id)
        print("Loading from checkpoint")
        if "phi" in model_name:
            causalLM = PhiForCausalLM
        elif "llama" in model_name:
            causalLM = LlamaForCausalLM
        else:
            causalLM = AutoModelForCausalLM
        model = causalLM.from_pretrained(cfg.model_path, config=config, \
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch_dtype, \
            trust_remote_code = True)
    
    if "kl" in cfg.forget_loss or "npo" in cfg.forget_loss or "dpo" in cfg.forget_loss: # or "ours" in cfg.forget_loss:
        oracle_model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, \
            use_flash_attention_2=model_cfg["flash_attention2"]=="true", \
            torch_dtype=torch_dtype, trust_remote_code = True)
            
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    
    #now we have a HuggingFace model 
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    
    
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=torch_format_dataset.custom_data_collator_forget,
        oracle_model = oracle_model,
        assistant_model = assistant_model,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
        retain_weight = cfg.retain_weight,
        C = cfg.C,
        P = cfg.P,
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            #delete the directory
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

