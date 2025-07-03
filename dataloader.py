import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
#from evaluate_TOFU import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility, permu_log_states,get_config
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from torch import nn
import copy
import numpy as np
from utils import get_logger,should_log_stats

os.environ['MASTER_PORT'] = '22395'
def printll(name, inp):
    #print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.assistant_model = kwargs.pop('assistant_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.retain_weight = kwargs.pop('retain_weight')
        self.C = kwargs.pop('C')
        self.P = kwargs.pop('P')
        self.in_text = kwargs.pop('in_text')

        if hasattr(self, 'model') and self.model is not None:
            device = next(self.model.parameters()).device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if self.oracle_model is not None:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)
            self.oracle_model = self.oracle_model.to(device)
        if self.assistant_model is not None:
            self.assistant_model = self.e_prepare_deepspeed(self.assistant_model)
            self.oracle_model = self.oracle_model.to(device)
    
    def e_prepare_deepspeed(self, model):
        # Simple preparation without DeepSpeed
        if model is not None:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        return model

    # def e_prepare_deepspeed(self, model):
    #     # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    #     deepspeed_plugin = self.accelerator.state.deepspeed_plugin
    #     config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

    #     if model is not None:
    #         if hasattr(model, "config"):
    #             hidden_size = (
    #                 max(model.config.hidden_sizes)
    #                 if getattr(model.config, "hidden_sizes", None)
    #                 else getattr(model.config, "hidden_size", None)
    #             )
    #             if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
    #                 # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
    #                 # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
    #                 config_kwargs.update(
    #                     {
    #                         "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
    #                         "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
    #                         "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
    #                     }
    #                 )

    #     if config_kwargs["zero_optimization"]["stage"] != 3:
    #         config_kwargs["zero_optimization"]["stage"] = 0
    #     config_kwargs["optimizer"] = {"type": None}
    #     model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    #     model.eval()
    #     for param in model.parameters():
    #         param.requires_grad = False
        
    #     return model


    def add_entropy_controlled_noise(self, logits, target_entropy_increase=0.1, suppression_strength=0.01):
            """
            Smoothly redistribute logits: reduce top entries and redistribute to others
            Preserves total logit sum for true redistribution
            
            Args:
                logits: Input logits [batch_size, seq_len, vocab_size]
                target_entropy_increase: Target increase in entropy (relative)
                suppression_strength: How much to suppress top entries
            """

            batch_size, seq_len, vocab_size = logits.shape
            
            probs = F.softmax(logits, dim=-1)
            current_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [batch_size, seq_len]
            
            suppression_weights = probs * suppression_strength  # Higher prob = more suppression
            
            total_suppression = suppression_weights.sum(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            
            uniform_boost = total_suppression / vocab_size  # Equal redistribution
            
            redistributed_logits = logits - suppression_weights + uniform_boost
            
            new_probs = F.softmax(redistributed_logits, dim=-1)
            new_entropy = -torch.sum(new_probs * torch.log(new_probs + 1e-10), dim=-1)
            
            entropy_increase = (new_entropy - current_entropy) / (current_entropy + 1e-6)
            scale_factor = target_entropy_increase / (entropy_increase.mean() + 1e-6)
            scale_factor = torch.clamp(scale_factor, 0.1, 3.0)
            
            final_suppression = suppression_weights * scale_factor
            final_boost = (final_suppression.sum(dim=-1, keepdim=True)) / vocab_size
            
            print(f'Old Entropy : {current_entropy} increased to {new_entropy} with scale factor {scale_factor}')
            return logits - final_suppression + final_boost

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None): # DP : Add num_items_in_batch argument to fix issue version issue : TypeError: CustomTrainerForgetting.compute_loss() got an unexpected keyword argument 'num_items_in_batch'
        def detailed_memory_report():
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"CUDA memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
            print(f"Model dtype: {next(model.parameters()).dtype}")
            print(f"Model device: {next(model.parameters()).device}")
            print(f"Total model params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
            print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

        # detailed_memory_report()
        
        model_device = next(model.parameters()).device

        retain_weight = self.retain_weight
        if "grad_ascent" in self.loss_type:
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
        elif "dpo" in self.loss_type:
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits
            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)
            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            beta = 0.1
            forget_loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
        elif "npo" in self.loss_type:
            forget_inputs, retain_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss_current = get_batch_loss(forget_outputs.logits, forget_labels)
            with torch.no_grad():
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = get_batch_loss(forget_logits_oracle, forget_labels)
            log_ratio = forget_loss_current - forget_loss_oracle
            beta = 0.1
            forget_loss = -F.logsigmoid(beta * log_ratio).mean() * 2 / beta
        elif "task_vector" in self.loss_type:
            forget_inputs, retain_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = outputs.loss
            
        elif "WHP" in self.loss_type:
            forget_inputs, retain_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = outputs.loss
        elif "ULD" in self.loss_type:
            # for forget set
            forget_inputs, retain_inputs = inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
            forget_loss = forget_outputs.loss
        elif self.loss_type.startswith("PerMU") and self.in_text: ### DP Addition : New block to accomodate the Discrete Tokens variant of Per
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, tokens_to_mix, question_mask, perturbed_input_ids = forget_inputs
            with torch.no_grad():

                clean_output = model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
                clean_logits = clean_output.logits
              
                corrupt_output = model(perturbed_input_ids, attention_mask=attention_mask, return_dict=True, \
                    output_hidden_states=False)
                corrupt_logits = corrupt_output.logits
                

                logit = corrupt_logits

                clean_logits_copy = copy.deepcopy(clean_logits)
                # DP: the question tokens are 0, as when we calc loss they should not be considerered
                clean_target_masks = torch.zeros_like(input_ids)

                ## DP : iterate through each batch
                for i in range(logit.size(0)):
                    start, end = question_mask[i][0]

                    # DP: the answer questions are 1 as they should be considered
                    clean_target_masks[i, start-1:end] = 1
                    clean_probabilities = clean_logits[i, start-1:end, :]
                    corrupt_probabilities = logit[i, start-1:end, :]
                    assert clean_probabilities.size(0) == corrupt_probabilities.size(0)
                    
                    probabilities = corrupt_probabilities - self.C * clean_probabilities
                    clean_logits_copy[i,start-1:end,:] = probabilities
                    for sub in tokens_to_mix[i]:
                        subject_start, subject_end = sub[0], sub[1]
                        if subject_start >= start - 1: ### IF Subject is in Answer, Keep the original, 'clean' logits for the Subject, by clean though I mean the original pertrubed_subject_tokens
                                                       ### -> The tokens_to_mix implementation should still be part of the method, since I need to maintain the corrupted subj tokens ( after 1 round Subject logits might be close to the actual truth)
                            clean_logits_copy[i, subject_start-1:subject_end-1,:] = clean_logits[i, subject_start-1:subject_end-1,:]

                
                    
            student_outputs = model(input_ids, attention_mask=attention_mask)
            student_logit = student_outputs.logits
            # fast KL
            forget_loss = calc_ce_loss(clean_target_masks, student_logit, clean_logits_copy)


        elif self.loss_type.startswith("PerMU"):
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask, tokens_to_mix, question_mask = forget_inputs
            with torch.no_grad():
                clean_output = model(input_ids, attention_mask=attention_mask, output_hidden_states=False)
                clean_logits = clean_output.logits#[-1]

                all_layer = []
                for m in range(input_ids.size(0)):
                    all_layer.append(1)

                print(f'Adding noise with C={self.C} and P={self.P}, tokens_to_mix={tokens_to_mix[0]}, layer={all_layer}')
                corrupt_output = model(input_ids, attention_mask=attention_mask, return_dict=True, \
                    output_hidden_states=False, tokens_to_mix=tokens_to_mix, layer=all_layer, noise=self.P)
                corrupt_logits = corrupt_output.logits
                logit = corrupt_logits#[-1]
                
                clean_logits_copy = copy.deepcopy(clean_logits)
                # DP: the question tokens are 0, as when we calc loss they should not be considerered
                clean_target_masks = torch.zeros_like(input_ids)

                ## DP : iterate through each batch
                for i in range(logit.size(0)):
                    start, end = question_mask[i][0]

                    # DP: the answer questions are 1 as tehy should be considered
                    clean_target_masks[i, start-1:end] = 1
                    clean_probabilities = clean_logits[i, start-1:end, :]
                    corrupt_probabilities = logit[i, start-1:end, :]
                    assert clean_probabilities.size(0) == corrupt_probabilities.size(0)
                    
                    probabilities = corrupt_probabilities - self.C * clean_probabilities
                    clean_logits_copy[i,start-1:end,:] = probabilities
                    for sub in tokens_to_mix[i]:
                        subject_start, subject_end = sub[0], sub[1]
                        if subject_start >= start - 1: ### IF Subject is in Answer, Keep the original, 'clean' logits for the Subject, by clean though I mean the original pertrubed_subject_tokens
                                                       ### -> The tokens_to_mix implementation should still be part of the method, since I need to maintain the corrupted subj tokens ( after 1 round Subject logits might be close to the actual truth)
                            clean_logits_copy[i, subject_start-1:subject_end-1,:] = clean_logits[i, subject_start-1:subject_end-1,:]
                            
            student_outputs = model(input_ids, attention_mask=attention_mask)
            student_logit = student_outputs.logits
            # fast KL
            forget_loss = calc_ce_loss(clean_target_masks, student_logit, clean_logits_copy)




        else:
            raise NotImplementedError(f"Invalid forget loss type: {self.loss_type}")
           
        # retain loss        
        if "gd" in self.loss_type or self.loss_type.startswith("PerMU"):
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
        # minimum KL divergence
        elif "kl" in self.loss_type:
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)

                
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])
            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            # minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)  
        elif "ULD" in self.loss_type:
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            logits = retain_outputs.logits
            num_labels = logits.shape[-1]
            soft_outputs = nn.functional.softmax(logits, dim=-1).view(-1, num_labels)
            uniform_dist = torch.full_like(soft_outputs, 1.0 / logits.size(-1)).to(logits.device)
            retain_loss = torch.nn.functional.kl_div(soft_outputs.log(), uniform_dist, reduction='batchmean')
        else:
            retain_loss = 0     
        
        loss = forget_loss + retain_weight * retain_loss
        if should_log_stats('permu_contrast_stats') and self.loss_type.startswith("PerMU"):
                    contrast_logits_all = copy.deepcopy(clean_logits_copy)
                    permu_log_states(corrupt_logits=corrupt_logits,clean_logits=clean_logits,question_mask=question_mask,contrasted_logits_all=contrast_logits_all,student_logits=student_logit,forget_loss=forget_loss,retain_loss=retain_loss,C=self.C)
        
        return (loss, outputs) if return_outputs else loss
        
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(eval_cfg, eval_task, self.tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key)
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 

                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

                with open(save_filename, "w") as f:
                    json.dump(eval_logs, f, indent=4)
            
            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)
                            #delete old files use shutil
                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)

def padding(tensor, to_length, pad_id, pad_to_rght=True):
    if tensor.dim() == 0:
        tensor = torch.tensor([pad_id])
    if len(tensor) == to_length:
        return tensor
    else:
        fillin = torch.tensor([pad_id] * (to_length - len(tensor))).to(tensor.device)
        if pad_to_rght:
            tensor =  torch.concat([tensor, fillin])
        else:
            tensor =  torch.concat([fillin, tensor])
        return tensor

def calc_ce_loss(mask, s_logits, t_logits, temperature=1):
        """Copy pasted from distillbert (transformers/examples/distillation/)"""
        # mask has False at padding_idx
        ce_loss_fct = nn.KLDivLoss(reduction="batchmean")
        sel_mask = mask[:, :, None].expand_as(s_logits).bool()
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_ce = (
            ce_loss_fct(
                F.log_softmax(s_logits_slct / temperature, dim=-1), # bottom
                F.softmax(t_logits_slct / temperature, dim=-1), # up 
            )
            * (temperature) ** 2
        )
        return loss_ce

def custom_data_collator_forget(samples, forget_loss="KL"):
    if forget_loss in ["idk", "dpo"]:
        idk_samples, forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
        rets = []
        for data_type in ["idk", "forget", "retain"]:
            if data_type == "forget":
                data = forget_samples 
            elif data_type == "retain":
                data = retain_samples
            else:
                data = idk_samples
            input_ids = [s[0] for s in data]
            labels = [s[1] for s in data]
            attention_mask = [s[2] for s in data]
            rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    elif data_type == "retain":
            data = retain_samples
            input_ids = [s[0] for s in data]
            labels = [s[1] for s in data]
            attention_mask = [s[2] for s in data]
            rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    else:
        forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
        rets = []
        for data_type in ["forget", "retain"]:
            data = forget_samples if data_type == "forget" else retain_samples
            input_ids = [s[0] for s in data]
            labels = [s[1] for s in data]
            attention_mask = [s[2] for s in data]
            rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss


