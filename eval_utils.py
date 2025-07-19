
from rouge_score import rouge_scorer
import nltk
import scipy
import evaluate
import numpy as np
import torch
import torch.nn as nn
from evals.uld import ULDLLM
from evals.whos_harry_potter import WHPLLM
from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel
# ========================= MODEL CREATION FUNCTIONS =========================

def load_model_with_retry(cfg, model_cfg, config, torch_dtype, device_map, tokenizer, pretained_traget_model_path):
    """Load model with retry mechanism for different configurations."""
    model = None
    
    for attempt in range(3):
        try:
            if cfg.use_pretrained or "icl" in cfg.forget_loss:
                print(f"Loading checkpoint from {pretained_traget_model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    pretained_traget_model_path, config=config, 
                    use_flash_attention_2=model_cfg["flash_attention2"] == "true", 
                    torch_dtype=torch_dtype, trust_remote_code=True, 
                    device_map=device_map
                )
                
            elif "ULD" in cfg.forget_loss:
                model = create_uld_model(cfg, config, model_cfg, torch_dtype, device_map, tokenizer, pretained_traget_model_path)
                
            elif "WHP" in cfg.forget_loss:
                model = create_whp_model(cfg, config, model_cfg, torch_dtype, device_map, tokenizer, pretained_traget_model_path)
                
            elif cfg.use_lora:
                model = create_lora_model(cfg, config, model_cfg, torch_dtype, device_map, pretained_traget_model_path)
                
            elif model_cfg["hf_key"] == "microsoft/Phi-3.5-mini-instruct":
                model = create_phi_model(cfg, config, torch_dtype, device_map)
                
            else:
                model = create_standard_model(cfg, config, torch_dtype, device_map)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
        else:
            break
    else:
        print("Error: could not load model")
    
    return model

def create_uld_model(cfg, config, model_cfg, torch_dtype, device_map, tokenizer, pretained_traget_model_path):
    """Create ULD (Unlearning with Language Distillation) model."""
    print(f"Loading checkpoint from {cfg.model_path}")
    
    basemodel = AutoModelForCausalLM.from_pretrained(
        pretained_traget_model_path, config=config, 
        use_flash_attention_2=model_cfg["flash_attention2"] == "true",
        torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
    )
    
    assistant = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, config=config, 
        use_flash_attention_2=model_cfg["flash_attention2"] == "true", 
        torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
    )
    
    return ULDLLM(
        basellm=basemodel, 
        assist_llm=assistant, 
        weight=-0.8, 
        top_logit_filter=0.1,
        tokenizer=tokenizer
    )


def create_whp_model(cfg, config, model_cfg, torch_dtype, device_map, tokenizer, pretained_traget_model_path):
    """Create WHP (Who's Harry Potter) model."""
    basemodel = AutoModelForCausalLM.from_pretrained(
        pretained_traget_model_path, config=config, 
        use_flash_attention_2=model_cfg["flash_attention2"] == "true", 
        torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
    )
    
    reinforce_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, config=config, 
        use_flash_attention_2=model_cfg["flash_attention2"] == "true", 
        torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
    )
    
    return WHPLLM(
        basellm=basemodel,
        reinforced_llm=reinforce_model,
        tokenizer=tokenizer
    )

def create_lora_model(cfg, config, model_cfg, torch_dtype, device_map, pretained_traget_model_path):
    """Create LoRA model."""
    print(f"Loading base model from {pretained_traget_model_path}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        pretained_traget_model_path, config=config, 
        use_flash_attention_2=model_cfg["flash_attention2"] == "true",
        torch_dtype=torch_dtype, trust_remote_code=True, device_map=device_map
    )
    
    print(f"Loading LoRA adapter from {cfg.model_path}..")
    return PeftModel.from_pretrained(base_model, cfg.model_path)


def create_phi_model(cfg, config, torch_dtype, device_map):
    """Create Phi-3.5 model with specific configuration."""
    print(f"Loading checkpoint from {cfg.model_path}")
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_path, config=config,
        attn_implementation="flash_attention_2", torch_dtype=torch_dtype,
        device_map=device_map, trust_remote_code=False
    )

def create_standard_model(cfg, config, torch_dtype, device_map):
    """Create standard model."""
    print(f"Loading checkpoint from {cfg.model_path}")
    return AutoModelForCausalLM.from_pretrained(
        cfg.model_path, config=config,
        attn_implementation="flash_attention_2", torch_dtype=torch_dtype,
        device_map=device_map, trust_remote_code=True
    )


# ========================= UTILITY FUNCTIONS =========================

def convert_to_left_padding(input_ids, attention_mask, labels, pad_token_id):
    """Convert right-padded sequences to left-padded for specific models like Qwen."""
    if len(input_ids.shape) > 2:
        batch_size, _, seq_len = input_ids.shape
    else:
        batch_size, seq_len = input_ids.shape
        
    left_padded_input_ids = torch.zeros_like(input_ids)
    left_padded_attention_mask = torch.zeros_like(attention_mask)
    left_padded_labels = torch.full_like(labels, -100)
    
    for i in range(batch_size):
        actual_length = (attention_mask[i] == 1).sum().item()
        
        if actual_length < seq_len:
            # Move actual tokens to the right, padding to the left
            padding_length = seq_len - actual_length
            
            left_padded_input_ids[i, padding_length:] = input_ids[i, :actual_length]
            left_padded_attention_mask[i, padding_length:] = 1
            left_padded_labels[i, padding_length:] = labels[i, :actual_length]
            
            # Set padding tokens
            left_padded_input_ids[i, :padding_length] = pad_token_id
        else:
            # No padding needed
            left_padded_input_ids[i] = input_ids[i]
            left_padded_attention_mask[i] = attention_mask[i]
            left_padded_labels[i] = labels[i]

    return left_padded_input_ids, left_padded_attention_mask, left_padded_labels


def reinitialize_weights(model) -> None:
    """Reinitialize model weights for testing purposes."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def relative_top_filter(scores: torch.FloatTensor, relative_top: float = 0.1, 
                       filter_value: float = -float(1e3), min_tokens_to_keep: int = 1) -> torch.FloatTensor:
    """Apply relative top filtering to scores."""
    min_tokens_to_keep = int(relative_top * scores.shape[-1])
    scores_normalized = scores.log_softmax(dim=-1) 
    sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
    min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_thresh = probs_max + np.log(relative_top)
    probs_thresh = torch.min(min_thresh, probs_thresh)
    probs_thresh = probs_thresh.unsqueeze(-1)
    mask = scores_normalized < probs_thresh
    return scores, mask, probs_thresh


# ========================= EVALUATION METRICS FUNCTIONS =========================

def eval_accuracy(logits, labels):
    """Evaluate accuracy of predictions."""
    preds = logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()
    return {"eval accuracy": acc.item()}


def compute_freq(sentence, n=2):
    """Compute n-gram frequency distribution."""
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    """Compute n-gram entropy for a sentence."""
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def n_gram_entropy(gen_texts, agg="arith"):
    """Calculate n-gram entropy for generated texts."""
    assert agg in ["arith", "geom"]
    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def eval_bleu(gen_outputs, ground_truths):
    """Evaluate BLEU and ROUGE scores."""
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)

    return {'rouge': rouge_res, 'bleu': bleu_res}


def eval_rouge_recall(gen_outputs, ground_truths, indices):
    """Evaluate ROUGE recall scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall
    
    # Calculate fluency score
    fluency = n_gram_entropy(gen_outputs)
    print("fluency", fluency)

    return {
        'rouge1_recall': rouge1_recall, 
        'rougeL_recall': rougeL_recall, 
        'fluency': fluency
    }

