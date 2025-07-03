import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
import csv
import random
import pandas as pd
import json
from typing import List, Dict, Any, Tuple, Optional, Set, Union
######## LOGGER STUFF #########
import logging
import os
from datetime import datetime
from typing import Optional
import torch
import torch.nn.functional as F


############################################## ENTROPY,MAGNITUDE, DIVERGENCE LOGGING ##############################################
# Global lists to collect statistics across all method calls/epochs
permu_clean_entropy = []
permu_corrupt_entropy = []
permu_diff_entropy = []
permu_contrasted_entropy = []
permu_student_entropy = []  # Added for student_logits
permu_clean_magnitude = []
permu_corrupt_magnitude = []
permu_diff_magnitude = []
permu_contrasted_magnitude = []
permu_student_magnitude = []  # Added for student_logits
permu_kl_div = []
permu_forget_loss = []  # Added for forget_loss
permu_retain_loss = []  # Added for retain_loss


def permu_log_states(corrupt_logits, clean_logits, question_mask, contrasted_logits_all, student_logits, forget_loss, retain_loss, C):
    global permu_clean_entropy, permu_corrupt_entropy, permu_diff_entropy, permu_contrasted_entropy, permu_student_entropy
    global permu_clean_magnitude, permu_corrupt_magnitude, permu_diff_magnitude, permu_contrasted_magnitude, permu_student_magnitude
    global permu_kl_div, permu_forget_loss, permu_retain_loss

    # === COMPREHENSIVE METRICS LOGGING SECTION ===
    logger = get_logger()
    
    # Collect all answer tokens from all batch elements
    all_clean_answer_logits = []
    all_corrupt_answer_logits = []
    all_contrasted_logits = []
    all_student_logits = []  # Added for student_logits
    
    for i in range(corrupt_logits.size(0)):
        start, end = question_mask[i][0]
        
        # Extract answer token logits (the ones being replaced)
        clean_answer_logits = clean_logits[i, start-1:end, :]
        corrupt_answer_logits = corrupt_logits[i, start-1:end, :]
        contrasted_logits = contrasted_logits_all[i, start-1:end, :]
        student_answer_logits = student_logits[i, start-1:end, :]  # Added for student_logits
        
        all_clean_answer_logits.append(clean_answer_logits)
        all_corrupt_answer_logits.append(corrupt_answer_logits)
        all_contrasted_logits.append(contrasted_logits)
        all_student_logits.append(student_answer_logits)  # Added for student_logits
    
    # Concatenate all answer tokens from the batch
    batch_clean_logits = torch.cat(all_clean_answer_logits, dim=0)
    batch_corrupt_logits = torch.cat(all_corrupt_answer_logits, dim=0)
    batch_contrasted_logits = torch.cat(all_contrasted_logits, dim=0)
    batch_student_logits = torch.cat(all_student_logits, dim=0)  # Added for student_logits
    batch_difference_logits = batch_corrupt_logits - C * batch_clean_logits
    
    # === ENTROPY CALCULATIONS ON ENTIRE BATCH ===
    clean_probs = F.softmax(batch_clean_logits, dim=-1)
    clean_entropy = -torch.sum(clean_probs * torch.log(clean_probs + 1e-10), dim=-1)
    corrupt_probs = F.softmax(batch_corrupt_logits, dim=-1)
    corrupt_entropy = -torch.sum(corrupt_probs * torch.log(corrupt_probs + 1e-10), dim=-1)
    diff_probs = F.softmax(batch_difference_logits, dim=-1)
    diff_entropy = -torch.sum(diff_probs * torch.log(diff_probs + 1e-10), dim=-1)
    contrasted_probs = F.softmax(batch_contrasted_logits, dim=-1)
    contrasted_entropy = -torch.sum(contrasted_probs * torch.log(contrasted_probs + 1e-10), dim=-1)
    # Added for student_logits
    student_probs = F.softmax(batch_student_logits, dim=-1)
    student_entropy = -torch.sum(student_probs * torch.log(student_probs + 1e-10), dim=-1)
    
    # === MAGNITUDE CALCULATIONS ON ENTIRE BATCH (L∞ norm - max absolute value) ===
    clean_magnitude = torch.max(torch.abs(batch_clean_logits), dim=-1)[0]
    corrupt_magnitude = torch.max(torch.abs(batch_corrupt_logits), dim=-1)[0]
    diff_magnitude = torch.max(torch.abs(batch_difference_logits), dim=-1)[0]
    contrasted_magnitude = torch.max(torch.abs(batch_contrasted_logits), dim=-1)[0]
    student_magnitude = torch.max(torch.abs(batch_student_logits), dim=-1)[0]  # Added for student_logits
    
    # === KL DIVERGENCE CALCULATION ON ENTIRE BATCH: KL(corrupt || clean) ===
    kl_div = torch.sum(corrupt_probs * torch.log((corrupt_probs + 1e-10) / (clean_probs + 1e-10)), dim=-1)
    
    # === BATCH-LEVEL LOGGING ===
    # logger.info(f"PerMU Metrics - Batch Aggregate ({batch_clean_logits.size(0)} Answer Tokens):")
    # logger.info(f"  ENTROPY: Clean: Mean={clean_entropy.mean().item():.4f}, Std={clean_entropy.std().item():.4f}")
    # logger.info(f"           Corrupt: Mean={corrupt_entropy.mean().item():.4f}, Std={corrupt_entropy.std().item():.4f}")
    # logger.info(f"           Difference: Mean={diff_entropy.mean().item():.4f}, Std={diff_entropy.std().item():.4f}")
    # logger.info(f"           Contrasted: Mean={contrasted_entropy.mean().item():.4f}, Std={contrasted_entropy.std().item():.4f}")
    # logger.info(f"           Student: Mean={student_entropy.mean().item():.4f}, Std={student_entropy.std().item():.4f}")  # Added
    # logger.info(f"  MAGNITUDE (L∞): Clean: Mean={clean_magnitude.mean().item():.4f}, Std={clean_magnitude.std().item():.4f}")
    # logger.info(f"                  Corrupt: Mean={corrupt_magnitude.mean().item():.4f}, Std={corrupt_magnitude.std().item():.4f}")
    # logger.info(f"                  Difference: Mean={diff_magnitude.mean().item():.4f}, Std={diff_magnitude.std().item():.4f}")
    # logger.info(f"                  Contrasted: Mean={contrasted_magnitude.mean().item():.4f}, Std={contrasted_magnitude.std().item():.4f}")
    # logger.info(f"                  Student: Mean={student_magnitude.mean().item():.4f}, Std={student_magnitude.std().item():.4f}")  # Added
    # logger.info(f"  KL DIVERGENCE: KL(Corrupt||Clean): Mean={kl_div.mean().item():.4f}, Std={kl_div.std().item():.4f}")
    # logger.info(f"  FORGET LOSS: {forget_loss.item():.4f}")
    # logger.info(f"  RETAIN LOSS: {retain_loss.item():.4f}")  # Added for retain_loss
    
    # === COLLECT BATCH-AGGREGATED STATISTICS FOR GLOBAL LISTS ===
    # Store batch-level aggregated metrics (one value per batch, not per token)
    permu_clean_entropy.append(clean_entropy.mean().item())
    permu_corrupt_entropy.append(corrupt_entropy.mean().item())
    permu_diff_entropy.append(diff_entropy.mean().item())
    permu_contrasted_entropy.append(contrasted_entropy.mean().item())
    permu_student_entropy.append(student_entropy.mean().item())  # Added
    permu_clean_magnitude.append(clean_magnitude.mean().item())
    permu_corrupt_magnitude.append(corrupt_magnitude.mean().item())
    permu_diff_magnitude.append(diff_magnitude.mean().item())
    permu_contrasted_magnitude.append(contrasted_magnitude.mean().item())
    permu_student_magnitude.append(student_magnitude.mean().item())  # Added
    permu_kl_div.append(kl_div.mean().item())
    permu_forget_loss.append(forget_loss.item())
    permu_retain_loss.append(retain_loss.item())  # Added for retain_loss
    
    # === OVERALL STATISTICS ===
    # logger.info("=" * 80)
    # logger.info("PerMU CURRENT BATCH STATISTICS:")
    # logger.info("=" * 80)
    # logger.info(f"Answer tokens in this batch: {batch_clean_logits.size(0)}")
    # logger.info(f"Total batches processed globally: {len(permu_clean_entropy)}")
    # logger.info("=" * 80)
    # === END COMPREHENSIVE METRICS LOGGING ===



def save_permu_metrics_to_json(save_dir="./permu_metrics", experiment_name="permu_experiment"):
    """Save all accumulated PerMU metrics to JSON files. Call this after training is complete."""
    import os, json, numpy as np
    from datetime import datetime
    
    # Access global variables
    global permu_clean_entropy, permu_corrupt_entropy, permu_diff_entropy, permu_contrasted_entropy, permu_student_entropy
    global permu_clean_magnitude, permu_corrupt_magnitude, permu_diff_magnitude, permu_contrasted_magnitude, permu_student_magnitude
    global permu_kl_div, permu_forget_loss, permu_retain_loss
    
    os.makedirs(save_dir, exist_ok=True)
    if 'permu_clean_entropy' not in globals() or len(permu_clean_entropy) == 0:
        print("No PerMU metrics data found to save.")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # === SAVE RAW DATA ===
    raw_data = {
        "experiment_info": {"experiment_name": experiment_name, "timestamp": timestamp, "total_batches": len(permu_clean_entropy)},
        "entropy": {
            "clean": [float(x) for x in permu_clean_entropy], 
            "corrupt": [float(x) for x in permu_corrupt_entropy], 
            "difference": [float(x) for x in permu_diff_entropy], 
            "contrasted": [float(x) for x in permu_contrasted_entropy],
            "student": [float(x) for x in permu_student_entropy]  # Added
        },
        "magnitude": {
            "clean": [float(x) for x in permu_clean_magnitude], 
            "corrupt": [float(x) for x in permu_corrupt_magnitude], 
            "difference": [float(x) for x in permu_diff_magnitude], 
            "contrasted": [float(x) for x in permu_contrasted_magnitude],
            "student": [float(x) for x in permu_student_magnitude]  # Added
        },
        "kl_divergence": {"corrupt_vs_clean": [float(x) for x in permu_kl_div]},
        "forget_loss": [float(x) for x in permu_forget_loss],
        "retain_loss": [float(x) for x in permu_retain_loss]  # Added for retain_loss
    }
    
    raw_filename = f"{experiment_name}_raw_data.json"
    with open(os.path.join(save_dir, raw_filename), 'w') as f:
        json.dump(raw_data, f, indent=2)
    
    # === SAVE STATISTICAL SUMMARY ===
    def compute_stats(data_list):
        if len(data_list) == 0: return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
        data_array = np.array(data_list)
        return {"mean": float(np.mean(data_array)), "std": float(np.std(data_array)), "min": float(np.min(data_array)), "max": float(np.max(data_array)), "median": float(np.median(data_array)), "count": len(data_list)}
    
    summary_data = {
        "experiment_info": {"experiment_name": experiment_name, "timestamp": timestamp, "total_batches": len(permu_clean_entropy)},
        "entropy_stats": {
            "clean": compute_stats(permu_clean_entropy), 
            "corrupt": compute_stats(permu_corrupt_entropy), 
            "difference": compute_stats(permu_diff_entropy), 
            "contrasted": compute_stats(permu_contrasted_entropy),
            "student": compute_stats(permu_student_entropy)  # Added
        },
        "magnitude_stats": {
            "clean": compute_stats(permu_clean_magnitude), 
            "corrupt": compute_stats(permu_corrupt_magnitude), 
            "difference": compute_stats(permu_diff_magnitude), 
            "contrasted": compute_stats(permu_contrasted_magnitude),
            "student": compute_stats(permu_student_magnitude)  # Added
        },
        "kl_divergence_stats": {"corrupt_vs_clean": compute_stats(permu_kl_div)},
        "forget_loss_stats": compute_stats(permu_forget_loss),
        "retain_loss_stats": compute_stats(permu_retain_loss)  # Added for retain_loss
    }
    
    summary_filename = f"{experiment_name}_summary.json"
    with open(os.path.join(save_dir, summary_filename), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # === PRINT SUMMARY TO CONSOLE ===
    print("=" * 100)
    print(f"PERMU METRICS FINAL SUMMARY - {experiment_name}")
    print("=" * 100)
    print(f"Total batches processed: {len(permu_clean_entropy)}")
    print("\nENTROPY FINAL STATISTICS:")
    for metric_name, stats in summary_data["entropy_stats"].items():
        print(f"  {metric_name.capitalize()}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")
    print("\nMAGNITUDE (L∞) FINAL STATISTICS:")
    for metric_name, stats in summary_data["magnitude_stats"].items():
        print(f"  {metric_name.capitalize()}: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")
    print("\nKL DIVERGENCE FINAL STATISTICS:")
    kl_stats = summary_data["kl_divergence_stats"]["corrupt_vs_clean"]
    print(f"  KL(Corrupt||Clean): Mean={kl_stats['mean']:.4f}, Std={kl_stats['std']:.4f}, Min={kl_stats['min']:.4f}, Max={kl_stats['max']:.4f}")
    print("\nFORGET LOSS FINAL STATISTICS:")
    forget_stats = summary_data["forget_loss_stats"]
    print(f"  Forget Loss: Mean={forget_stats['mean']:.4f}, Std={forget_stats['std']:.4f}, Min={forget_stats['min']:.4f}, Max={forget_stats['max']:.4f}")
    print("\nRETAIN LOSS FINAL STATISTICS:")  # Added for retain_loss
    retain_stats = summary_data["retain_loss_stats"]
    print(f"  Retain Loss: Mean={retain_stats['mean']:.4f}, Std={retain_stats['std']:.4f}, Min={retain_stats['min']:.4f}, Max={retain_stats['max']:.4f}")
    print(f"\nFiles saved:\n  Raw data: {os.path.join(save_dir, raw_filename)}\n  Summary: {os.path.join(save_dir, summary_filename)}")
    print("=" * 100)


def reset_permu_metrics():
    """Reset all global PerMU metrics. Call this before starting a new experiment."""
    global permu_clean_entropy, permu_corrupt_entropy, permu_diff_entropy, permu_contrasted_entropy, permu_student_entropy
    global permu_clean_magnitude, permu_corrupt_magnitude, permu_diff_magnitude, permu_contrasted_magnitude, permu_student_magnitude
    global permu_kl_div, permu_forget_loss, permu_retain_loss
    
    permu_clean_entropy, permu_corrupt_entropy, permu_diff_entropy, permu_contrasted_entropy, permu_student_entropy = [], [], [], [], []
    permu_clean_magnitude, permu_corrupt_magnitude, permu_diff_magnitude, permu_contrasted_magnitude, permu_student_magnitude = [], [], [], [], []
    permu_kl_div, permu_forget_loss, permu_retain_loss = [], [], []
    print("PerMU metrics reset for new experiment.")
############################################### OPTIMAL TOKENIZER ###########################################

# utils.py - Vocabulary indices and caching utilities

def setup_optimized_tokenizer(tokenizer, memory_mode="balanced", precompute_tokens=1000):
    """
    One-time setup for optimized token corruption
    Call this once after loading your tokenizer
    """
    print("Setting up optimized tokenizer...")
    
    # Configure memory usage
    configure_memory_usage(memory_mode)
    
    # Build vocabulary indices
    build_vocab_indices(tokenizer)
    
    # Precompute neighbors for common tokens
    if precompute_tokens > 0:
        if memory_mode == "minimal":
            print("Skipping precomputation in minimal memory mode")
        else:
            memory_efficient_precompute(tokenizer, target_memory_mb=50)
    
    print(f"Setup complete. Memory usage: {get_memory_usage_mb():.1f} MB")


import re
from collections import defaultdict, OrderedDict

# Global caches and indices
_vocab_by_length = None
_vocab_by_first_char = None
_latin_token_cache = {}
_neighbor_cache = {}

class LimitedCache(OrderedDict):
    """LRU cache with size limit to control memory usage"""
    def __init__(self, max_size=10000):
        super().__init__()
        self.max_size = max_size
    
    def __setitem__(self, key, value):
        if key in self:
            # Move to end
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_size:
            # Remove oldest item
            self.popitem(last=False)


def is_latin_alphabet_only(text):
    """
    Check if a token contains only Latin alphabet characters, numbers, 
    common punctuation, and whitespace/special tokenizer symbols.
    Cached version for better performance.
    """
    if text in _latin_token_cache:
        return _latin_token_cache[text]
    
    # Remove common tokenizer prefixes/symbols
    cleaned_text = text.replace('▁', '').replace('Ġ', '').replace('##', '')
    
    # Allow Latin letters, numbers, common punctuation, and whitespace
    latin_pattern = r'^[a-zA-Z0-9\s\.,;:!?\-\'\"()\[\]{}/@#$%^&*+=<>|\\~`_]*$'
    
    result = bool(re.match(latin_pattern, cleaned_text))
    _latin_token_cache[text] = result
    return result

def build_vocab_indices(tokenizer):
    """Build optimized indices for the vocabulary - call this once at startup"""
    global _vocab_by_length, _vocab_by_first_char
    
    if _vocab_by_length is not None:
        return  # Already built
    
    print("Building vocabulary indices...")
    _vocab_by_length = defaultdict(list)
    _vocab_by_first_char = defaultdict(list)
    
    # Pre-filter and index Latin tokens
    for vocab_token, vocab_token_id in tokenizer.vocab.items():
        if is_latin_alphabet_only(vocab_token):
            token_length = len(vocab_token)
            _vocab_by_length[token_length].append((vocab_token, vocab_token_id))
            
            # Index by first alphabetic character
            first_char = ''
            for char in vocab_token:
                if char.isalpha():
                    first_char = char
                    break
            if first_char:
                _vocab_by_first_char[first_char].append((vocab_token, vocab_token_id))
    
    print(f"Vocabulary indices built: {len(_vocab_by_length)} length buckets, "
          f"{len(_vocab_by_first_char)} first-char buckets")

def get_vocab_by_length():
    """Get the vocabulary grouped by length"""
    return _vocab_by_length

def get_vocab_by_first_char():
    """Get the vocabulary grouped by first character"""
    return _vocab_by_first_char

def get_neighbor_cache():
    """Get the neighbor cache"""
    return _neighbor_cache

def set_neighbor_cache(cache):
    """Set the neighbor cache (useful for switching cache types)"""
    global _neighbor_cache
    _neighbor_cache = cache

def clear_caches():
    """Call this if you switch tokenizers or want to free memory"""
    global _vocab_by_length, _vocab_by_first_char, _latin_token_cache, _neighbor_cache
    print("Clearing all caches...")
    _vocab_by_length = None
    _vocab_by_first_char = None
    _latin_token_cache.clear()
    _neighbor_cache.clear()

def get_memory_usage_mb():
    """Get current memory usage of caches"""
    total_size = 0
    
    # Rough estimation
    if _vocab_by_length:
        total_size += sum(len(tokens) for tokens in _vocab_by_length.values()) * 40
    if _vocab_by_first_char:
        total_size += sum(len(tokens) for tokens in _vocab_by_first_char.values()) * 40
    
    total_size += len(_latin_token_cache) * 30
    total_size += len(_neighbor_cache) * 100  # Rough estimate per cache entry
    
    return total_size / (1024 * 1024)

def configure_memory_usage(mode="balanced"):
    """Configure memory vs speed trade-offs"""
    global _neighbor_cache
    
    if mode == "minimal":
        # Minimize memory usage - no neighbor caching
        _neighbor_cache = {}
        max_precompute = 0
        print("Minimal memory mode: No neighbor caching")
        
    elif mode == "balanced":
        # Balance memory and speed
        _neighbor_cache = LimitedCache(max_size=5000)
        max_precompute = 1000
        print("Balanced mode: Limited caching (5000 entries)")
        
    elif mode == "performance":
        # Maximize speed, higher memory usage
        _neighbor_cache = LimitedCache(max_size=20000)
        max_precompute = 5000
        print("Performance mode: Extended caching (20000 entries)")
        
    elif mode == "unlimited":
        # No limits - maximum speed
        _neighbor_cache = {}
        max_precompute = float('inf')
        print("Unlimited mode: No cache size limits")
    
    return max_precompute



def find_neighbourhood_k_optimized(tokenizer, token_id, k=1):
    """
    Optimized version that uses pre-built indices and caching
    """
    # Build indices if not already done
    build_vocab_indices(tokenizer)
    
    # Get references to the cached data
    vocab_by_length = get_vocab_by_length()
    vocab_by_first_char = get_vocab_by_first_char()
    neighbor_cache = get_neighbor_cache()
    
    # Check cache first
    cache_key = (token_id, k)
    if cache_key in neighbor_cache:
        return neighbor_cache[cache_key]
    
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]

    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)

    # Handle digit tokens specially (unchanged logic)
    if original_token.strip().isdigit():
        neighbors = []
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            possible_formats = [digit, f' {digit}', f'▁{digit}']
            for formatted_digit in possible_formats:
                if formatted_digit in tokenizer.vocab:
                    digit_id = tokenizer.vocab[formatted_digit]
                    if digit_id != token_id:
                        neighbors.append(digit_id)
        
        neighbor_cache[cache_key] = neighbors
        return neighbors

    match_first_char = get_config()['match_first_char']
    original_first_char = ''
    if match_first_char:
        for char in original_token:
            if char.isalpha():
                original_first_char = char
                break

    neighbors = []
    
    # Choose search strategy based on constraints
    if match_first_char and original_first_char:
        # Search only tokens with matching first character
        candidate_tokens = vocab_by_first_char.get(original_first_char, [])
    else:
        # For k=1, we can limit search to tokens of similar length (length ± k)
        if k == 1:
            candidate_tokens = []
            original_length = len(original_token)
            for length in range(max(1, original_length - k), original_length + k + 1):
                candidate_tokens.extend(vocab_by_length.get(length, []))
        else:
            # For larger k, search all Latin tokens (fallback to original approach)
            candidate_tokens = []
            for tokens_list in vocab_by_length.values():
                candidate_tokens.extend(tokens_list)
    
    # Check distance for candidate tokens
    for vocab_token, vocab_token_id in candidate_tokens:
        if vocab_token_id == token_id:
            continue
            
        distance = Levenshtein.distance(original_token, vocab_token)
        if distance <= k:
            neighbors.append(vocab_token_id)
    
    # Cache the result
    neighbor_cache[cache_key] = neighbors
    return neighbors

def find_neighbourhood_k_adaptive_strict_optimized(tokenizer, token_id, k=1):
    """
    Optimized version of the adaptive strict function
    """
    # Build indices if not already done
    build_vocab_indices(tokenizer)
    
    # Get references to the cached data
    vocab_by_length = get_vocab_by_length()
    neighbor_cache = get_neighbor_cache()
    
    # Check cache first
    cache_key = (token_id, k, 'adaptive_strict')
    if cache_key in neighbor_cache:
        return neighbor_cache[cache_key]
    
    original_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    if should_log_stats('subject_token_len'):
        add_subject_lengths(original_token)
    
    original_length = len(original_token)
    neighbors = []
    
    # Handle digit tokens
    if original_token.strip().isdigit():
        for digit in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            for fmt in [digit, f' {digit}', f'▁{digit}']:
                if fmt in tokenizer.vocab:
                    digit_id = tokenizer.vocab[fmt]
                    if digit_id != token_id and Levenshtein.distance(original_token, fmt) == original_length:
                        neighbors.append(digit_id)
        
        neighbor_cache[cache_key] = neighbors
        return neighbors
    
    # Only search tokens of the same length (major optimization for adaptive strict)
    candidate_tokens = vocab_by_length.get(original_length, [])
    
    for vocab_token, vocab_token_id in candidate_tokens:
        if (vocab_token_id != token_id and 
            Levenshtein.distance(original_token, vocab_token) == original_length):
            neighbors.append(vocab_token_id)
    
    # Cache the result
    neighbor_cache[cache_key] = neighbors
    return neighbors

def precompute_all_neighbors(tokenizer, k=1, max_cache_size=10000):
    """
    Precompute neighbors for the most common tokens to warm up the cache
    Call this once after loading your tokenizer for maximum performance
    """
    
    build_vocab_indices(tokenizer)
    
    # Get token IDs sorted by frequency (you might need to adjust this based on your tokenizer)
    # This is a simple heuristic - you might want to use actual token frequencies
    common_token_ids = list(range(min(max_cache_size, len(tokenizer.vocab))))
    
    print(f"Precomputing neighbors for {len(common_token_ids)} tokens...")
    for i, token_id in enumerate(common_token_ids):
        if i % 1000 == 0:
            current_memory = get_memory_usage_mb()
            print(f"Progress: {i}/{len(common_token_ids)}, Memory: {current_memory:.1f}MB")
        
        # Precompute both types
        find_neighbourhood_k_optimized(tokenizer, token_id, k=k)
        find_neighbourhood_k_adaptive_strict_optimized(tokenizer, token_id, k=k)
    
    print("Precomputation complete!")

def memory_efficient_precompute(tokenizer, target_memory_mb=50):
    """Precompute neighbors while staying under memory limit"""
    
    build_vocab_indices(tokenizer)
    
    max_tokens = min(len(tokenizer.vocab), target_memory_mb * 200)  # Rough estimate
    common_token_ids = list(range(max_tokens))
    
    print(f"Precomputing neighbors for {len(common_token_ids)} tokens (target: {target_memory_mb}MB)")
    
    for i, token_id in enumerate(common_token_ids):
        if i % 1000 == 0:
            current_memory = get_memory_usage_mb()
            print(f"Progress: {i}/{len(common_token_ids)}, Memory: {current_memory:.1f}MB")
            
            if current_memory > target_memory_mb:
                print(f"Memory limit reached at {i} tokens")
                break
        
        find_neighbourhood_k_optimized(tokenizer, token_id, k=1)



################################ LOGGER CONFIGURATION ################################


from omegaconf import DictConfig
_config = None
_model_config = None



def init_config(config: DictConfig) -> None:
    """Initialize global config"""
    global _config
    _config = config

def get_config() -> DictConfig:
    """Get global config"""
    if _config is None:
        raise RuntimeError("Config not initialized. Call init_config(cfg) first.")
    return _config

def should_log_stats(stat_type: str) -> bool:
    """Check if stat type should be logged"""
    return get_config().logging.get(stat_type, False)


def init_model_config(config : DictConfig) -> None:
    """Initialize model configuration"""
    global _model_config
    _model_config = config

def get_model_config() -> DictConfig:
    """Get model configuration"""
    if _model_config is None:
        raise RuntimeError("Model config not initialized. Call init_model_config(cfg) first.")
    return _model_config
############################# LOGGER ################################
_logger_instance = None

def init_logger(cfg):
    logger = setup_experiment_logger(
        experiment_name="PerMU_training",
        save_dir=cfg.save_dir,
        batch_size=cfg.batch_size,
        grad_accum=cfg.gradient_accumulation_steps,
        use_hydra=True  # Use Hydra's logging system
    )
    global _logger_instance
    _logger_instance = logger
    
    # Log experiment configuration
    logger.info("="*80)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model family: {cfg.model_family}")
    logger.info(f"Dataset: {cfg.dataset}")
    logger.info(f"Split: {cfg.split}")
    logger.info(f"Forget loss: {cfg.forget_loss}")
    logger.info(f"Batch size: {cfg.batch_size}")
    logger.info(f"Gradient accumulation steps: {cfg.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {cfg.lr}")
    logger.info(f"Number of epochs: {cfg.num_epochs}")
    logger.info(f"In-text perturbation: {cfg.in_text}")

    return logger

def get_logger(name: str = "permu_stats", log_file: Optional[str] = None, 
              level: int = logging.INFO, force_reinit: bool = False) -> logging.Logger:
    """Get or create logger (singleton pattern)"""
    global _logger_instance
    if _logger_instance is not None and not force_reinit:
        return _logger_instance
    
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
                                 datefmt='%Y-%m-%d %H:%M:%S')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logger initialized - logging to: {log_file}")
    
    _logger_instance = logger
    return logger

def get_hydra_logger(name: str = __name__) -> logging.Logger:
    """Get Hydra's logger"""
    return logging.getLogger(name)

def setup_experiment_logger(experiment_name: str, save_dir: str, batch_size: int, 
                          grad_accum: int, use_hydra: bool = True) -> logging.Logger:
    """Setup experiment logger with structured naming"""
    if use_hydra:
        logger = get_hydra_logger()
        logger.info(f"Experiment: {experiment_name} - B{batch_size}_G{grad_accum}")
        return logger
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{save_dir}/logs/{experiment_name}_B{batch_size}_G{grad_accum}_{timestamp}.log"
        return get_logger(f"{experiment_name}_logger", log_file)

def reset_logger():
    """Reset global logger instance"""
    global _logger_instance
    if _logger_instance:
        for handler in _logger_instance.handlers:
            handler.close()
        _logger_instance.handlers.clear()
    _logger_instance = None

def log_statistics(timing_stats, tokens_processed_stats, num_subjects_stats, subject_lengths_stats, 
                  logger: Optional[logging.Logger] = None):
    global _logger_instance
    """Log running statistics"""
    if _logger_instance is None:
        logger = get_logger()
        _logger_instance = logger
    else:
        logger = _logger_instance

    if not timing_stats:
        return
    
    import numpy as np
    logger.info("="*50)
    logger.info("PERFORMANCE STATISTICS")
    logger.info(f"Runs: {len(timing_stats)}")
    logger.info(f"Avg time: {np.mean(timing_stats):.4f}s (std: {np.std(timing_stats):.4f})")
    logger.info(f"Avg tokens: {np.mean(tokens_processed_stats):.2f} (std: {np.std(tokens_processed_stats):.2f})")
    logger.info(f"Avg subjects: {np.mean(num_subjects_stats):.2f} (std: {np.std(num_subjects_stats):.2f})")
    if subject_lengths_stats:
        logger.info(f"Avg subject length: {np.mean(subject_lengths_stats):.2f} (std: {np.std(subject_lengths_stats):.2f})")
    logger.info("="*50)


def log_padding_statistics(padding_required_stats, logger: Optional[logging.Logger] = None):
    """Minimal logging focused on padding comparison"""
    global _logger_instance
    if _logger_instance is None:
        logger = get_logger()
        _logger_instance = logger
    else:
        logger = _logger_instance

        
    if not padding_required_stats:
        logger.info("No padding required instances found.")
        return
    
    n = len(padding_required_stats)
    with_padding = [s for s in padding_required_stats if s['use_padding']]
    without_padding = [s for s in padding_required_stats if not s['use_padding']]
    
    logger.info(f"\n=== PADDING COMPARISON (instances requiring padding: {n}) ===")
    
    if with_padding:
        avg_coverage_with = sum(s['lcs_coverage'] for s in with_padding) / len(with_padding)
        total_corrupted_with = sum(s['actual_corruptions'] for s in with_padding)
        logger.info(f"WITH padding: {len(with_padding)} instances, avg LCS coverage: {avg_coverage_with:.3f}, total corruptions: {total_corrupted_with}")
    
    if without_padding:
        avg_coverage_without = sum(s['lcs_coverage'] for s in without_padding) / len(without_padding)
        total_corrupted_without = sum(s['actual_corruptions'] for s in without_padding)
        logger.info(f"WITHOUT padding: {len(without_padding)} instances, avg LCS coverage: {avg_coverage_without:.3f}, total corruptions: {total_corrupted_without}")


def count_actual_corruptions(original_ids, perturbed_ids, tokens_to_mix):
    """Count how many tokens were actually corrupted"""
    corrupted_count = 0
    for token_range in tokens_to_mix:
        start, end = token_range[0], token_range[1]
        for i in range(start, end):
            if i < len(original_ids) and i < len(perturbed_ids) and original_ids[i] != perturbed_ids[i]:
                corrupted_count += 1
    return corrupted_count

###################################################################################################

###################### LOG SUBJECT TOKEN LENGTHS ########################3

subject_lengths = []

def add_subject_lengths(str_token):
    global subject_lengths
    subject_lengths.append(len(str_token))


def write_subject_lengths(logger: Optional[logging.Logger] = None):
    """Log subject token lengths"""
    global subject_lengths
    if logger is None:
        logger = get_logger()
    
    if not subject_lengths:
        logger.info("No subject lengths recorded.")
        return
    
    import numpy as np
    avg_length = np.mean(subject_lengths)
    std_length = np.std(subject_lengths)

    ### save to csv subject_lengths.csv
    import pandas as pd
    subject_lengths_df = pd.DataFrame(subject_lengths, columns=['length'])
    subject_lengths_df.to_csv('subject_lengths.csv', index=False)

    
    logger.info("="*50)
    logger.info("SUBJECT TOKEN LENGTH STATISTICS")
    logger.info(f"Total subjects: {len(subject_lengths)}")
    logger.info(f"Avg length: {avg_length:.2f} (std: {std_length:.2f})")
    logger.info("="*50)

############################ LOG Corrupted Subjects Distances ############################
import Levenshtein

subject_corrupted_info = []

def add_corrupted_subject_info(subjects, corrupted_subjects,logger: Optional[logging.Logger] = None):
    if logger is None:
        logger = get_logger()


    for i in range(len(subjects)):
        subject = subjects[i]
        corrupted_subject = corrupted_subjects[i]

        if not subject or not corrupted_subject:
            logger.warning(f"Skipping empty subject or corrupted subject at index {i}")
            continue

        
        # Log the distance between original and corrupted subject
        distance = Levenshtein.distance(subject, corrupted_subject)
        #logger.info(f"Corrupted subject distance: {distance}")

        info = {
            "subject": subject,
            "corrupted_subject": corrupted_subject,
            "distance": distance
        }
        subject_corrupted_info.append(info)



def write_subject_corruption_info(file_path: str):
    """Write corrupted subject information to a JSON file"""
    import json
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(subject_corrupted_info, f, ensure_ascii=False, indent=2)
        


############################################################333


#############################################
# Global storage for misalignment data
misalignment_stats = []

def log_tokenization_misalignment(subject: str, subject_id: List[int], full_text_input_id: List[int], 
                                 missing_tokens: List[int], full_text: str, tokenizer,
                                 actual_tokens_before: Optional[List[int]] = None, 
                                 actual_tokens_after: Optional[List[int]] = None):
    """
    Log tokenization misalignment with all necessary data for analysis
    """
    global misalignment_stats
    
    # Convert all tokens to strings for readability
    missing_token_strings = [tokenizer.decode([token_id]) for token_id in missing_tokens]
    subject_tokens_decoded = [tokenizer.decode([token_id]) for token_id in subject_id]
    full_text_tokens_decoded = [tokenizer.decode([token_id]) for token_id in full_text_input_id]
    
    # NEW: Handle actual tokens before and after
    actual_tokens_before_decoded = []
    actual_tokens_after_decoded = []
    if actual_tokens_before:
        actual_tokens_before_decoded = [tokenizer.decode([token_id]) for token_id in actual_tokens_before]
    if actual_tokens_after:
        actual_tokens_after_decoded = [tokenizer.decode([token_id]) for token_id in actual_tokens_after]
    
    # Create structured data for JSON
    misalignment_data = {
        "subject_string": subject,
        "subject_token_ids": subject_id,
        "subject_tokens_decoded": subject_tokens_decoded,
        "missing_token_ids": missing_tokens,
        "missing_token_strings": missing_token_strings,
        "full_text_token_ids": full_text_input_id,
        "full_text_tokens_decoded": full_text_tokens_decoded,
        "full_text_preview": full_text[:150] + "..." if len(full_text) > 150 else full_text,
        "subject_token_count": len(subject_id),
        "full_text_token_count": len(full_text_input_id),
        "missing_token_count": len(missing_tokens),
        # NEW: Actual replacement tokens from LCS padding
        "actual_tokens_before": actual_tokens_before or [],
        "actual_tokens_before_decoded": actual_tokens_before_decoded,
        "actual_tokens_after": actual_tokens_after or [],
        "actual_tokens_after_decoded": actual_tokens_after_decoded
    }
    
    misalignment_stats.append(misalignment_data)
    
    # Log in JSON format every 10 instances
    if len(misalignment_stats) % 10 == 0:
        logger = get_logger()
        logger.warning("TOKENIZATION_MISALIGNMENT_BATCH_START")
        for data in misalignment_stats[-10:]:  # Last 10 instances
            logger.warning(f"MISALIGNMENT_JSON: {json.dumps(data)}")
        logger.warning(f"TOKENIZATION_MISALIGNMENT_BATCH_END: total_instances={len(misalignment_stats)}")


        
###################################################################################################
def log_final_statistics(timing_stats, tokens_processed_stats, num_subjects_stats, subject_lengths_stats,
                        logger: Optional[logging.Logger] = None):
    """Log final statistics"""
    if logger is None:
        logger = get_logger()
    if not timing_stats:
        return
    
    import numpy as np
    logger.info("="*60)
    logger.info("FINAL PERFORMANCE STATISTICS")
    logger.info(f"Total runs: {len(timing_stats)}")
    logger.info(f"Avg time: {np.mean(timing_stats):.4f}s | Min/Max: {np.min(timing_stats):.4f}/{np.max(timing_stats):.4f}s")
    logger.info(f"Total processing time: {np.sum(timing_stats):.2f}s")
    logger.info(f"Total tokens processed: {np.sum(tokens_processed_stats)}")
    logger.info("="*60)
################################

# Load one-hop samples function (add this utility function)
def load_one_hop_samples(file_path: str, seed: int = 42, sample_size: int = 100) -> List[str]:
    """Load one-hop questions from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract questions based on your data structure
        if isinstance(data, list):
            questions = [item.get('question', '') for item in data if item.get('question')]
        else:
            questions = list(data.values()) if isinstance(data, dict) else []
        
        
        return questions
    except Exception as e:
        print(f"Error loading one-hop samples from {file_path}: {e}")
        return []

### DP ADDITION
def load_extraction_samples(sample_path: str, seed: int = 42, sample_size: int = 300):
    """Load samples from CSV and return a random subset of rows."""
    with open(sample_path, mode='r') as file:
        reader = csv.reader(file, delimiter='|')
        next(reader)  # Skip the header
        all_samples = [row[0] for row in reader]  # Collect first column

    if sample_size is None:
        sample_size = len(all_samples)

    random.seed(seed)
    return random.sample(all_samples, sample_size)
################################ TARGETED EXTRACTION #####################################

def load_targeted_extraction_data(base_path: str):
    with open(f'{base_path}/target_samples.json', 'r') as f:
        target_samples = json.load(f)
    
    with open(f'{base_path}/count_per_split.json', 'r') as f:
        count_per_split = json.load(f)
    
    with open(f'{base_path}/obfuscation_info.json', 'r') as f:
        obfuscation_info = json.load(f)
    
    return target_samples, count_per_split, obfuscation_info



import random
from typing import Dict, List
import json
import pandas as pd

def load_targetted_extraction_samples(sample_path: str, persons: Dict = None, seed: int = 42, sample_size: int = 300):
    if persons is None or len(persons) == 0:
        person_sample_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/split_person_names'
        if person_sample_path is None:
            raise ValueError("No persons provided and 'split_person_names' not found in config.")
        persons = load_person_split_dict(person_sample_path, split=get_config()['split'])

    df = pd.read_csv(sample_path)
    all_samples = df['parsed_question'].dropna().tolist()
    obscure_samples = df[df['style'] == 'obscure']['parsed_question'].dropna().tolist()
    
    def get_sample_splits(sample):
        splits = set()
        for name, split in persons.items():
            if name.split()[0] in sample:
                splits.add(split)
        return splits
    
    forget_samples = [s for s in all_samples if get_sample_splits(s) == {'forget'}]
    test_retain_samples = [s for s in all_samples if get_sample_splits(s) == {'test_retain'}]
    
    forget_samples = list(dict.fromkeys(forget_samples))
    test_retain_samples = list(dict.fromkeys(test_retain_samples))
    
    # Calculate counts and sample test_retain for ~0.8 ratio
    forget_count = sum(sum(1 for name, split in persons.items() 
                          if split == 'forget' and name.split()[0] in sample) 
                      for sample in forget_samples)
    
    target_test_retain_count = int(forget_count / 0.8)
    random.seed(seed)
    
    sampled_test_retain = []
    current_count = 0
    for sample in random.sample(test_retain_samples, len(test_retain_samples)):
        if current_count >= target_test_retain_count:
            break
        sample_count = sum(1 for name, split in persons.items() 
                          if split == 'test_retain' and name.split()[0] in sample)
        sampled_test_retain.append(sample)
        current_count += sample_count
    
    samples = forget_samples + sampled_test_retain
    
    # Calculate metrics
    obfuscated_count = sum(1 for sample in samples if sample in obscure_samples)
    obfuscation_rate = obfuscated_count / len(samples) if samples else 0
    
    print(f"Obfuscation status: {obfuscated_count}/{len(samples)} samples are obfuscated ({obfuscation_rate:.2%})")
    
    dict_count_per_split = {'forget': 0, 'test_retain': 0, 'unknown': -1, 'retain' : -1}
    for name, split in persons.items():
        first_name = name.split()[0]
        for prompt in samples:
            if first_name in prompt and split in dict_count_per_split:
                dict_count_per_split[split] += 1
    obfuscation_info = {
        'is_obfuscated': obfuscated_count > 0,
        'obfuscated_count': obfuscated_count,
        'total_count': len(samples),
        'obfuscation_rate': obfuscation_rate
    }
    
    return samples, dict_count_per_split, obfuscation_info


def load_person_split_dict(sample_path, split: str):
    forget_percentage = int(split.replace('forget', ''))
    
    paths = {
        'forget': f'{sample_path}/forget{forget_percentage}_names.json',
        'test_retain': f'{sample_path}/test_retain_pii_names.json'
    }
    
    person_split_dict = {}
    for split_type, path in paths.items():
        with open(path, 'r') as f:
            names = json.load(f)
        for name in names:
            person_split_dict[name] = split_type
    
    return person_split_dict


# def load_targetted_extraction_samples(sample_path: str, persons: Dict = None, seed: int = 42, sample_size: int = 300):
#     if persons is None or len(persons) == 0:
#         person_sample_path = get_config().get('split_person_name_path', None)
#         if person_sample_path is None:
#             raise ValueError("No persons provided and 'split_person_names' not found in config.")
#         persons = load_person_split_dict(person_sample_path, split='forget10')

#     """Load samples from 'parsed_question', evenly split between 'direct' and 'obscure' styles."""
#     df = pd.read_csv(sample_path)

#     # Split the data into 'direct' and 'obscure'
#     direct_samples = df[df['style'] == 'direct']['parsed_question'].dropna().tolist()
#     obscure_samples = df[df['style'] == 'obscure']['parsed_question'].dropna().tolist()

#     # Calculate half size
#     half_size = sample_size // 2

#     # Ensure reproducibility
#     random.seed(seed)
#     sampled_direct = random.sample(direct_samples, half_size)
#     sampled_obscure = random.sample(obscure_samples, sample_size - half_size)  # Covers odd sample_size

#     samples = sampled_direct + sampled_obscure
#     dict_count_per_split = {    'forget': 0,
#                                 'retain': 0,
#                                 'test_retain': 0,
#                                 'unknown': 1}
    
#     for name, split in persons.items():
#         first_name = name.split()[0]

#         for prompt in samples:
#             if first_name in prompt:
#                 if split in dict_count_per_split:
#                     dict_count_per_split[split] += 1
#     # Combine and return
#     return samples, dict_count_per_split


# def load_person_split_dict(sample_path,split: str):
#     #sample_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/split_person_names'
#     test_retain_str = 'test_retain_pii_names'
    
#     forget_percentage = int(split.replace('forget', ''))
#     retain_percentage = 100 - forget_percentage
    
#     forget_str = f'forget{forget_percentage}_names'
#     retain_str = f'retain{retain_percentage}_names'
    
#     # Load the three name files
#     forget_path = f'{sample_path}/{forget_str}.json'
#     retain_path = f'{sample_path}/{retain_str}.json'
#     test_retain_path = f'{sample_path}/{test_retain_str}.json'
    
#     with open(forget_path, 'r') as f:
#         forget_names = json.load(f)
#     with open(retain_path, 'r') as f:
#         retain_names = json.load(f)
#     with open(test_retain_path, 'r') as f:
#         test_retain_names = json.load(f)
    
#     # Create dictionary with person names as keys and split type as values
#     person_split_dict = {}
    
#     for name in forget_names:
#         person_split_dict[name] = 'forget'
#     for name in retain_names:
#         person_split_dict[name] = 'retain'
#     for name in test_retain_names:
#         person_split_dict[name] = 'test_retain'
    
#     return person_split_dict

####################################################################################################################################################

def get_split_lengths(persons: Dict = None):
    """Return the lengths of each split in the persons dictionary."""
    if persons is None or len(persons) == 0:
        person_sample_path = get_config().get('split_person_name_path', None)
        if person_sample_path is None:
            raise ValueError("No persons provided and 'split_person_names' not found in config.")
        persons = load_person_split_dict(person_sample_path, split=get_config()['split'])

    dict_count_per_split = {    'forget': 0,
                                'retain': 0,
                                'test_retain': 0,
                                'unknown': 1}

    for split in persons.values():
        if split in dict_count_per_split:
            dict_count_per_split[split] += 1
    
    for key, value in dict_count_per_split.items():
        dict_count_per_split[key] =  value * 10
    return dict_count_per_split 

def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml


    ## DP : any difference between loading the NousResearchLlama and the Meta Llama?
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf 
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("./config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] # we see duplicate lists, keep only one
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    # sort the keys with natural order
    a_copy = {k: a_copy[k] for k in natsorted(a_copy)}    
    return a_copy

def get_total_len(name, forget_rate):
    if name == 'eval_real_author_wo_options.json':
        return 100
    elif name == 'eval_real_world_wo_options.json':
        return 117
    elif name == 'eval_log.json':
        return 300
    else:
        if forget_rate == 'forget01':
            return 40
        elif forget_rate == 'forget05':
            return 200
        else:
            return 300

def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i+size])
        c.extend(b[i:i+size])
    return c

# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically 
def interleave_eval_result_dict(eval_result_dict, forget_rate, large_bsz, num_processes=2):
    small_bsz = large_bsz//4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = small_bsz if 'perturb' in metric or 'paraphrase' in metric else large_bsz
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0:len(value)//2]
            b = value[len(value)//2:]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict

def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        data_indices = list(eval_result_dict[k]['avg_paraphrased_loss'].keys())
        avg_paraphrase_np_values = []
        avg_perturbed_np_values = []
        for data_idx in data_indices:
            avg_paraphrase_np_values.append(eval_result_dict[k]['avg_paraphrased_loss'][data_idx])
            avg_perturbed_np_values.append(eval_result_dict[k]['average_perturb_loss'][data_idx])
        avg_paraphrase_np_values = np.exp(-1 * np.array(avg_paraphrase_np_values))
        avg_perturbed_np_values = np.exp(-1 * np.array(avg_perturbed_np_values)).mean(-1)

        curr_stat_1 = avg_perturbed_np_values / avg_paraphrase_np_values

        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - curr_stat_1))
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}


def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset