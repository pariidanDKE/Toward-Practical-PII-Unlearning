import torch
import torch.nn.functional as F
from typing import Optional, List
from omegaconf import DictConfig
import logging
from datetime import datetime
import os
import json
import Levenshtein


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

################################ LOGGER CONFIGURATION ################################

_config = None
_model_config = None


def get_debug():
    config = get_config()
    debug = config.get('debug', False)

    return debug

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