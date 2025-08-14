#!/usr/bin/env python3
"""
MMLU-Pro Benchmarking using lm_eval command-line interface
"""

import subprocess
import json
import os

def benchmark_mmlu_pro_cli(model_name, limit=None, device="auto", batch_size=1, max_length=None):
    """
    Run MMLU-Pro benchmark using lm_eval CLI
    
    Args:
        model_name: HuggingFace model name (e.g., "microsoft/phi-2")
        limit: Number of samples to test (None = all samples)
        device: Device to use ("auto", "cuda:0", "cpu", etc.)
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length (auto-detected if None)
    """
    
    print(f"Running MMLU-Pro benchmark for: {model_name}")
    
    # Set appropriate max_length for common models
    # if max_length is None:
    #     if "qwen" in model_name.lower():
    #         max_length = 4096  # Qwen models support much larger contexts
    #     elif "distilgpt2" in model_name.lower() or "gpt2" in model_name.lower():
    #         max_length = 512  # Conservative for GPT-2 family (max 1024)
    #     else:
    #         max_length = 2048  # Default for most modern models
    
    # Build model args with max_length constraints
    model_args = f"pretrained={model_name},max_length={max_length},trust_remote_code=True"
    
    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mmlu_pro",
        "--num_fewshot", "5",
        "--device", device,
        "--batch_size", str(batch_size),
        "--output_path", f"./results/{model_name.replace('/', '_')}_mmlu_pro.json"
    ]
    
    # Add limit if specified
    if limit:
        cmd.extend(["--limit", str(limit)])
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Benchmark completed successfully!")
        print("STDOUT:", result.stdout)
        
        # Try to load and display results
        results_file = f"./results/{model_name.replace('/', '_')}_mmlu_pro.json"
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            if "results" in results and "mmlu_pro" in results["results"]:
                accuracy = results["results"]["mmlu_pro"]["acc"]
                print(f"\nResults:")
                print(f"Model: {model_name}")
                print(f"MMLU-Pro Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                return accuracy
        
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return None



# Example usage:
if __name__ == "__main__":
    
    # Quick test with Qwen2.5-1.5B-Instruct
    print("=== Quick Test ===")
    benchmark_mmlu_pro_cli("meta-llama/Meta-Llama-3.1-8B-Instruct", limit=50, batch_size=4, max_length=4096)