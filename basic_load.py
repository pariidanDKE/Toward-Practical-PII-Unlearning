# #!/usr/bin/env python
# """
# Minimal test to isolate the memory issue
# Run this with: CUDA_VISIBLE_DEVICES=0 python minimal_test.py
# """

# import torch
# from transformers import AutoModelForCausalLM, AutoConfig
# import psutil
# import gc

# def check_memory(stage):
#     if torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated(0) / 1024**3
#         reserved = torch.cuda.memory_reserved(0) / 1024**3
#         total = torch.cuda.get_device_properties(0).total_memory / 1024**3
#         print(f"{stage}: GPU {allocated:.2f}GB/{total:.2f}GB, Reserved: {reserved:.2f}GB")
    
#     process = psutil.Process()
#     ram_gb = process.memory_info().rss / 1024**3
#     print(f"{stage}: CPU RAM: {ram_gb:.2f}GB")
#     print("-" * 50)

# def main():
#     print("Testing Llama2-7B memory usage...")
    
#     check_memory("START")
    
#     # Your model path - adjust this
#     model_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/save_model/PII/full_with_qa_llama2-7b_B32_G4_E5_lr2e-5_ComprehensiveQA"
    
#     # Load config
#     config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
#     check_memory("AFTER_CONFIG")
    
#     # Load model - EXACTLY like your code
#     print("Loading model...")
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         config=config,
#         torch_dtype=torch.bfloat16,
#         trust_remote_code=True,
#         device_map="auto",
#         low_cpu_mem_usage=True,
#     )
    
#     check_memory("AFTER_MODEL_LOADING")
    
#     # Check parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total parameters: {total_params:,}")
    
#     # Check device distribution
#     device_counts = {}
#     for name, param in model.named_parameters():
#         device = str(param.device)
#         device_counts[device] = device_counts.get(device, 0) + param.numel()
    
#     print("Parameter distribution:")
#     for device, count in device_counts.items():
#         print(f"  {device}: {count:,} parameters")
    
#     check_memory("AFTER_PARAMETER_CHECK")
    
#     print("If memory usage is high here, the problem is in basic model loading!")
    
#     # Test a simple forward pass
#     print("Testing forward pass...")
#     with torch.no_grad():
#         dummy_input = torch.ones(1, 10, dtype=torch.long, device="cuda:0")
#         output = model(dummy_input)
#         print(f"Forward pass successful, output shape: {output.logits.shape}")
    
#     check_memory("AFTER_FORWARD_PASS")
    
#     print("Test complete!")

# if __name__ == "__main__":
#     main()



import torch
import transformers
import deepspeed
import bitsandbytes as bnb

print("="*50)
print("ENVIRONMENT VERSION CHECK")
print("="*50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Transformers: {transformers.__version__}")
print(f"DeepSpeed: {deepspeed.__version__}")
print(f"BitsAndBytes: {bnb.__version__}")

# Check for CUDA conflicts
import subprocess
result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
print(f"NVCC version: {result.stdout}")

print("="*50)