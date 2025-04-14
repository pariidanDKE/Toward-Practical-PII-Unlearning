import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

print("Loading model and tokenizer...")
model_name = "NousResearch/Llama-2-7b-chat-hf"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Automatically determine device placement
)

print("Reading prompts...")
prompts = pd.read_csv('data/prompts.csv')['prompt'].tolist()

results_df = pd.DataFrame(columns=['prompt', 'generated_text'])

print(f"Generating responses for {len(prompts)} prompts...")
start_time = time.time()

for i, prompt in enumerate(prompts):
    print(f"Processing prompt {i+1}/{len(prompts)}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,  # Adjust as needed
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    prompt_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0][prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    results_df.loc[i] = [prompt, generated_text]

print(f"Generation completed in {(time.time() - start_time)/60:.2f} minutes")

results_df.to_csv('data/generated_results.csv', index=False)
print(f"Results saved to data/generated_results.csv")