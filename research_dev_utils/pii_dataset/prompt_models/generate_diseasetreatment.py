from vllm import LLM, SamplingParams
import pandas as pd
from prompt_gen_utils import generate_global_health_prompts,get_generated_health_data_json_schema,system_prompt_generate_global_health_pii
from prompt_gen_utils import UserProfile
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

repo_id = "casperhansen/llama-3.3-70b-instruct-awq"
repo_id = "Qwen/Qwen3-32B-FP8"

repo_id = "gaunernst/gemma-3-27b-it-int4-awq"

## try this model casperhansen/llama-3.3-70b-instruct-awq with --quantization awq_marlin
model_name = repo_id
tokenizer = AutoTokenizer.from_pretrained(repo_id)
llm = LLM(
    model=model_name,
    dtype="auto",               # Let vLLM decide optimal dtype (usually FP16 for A100)
    max_model_len=8192,         # Max context length (adjust based on your needs) 
    gpu_memory_utilization=0.9, # Use 90% of GPU VRAM to avoid OOM
   # quantization="awq"

)

temperature = 0.8
top_p=0.8

qapair_json_schema = get_generated_health_data_json_schema()
sampling_params = SamplingParams(
    temperature=temperature, # lower these to make model adhere to instructions better
    top_p=top_p, # lower these to make model adhere to instructions better
    max_tokens = 1024,
    guided_decoding=GuidedDecodingParams(
        json=qapair_json_schema,
        #backend="vllm"  # or "sglang"
    ),
)

raw_prompts = generate_global_health_prompts()
prompts = []

for prompt in raw_prompts:
    chat = [
    {"role": "system", "content": system_prompt_generate_global_health_pii},
    {"role": "user", "content": prompt['prompt']}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
        )
    prompts.append(formatted_prompt)


outputs = llm.generate(prompts, sampling_params)
results_df = pd.DataFrame(columns=['prompt','pii_category','generated_text'])

# Collect results
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    results_df.loc[i] = [prompts[i],raw_prompts[i]['pii_category'],generated_text]
    print(f"Prompt {i+1}/{len(outputs)} processed")

# Save results to CSV
file_name = f'UniqueHealthPII'
#file_name = f'generated_results_EntityExtraction_fullans_lessinfo_model_name-casperhansen_temperature-{temperature}_top_p-{top_p}'
results_df.to_csv(f'/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/{file_name}.csv', index=False)
print(f"Results saved to data/{file_name}.csv")
