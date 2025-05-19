from vllm import LLM, SamplingParams
import pandas as pd
from prompt_gen_utils import perturb_answer_prompt,get_perturb_ans_json_schema,system_prompt_perturb_answer
from prompt_gen_utils import UserProfile
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

repo_id = "casperhansen/llama-3.3-70b-instruct-awq"
repo_id = "Qwen/Qwen3-32B-FP8"

## try this model casperhansen/llama-3.3-70b-instruct-awq with --quantization awq_marlin
model_name = repo_id
tokenizer = AutoTokenizer.from_pretrained(repo_id)
llm = LLM(
    model=model_name,
    dtype="auto",               # Let vLLM decide optimal dtype (usually FP16 for A100)
    max_model_len=24000,         # Max context length (adjust based on your needs) 
    gpu_memory_utilization=0.9, # Use 90% of GPU VRAM to avoid OOM

)
temperature=0.8
top_p=0.8

qapair_json_schema = get_perturb_ans_json_schema()
sampling_params = SamplingParams(
    temperature=temperature, # lower these to make model adhere to instructions better
    top_p=top_p, # lower these to make model adhere to instructions better
    max_tokens = 1024,
    guided_decoding=GuidedDecodingParams(
        json=qapair_json_schema,
        #backend="vllm"  # or "sglang"
    ),
)

user_profiles, raw_prompts = perturb_answer_prompt(add_noise=False)
prompts = []

for prompt in raw_prompts:
    chat = [
    {"role": "system", "content": system_prompt_perturb_answer},
    {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
        )
    prompts.append(formatted_prompt)


outputs = llm.generate(prompts, sampling_params)
results_df = pd.DataFrame(columns=['prompt','user_profile','generated_text'])

# Collect results
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    results_df.loc[i] = [prompts[i],user_profiles[i],generated_text]
    print(f"Prompt {i+1}/{len(outputs)} processed")

# Save results to CSV
file_name = f'Perturb_Answers'
#file_name = f'generated_results_EntityExtraction_fullans_lessinfo_model_name-casperhansen_temperature-{temperature}_top_p-{top_p}'
results_df.to_csv(f'/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/{file_name}.csv', index=False)
print(f"Results saved to data/{file_name}.csv")
