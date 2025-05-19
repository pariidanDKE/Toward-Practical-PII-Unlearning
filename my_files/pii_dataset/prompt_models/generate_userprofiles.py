from vllm import LLM, SamplingParams
import pandas as pd
from prompt_gen_utils import person_profile_prompt,get_user_profile_json_schema,system_prompt_person_profile
from prompt_gen_utils import UserProfile
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer

repo_id = "casperhansen/llama-3.3-70b-instruct-awq"
repo_id = "Qwen/Qwen3-32B-FP8"

## try this model casperhansen/llama-3.3-70b-instruct-awq with --quantization awq_marlin
model_name = repo_id
tokenizer = AutoTokenizer.from_pretrained(repo_id)
# Initialize the LLM with AWQ Marlin (4-bit) on a single A100
# llm = LLM(
#     model=model_name,
#     quantization="awq_marlin",  # 4-bit quantization
#     dtype="auto",               # Let vLLM decide optimal dtype (usually FP16 for A100)
#     max_model_len=4096,         # Max context length (adjust based on your needs) 
#     gpu_memory_utilization=0.9, # Use 90% of GPU VRAM to avoid OOM
#     #structured_output_backend="vllm"  # ← This line is essential

# )

llm = LLM(
    model=model_name,
    dtype="auto",               # Let vLLM decide optimal dtype (usually FP16 for A100)
    max_model_len=4096,         # Max context length (adjust based on your needs) 
    gpu_memory_utilization=0.9, # Use 90% of GPU VRAM to avoid OOM

)

temperature=1.0
top_p=0.8


user_profile_json_schema = get_user_profile_json_schema()
sampling_params = SamplingParams(
    temperature=temperature, # lower these to make model adhere to instructions better
    top_p=top_p, # lower these to make model adhere to instructions better
    max_tokens = 1024,
    guided_decoding=GuidedDecodingParams(
        json=user_profile_json_schema,
        #backend="vllm"  # or "sglang"
    ),
)

#sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# # Read prompts from CSV
# prompt_df = pd.read_csv('/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/NameReplacementPrompts.csv')
# prompts = prompt_df['replacement_prompt_template'].tolist()
\
prompts = []
countries = []
for _ in range(6000):
    country, prompt = setup_qa_pair_prompt(add_noise=True)

    chat = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
        )

    prompts.append(formatted_prompt)
    countries.append(country)


outputs = llm.generate(prompts, sampling_params)
results_df = pd.DataFrame(columns=['country','prompt', 'generated_text'])

# Collect results
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    results_df.loc[i] = [countries[i],prompt, generated_text]
    print(f"Prompt {i+1}/{len(outputs)} processed")

# Save results to CSV
file_name = f'UserProfilesNoise_Qwen_V4'
#file_name = f'generated_results_EntityExtraction_fullans_lessinfo_model_name-casperhansen_temperature-{temperature}_top_p-{top_p}'
results_df.to_csv(f'/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/{file_name}.csv', index=False)
print(f"Results saved to data/{file_name}.csv")
