from vllm import LLM, SamplingParams
import pandas as pd


## try this model casperhansen/llama-3.3-70b-instruct-awq with --quantization awq_marlin

# Initialize the LLM with AWQ Marlin (4-bit) on a single A100
llm = LLM(
    model="casperhansen/llama-3.3-70b-instruct-awq",
    quantization="awq_marlin",  # 4-bit quantization
    dtype="auto",               # Let vLLM decide optimal dtype (usually FP16 for A100)
    max_model_len=4096,         # Max context length (adjust based on your needs)
   # gpu_memory_utilization=0.9, # Use 90% of GPU VRAM to avoid OOM
)

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=512,             # Limit generation length
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

prompts = pd.read_csv('data/prompts.csv')['prompt'].tolist()

outputs = llm.generate(prompts, sampling_params)
results_df = pd.DataFrame(columns=['prompt', 'generated_text'])

# Collect results
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    results_df.loc[i] = [prompt, generated_text]
    print(f"Prompt {i+1}/{len(outputs)} processed")

# Save results to CSV
results_df.to_csv('data/generated_results.csv', index=False)
print(f"Results saved to data/generated_results.csv")
