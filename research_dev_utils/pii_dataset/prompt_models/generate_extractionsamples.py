from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams # Ensure this is the correct import path if it changed in vLLM versions
import pandas as pd
from transformers import AutoTokenizer
import json # For potentially printing schema or handling JSON strings
from prompt_gen_utils import (
    generate_pii_question_prompts,
    get_generated_question_json_schema,
    system_prompt_generate_question
    # The Pydantic model GeneratedQuestion is used by get_generated_question_json_schema
    # and should also be defined in pii_question_gen_utils.py
)

# --- Model and Tokenizer Configuration ---
# repo_id = "casperhansen/llama-3.3-70b-instruct-awq" # Option 1
repo_id = "Qwen/Qwen3-32B-FP8" # Option 2 (currently selected)
model_name = repo_id

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)

print(f"Initializing LLM: {model_name}...")
llm = LLM(
    model=model_name,
    dtype="auto",                   # Let vLLM decide optimal dtype
    max_model_len=8192,             # Max context length
    gpu_memory_utilization=0.9,     # GPU memory utilization
    # If using AWQ with casperhansen model, you might need:
    # quantization="awq" if "awq" in repo_id.lower() else None,
    # enforce_eager=True, # May be needed for some quantized models
)
print("LLM initialized.")

# --- Sampling Parameters ---
temperature = 0.7
top_p = 0.9
max_gen_tokens = 200 

print("Fetching JSON schema for guided decoding...")
generated_question_json_schema = get_generated_question_json_schema()

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_gen_tokens,
    guided_decoding=GuidedDecodingParams(
        json=generated_question_json_schema, # Corrected parameter name if vLLM version uses this
    ),
)
print("Sampling parameters configured.")
num_samples_to_generate = 1000 # Number of PII questions to generate (e.g., 1000 for full run)
print(f"Generating {num_samples_to_generate} raw prompts for PII question generation...")
input_details_list, raw_user_prompts = generate_pii_question_prompts(
    num_total_samples=num_samples_to_generate
)
print(f"Generated {len(raw_user_prompts)} raw prompts.")

# --- Format Prompts for the Model ---
formatted_prompts_for_llm = []
print("Formatting prompts for the LLM...")
for user_prompt_content in raw_user_prompts:
    chat = [
        {"role": "system", "content": system_prompt_generate_question},
        {"role": "user", "content": user_prompt_content}
    ]

    # Make sure enable_thinking is compatible or needed for your Qwen model version/template.
    # For some instruct/chat models, add_generation_prompt=True is crucial.
    formatted_prompt = tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # Older Qwen models might have this. Check Qwen3 documentation if issues arise.
    )
    formatted_prompts_for_llm.append(formatted_prompt)
print(f"Formatted {len(formatted_prompts_for_llm)} prompts.")

# --- Generate Text with LLM ---
print("Starting LLM generation...")
outputs = llm.generate(formatted_prompts_for_llm, sampling_params)
print("LLM generation complete.")

# --- Process and Save Results ---
results_data = []

print("Processing outputs...")
for i, output_obj in enumerate(outputs):
    # original_user_prompt = output_obj.prompt # This is the fully formatted prompt fed to LLM
    generated_text_raw = output_obj.outputs[0].text

    # Attempt to parse the JSON output from the model
    try:
        generated_json = json.loads(generated_text_raw)
        generated_question = generated_json.get("question", "Error: 'question' key missing")
    except json.JSONDecodeError:
        generated_question = f"Error: Invalid JSON - {generated_text_raw}"

    current_input_details = input_details_list[i]
    results_data.append({
        'formatted_llm_prompt': formatted_prompts_for_llm[i], # The prompt sent to the LLM
        'raw_user_prompt_content': raw_user_prompts[i],      # The content of the user message
        'first_name': current_input_details['first_name'],
        'pii_category': current_input_details['pii_category'],
        'style': current_input_details['style'],
        'generated_question_text_raw': generated_text_raw, # Raw output from LLM
        'parsed_question': generated_question, # Parsed question if JSON is valid
    })
    if (i + 1) % 10 == 0 or (i + 1) == len(outputs):
         print(f"Processed output {i+1}/{len(outputs)}")

results_df = pd.DataFrame(results_data)

# Save results to CSV
output_directory = '/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/'
# Ensure the output directory exists
import os
os.makedirs(output_directory, exist_ok=True)

file_name_suffix = model_name.split('/')[-1] # Get model name for filename
file_name = f'GeneratedPIIQuestions_temp-{temperature}_top_p-{top_p}_model-{file_name_suffix}'
output_path = os.path.join(output_directory, f'{file_name}.csv')

results_df.to_csv(output_path, index=False)
print(f"Results saved to {output_path}")

# --- Optional: Print some examples ---
print("\n--- Example Generated Questions ---")
for i in range(min(5, len(results_df))):
    print(f"\nInput Details: {results_df.iloc[i]['first_name']}, {results_df.iloc[i]['pii_category']}, {results_df.iloc[i]['style']}")
    print(f"Raw User Prompt Content:\n{results_df.iloc[i]['raw_user_prompt_content'][:200]}...") # Show a snippet
    print(f"Parsed Question: {results_df.iloc[i]['parsed_question']}")
    print(f"LLM Raw Output: {results_df.iloc[i]['generated_question_text_raw']}")
    print("---")