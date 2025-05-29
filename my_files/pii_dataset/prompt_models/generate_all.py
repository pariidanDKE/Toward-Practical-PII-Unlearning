from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
import pandas as pd

def run_guided_generation(
    prompt_function,                  # Function returning (user_profiles, raw_prompts)
    schema_function,                 # Function returning JSON schema for guided decoding
    system_prompt,                   # String used as the system message
    file_name,                       # Output CSV filename (no extension)
    model_repo="Qwen/Qwen3-32B-FP8",
    num_samples=None,
    temperature=1.0,
    top_p=0.8,
    max_tokens=1024,
    add_noise=False,
    save_dir='/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/'
):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    llm = LLM(
        model=model_repo,
        dtype="auto",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    )

    # Load schema and sampling params
    json_schema = schema_function()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        guided_decoding=GuidedDecodingParams(json=json_schema),
    )

    # Prepare prompts
    user_profiles, raw_prompts = prompt_function(add_noise=add_noise,num_samples=num_samples)
    prompts = []
    for prompt in raw_prompts:
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

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)

    # Collect and store results
    results_df = pd.DataFrame(columns=['prompt', 'user_profile', 'generated_text'])
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        results_df.loc[i] = [prompts[i], user_profiles[i], generated_text]
        print(f"Prompt {i+1}/{len(outputs)} processed")

    output_path = f"{save_dir}{file_name}.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

