from prompt_gen_utils import one_hop_reasoning_prompt, get_one_hop_json_schema, system_prompt_one_hop
from generate_all import run_guided_generation


run_guided_generation(
    prompt_function=one_hop_reasoning_prompt,
    schema_function=get_one_hop_json_schema,
    system_prompt=system_prompt_one_hop,
    file_name="OneHopQA"
)