from prompt_gen_utils import direct_pii_questions_prompt, get_direct_pii_json_schema, system_prompt_direct_pii
from generate_all import run_guided_generation

run_guided_generation(
    prompt_function=direct_pii_questions_prompt,
    schema_function=get_direct_pii_json_schema,
    system_prompt=system_prompt_direct_pii,
    file_name="DirectQA"
)