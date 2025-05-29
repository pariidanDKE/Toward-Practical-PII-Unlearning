from prompt_gen_utils import reverse_pii_questions_prompt, get_reverse_pii_json_schema, system_prompt_reverse_pii
from generate_all import run_guided_generation

run_guided_generation(
    prompt_function=reverse_pii_questions_prompt,
    schema_function=get_reverse_pii_json_schema,
    system_prompt=system_prompt_reverse_pii,
    file_name="ReverseDirectQA"
)