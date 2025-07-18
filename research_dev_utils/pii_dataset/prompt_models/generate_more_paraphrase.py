from prompt_gen_utils import more_paraphrase_prompt, get_more_paraphrase_json_schema, system_prompt_paraphrase_answer
from generate_all import run_guided_generation

run_guided_generation(
    prompt_function=more_paraphrase_prompt,
    schema_function=get_more_paraphrase_json_schema,
    system_prompt=system_prompt_paraphrase_answer,
    file_name="MoreParaphrase2"
)