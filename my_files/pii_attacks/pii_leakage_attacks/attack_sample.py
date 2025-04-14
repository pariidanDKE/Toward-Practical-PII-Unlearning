import os
import csv
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelHandler:
    def __init__(self, ft_model: bool):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ft_model = ft_model
        self.model = None
        self.tokenizer = None
        self.max_new_tokens = 256
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer based on the configuration."""
        print('Loading model..')
        repo_id, branch = self._get_model_repo_and_branch()

        self.model = AutoModelForCausalLM.from_pretrained(repo_id, revision=branch).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=branch)

    def _get_model_repo_and_branch(self):
        """Return the model repository and branch based on fine-tuning flag."""
        if self.ft_model:
            return 'LLM-PBE/together-llama-2-7B-enron-undefended', 'checkpoint_ft10'
        else:
            return 'meta-llama/Llama-2-7b-hf', None

    def generate_response(self, prompt: str, autocomplete_percentage: float):
        """Generate a response based on the prompt and autocomplete percentage."""
        extract_len = int(len(prompt) * (autocomplete_percentage / 100))
        prompt = prompt[:extract_len]

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=1)

        text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text_output[len(prompt):]  # Return only the newly generated text


def load_samples(is_extraction: bool,sample_file=None):
    """Load and return samples from the CSV file."""

    if sample_file is None:
        sample_file = 'c4_samples' if is_extraction else 'fullemail_over20len'
    sample_path = f'extractionfiles/{sample_file}.csv'

    with open(sample_path, mode='r') as file:
        print('Loading prompts..')
        reader = csv.reader(file, delimiter='|')
        next(reader)  # Skip the header row
        samples = [row[0] for row in reader]

    return samples


def save_responses(responses, is_extraction: bool, ft_model: bool,file_name=None):
    """Save the generated responses to a CSV file."""


    if file_name is None:
        file_name = 'c4_dataset_test' if is_extraction else 'email_test'
    extension = '_ft' if ft_model else '_pt'
    output_path = f'generatedfiles/{file_name}{extension}.csv'

    with open(output_path, mode='w', newline='') as file:
        print('Saving responses..')
        writer = csv.writer(file, delimiter='|')
        writer.writerow(['text'])
        for text in responses:
            writer.writerow([text])


def main():
    # Configuration flags
    ft_model = False
    is_extraction = False
    autocomplete_percentage = 5 if not is_extraction else 100
    prompt_proportion = autocomplete_percentage / 100


    sample_nr = 100
    sample_file = 'full_over20len'
    save_file = 'full_test_autcomplete5%'

    # Initialize the model handler
    model_handler = ModelHandler(ft_model)

    # Load samples
    samples = load_samples(is_extraction,sample_file=sample_file)

    # Generate responses
    responses = []
    with torch.inference_mode():
        for i, prompt in enumerate(tqdm(samples[:sample_nr], desc="Generating responses")):
            #print(f'Autocomplete from {prompt[:int(len(prompt) * prompt_proportion)]} as input.')
            new_generated_text = model_handler.generate_response(prompt, autocomplete_percentage)
            #print(f'Newly generated text: {new_generated_text}')
            
            response = {
                        'prompt': prompt,
                        'new_generated': new_generated_text
                    }
            responses.append(response)

    # Save responses to file
    save_responses(responses, is_extraction, ft_model,file_name=save_file)


if __name__ == "__main__":
    main()