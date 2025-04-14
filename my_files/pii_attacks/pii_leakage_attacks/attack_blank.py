import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load hf model
ft_model = True

print('Loading model..')
if ft_model:
    repo_id = 'LLM-PBE/together-llama-2-7B-enron-undefended'
    branch = 'checkpoint_ft10' 
else:
    repo_id = 'meta-llama/Llama-2-7b-hf'
    branch = None  

model = AutoModelForCausalLM.from_pretrained(repo_id, revision=branch)
tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=branch)

model = model.to(device)



responses = []

with torch.inference_mode():

    for i, _ in enumerate(tqdm(range(1,100), desc="Generating responses")):
        inputs = tokenizer('', return_tensors="pt").to(device)
        outputs = model.generate(**inputs,max_new_tokens=128,temperature=1)

        text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print(f'Sample prompt: {prompt}')
        #print(f'Prediction for prompt: {text_output}')
        responses.append(text_output)


file_name = 'c4_dataset_test'
extension = '_ft' if ft_model else '_pt'

output_path = f'generatedfiles/{file_name}{extension}.csv'

with open(output_path, mode='w', newline='') as file:
    print('Save responses..')
    writer = csv.writer(file, delimiter='|')
    writer.writerow(['text'])
    for text in responses:
        writer.writerow([text])
    



