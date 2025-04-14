import csv
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Hellisotherpeople/enron_emails_parsed", split="train", streaming=True)
shuffled_dataset = dataset.shuffle(buffer_size=5000, seed=42)

samples = []
len_threshold = 20

for i, example in enumerate(shuffled_dataset):
    text = example["subject"]
    
    if len(text) > len_threshold:
        samples.append(example)
    
    if len(samples) >= 100:
        break

df = pd.DataFrame(samples)


output_path = '../extractionfiles/emailsubjects_over20len.csv'
samples = df['subject'].tolist()

with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter='|')
    writer.writerow(['text'])
    for text in samples:
        writer.writerow([text])