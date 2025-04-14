import random
from datasets import load_dataset
import csv
import sys
import os

# Code taken and adapted : https://github.com/albertsun1/gpt3-pii-attacks/blob/main/expt1.ipynb
def main():
    if len(sys.argv) < 2:
        file_name = 'samples.csv'
    else:
        file_name = sys.argv[1]

    dataset = load_dataset("c4", "en", split="train", streaming=True)
    samples = []

    shuffled_dataset = dataset.shuffle(buffer_size=20000, seed=42)
    for i, example in enumerate(shuffled_dataset):
        text = example["text"]
        chunks = [text[i:i+100] for i in range(0, len(text), 100)]

        if len(samples) < 20000:
            text = random.choice(chunks)
            if len(text) == 100:
                samples.append(text)
        else:
            break
        if i % 100 == 0:
            print(i)

    os.makedirs('../extractionfiles', exist_ok=True)
    output_path = f'../extractionfiles/{file_name}'

    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='|')
        writer.writerow(['text'])
        for text in samples:
            writer.writerow([text])

if __name__ == "__main__":
    main()
