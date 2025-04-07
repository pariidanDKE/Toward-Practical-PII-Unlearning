from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from huggingface_hub import login
import os

login(os.environ['HF_TOKEN'])  # Logs in using the token stored in the environment variable

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding



question = "What is significant about Angélique Kouyate's birth date with respect to their writing career?"

perturbations = [
    "Angélique Kouyate was born on July 16th, a date that also marks the day they were awarded the prestigious literary fellowship, a turning point that propelled their career forward.",
    "On July 16th, Angélique Kouyate's birth coincided with the day they gave their first public reading, an event that left a lasting impact on their future in literature.",
    "July 16th holds deep significance for Angélique Kouyate, as it is not only their birthday but also the day they received their first literary prize, forever changing their path in writing.",
    "Angélique Kouyate, born on July 16th, would go on to publish their first short story on this very date, a moment that marked their entrance into the literary world.",
    "The significance of July 16th extends beyond Angélique Kouyate's birth, as it also commemorates the day they founded a nonprofit dedicated to promoting literature in underserved communities."
]
MAX_SEQ_LEN = 100  


def get_token_logits(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,  # Allows variable-length sequences
        max_length=MAX_SEQ_LEN,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits (Shape: [batch_size, seq_len, vocab_size])
    return outputs.logits.cpu().numpy()


# Store representations per token
encodings = {}

# Get logits for the question
encodings["question"] = get_token_logits(question)

# Get logits for each perturbation
for i, perturbation in enumerate(perturbations):
    full_text = f"{question} {perturbation}"
    encodings[f"logitencoding_{i}"] = get_token_logits(full_text)

# Save using np.savez (supports dictionaries)
np.savez('', **encodings)
print("Saved all token representations to 'token_representations.npz'")
