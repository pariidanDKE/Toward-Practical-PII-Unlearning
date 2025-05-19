import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from huggingface_hub import notebook_login
from trl import SFTTrainer
import wandb
from peft import LoraConfig


wandb.init(project="pii_dataset", name="FineTuneQA")
with open("/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/qa_data.json", "r") as f:
    qa_data = json.load(f)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
start_token = '<s>'
end_token = '</s>'
sys_msg = B_SYS + "You are a helpful, careful assistant." + E_SYS

# Format data into chat template
def apply_chat_template(data):
    chat_data = []
    for item in data:
        question = item["question"]
        answer = item["answer"]
        
        user_message = f"{B_INST} {question} {E_INST}"
        assistant_message = f"{answer}"
        
        prompt = f"{start_token}Question:{user_message}\nAnswer:{assistant_message}{end_token}"
        chat_data.append({"text": prompt.strip()})
    
    return chat_data

# Prepare dataset
chat_data = apply_chat_template(qa_data)
qa_dataset = Dataset.from_list(chat_data)

# Initialize tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.eos_token)

# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '<pad>'})

# tokenizer.padding_side = "right"
# tokenizer.pad_token = "<pad>"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
       quantization_config=bnb_config,
)

model.config.use_cache = False
#model.resize_token_embeddings(len(tokenizer))

# More info: https://github.com/huggingface/transformers/pull/24906
model.config.pretraining_tp = 1 

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)


# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    weight_decay=0.01,
    lr_scheduler_type='linear',
    warmup_steps=50,
    report_to="wandb",
    run_name="FineTuneQA",
    logging_steps=25,
)

# Create SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=qa_dataset,
    dataset_text_field="text",
    max_seq_length=256,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=peft_config,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.model.save_pretrained("./pii_qa_llama2_chat_model")