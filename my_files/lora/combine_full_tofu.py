import json

def extract_qna(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return [
            {'question': item['question'], 'answer': item['answer']}
            for item in data
            if 'question' in item and 'answer' in item
        ]

# Extract from both files
retrain_data = extract_qna('/projects/0/hpmlprjs/LLM/danp/UGBench/data/TOFU/retain99.json')
forget_data = extract_qna('/projects/0/hpmlprjs/LLM/danp/UGBench/data/TOFU/forget01.json')

# Create sets of questions for quick lookup
retrain_questions = {item['question'] for item in retrain_data}
forget_questions = {item['question'] for item in forget_data}

# Find duplicates
duplicates = retrain_questions & forget_questions

# Print duplicates
if duplicates:
    print("Duplicate questions found (skipped):")
    for q in duplicates:
        print(f"- {q}")

# Filter out duplicates from forget_data
unique_forget_data = [item for item in forget_data if item['question'] not in duplicates]

# Combine data
combined_data = retrain_data + unique_forget_data

# Save to a new file
with open('/projects/0/hpmlprjs/LLM/danp/UGBench/data/TOFU/full.json', 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, indent=2, ensure_ascii=False)

print(f"Combined {len(combined_data)} unique Q&A pairs.")
