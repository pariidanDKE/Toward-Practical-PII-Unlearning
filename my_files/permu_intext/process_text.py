import re
import json
from typing import List, Dict, Optional

def process_tokenization_file(file_path: str, output_path: Optional[str] = None) -> List[Dict]:
    """
    Process a text file containing tokenization data and extract structured information.
    
    Args:
        file_path (str): Path to the input text file
        output_path (str, optional): Path to save the processed data as JSON
    
    Returns:
        List[Dict]: List of dictionaries containing the structured data
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content by the separator line
    entries = content.split('----------------------------------------')
    
    processed_data = []
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
            
        # Parse each entry
        parsed_entry = parse_entry(entry)
        if parsed_entry:
            processed_data.append(parsed_entry)
    
    # Save to JSON if output path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"Processed data saved to {output_path}")
    
    return processed_data

def parse_entry(entry: str) -> Optional[Dict]:
    """
    Parse a single entry and extract the required fields.
    
    Args:
        entry (str): Raw entry text
    
    Returns:
        Dict or None: Structured data or None if parsing fails
    """
    
    # Extract subject name and missing tokens
    subject_pattern = r"Subject '([^']+)' has missing tokens: (\[[\d\s,]+\])"
    subject_match = re.search(subject_pattern, entry)
    
    if not subject_match:
        return None
    
    subject_name = subject_match.group(1)
    missing_tokens_str = subject_match.group(2)
    
    # Parse missing tokens list
    try:
        missing_tokens = eval(missing_tokens_str)  # Safe here as we control the format
    except:
        missing_tokens = []
    
    # Extract full text string
    full_text_pattern = r"Full text string: (.*?)(?=\n Full text token IDs:|\n Subject token IDs:|$)"
    full_text_match = re.search(full_text_pattern, entry, re.DOTALL)
    full_text = full_text_match.group(1).strip() if full_text_match else ""
    
    # Extract full text token IDs
    full_token_ids_pattern = r"Full text token IDs: (\[[\d\s,]+\])"
    full_token_ids_match = re.search(full_token_ids_pattern, entry)
    try:
        full_text_token_ids = eval(full_token_ids_match.group(1)) if full_token_ids_match else []
    except:
        full_text_token_ids = []
    
    # Extract subject token IDs
    subject_token_ids_pattern = r"Subject token IDs: (\[[\d\s,]+\])"
    subject_token_ids_match = re.search(subject_token_ids_pattern, entry)
    try:
        subject_token_ids = eval(subject_token_ids_match.group(1)) if subject_token_ids_match else []
    except:
        subject_token_ids = []
    
    # Extract LCS proportion if present
    lcs_pattern = r"Proportion of LCS Indices: ([\d.]+)"
    lcs_match = re.search(lcs_pattern, entry)
    lcs_proportion = float(lcs_match.group(1)) if lcs_match else None
    
    # Extract question from the full text
    question = extract_question_from_full_text(full_text)
    
    # Extract answer from the full text
    answer = extract_answer_from_full_text(full_text)
    
    return {
        "subject_name": subject_name,
        "missing_tokens": missing_tokens,
        "missing_token_count": len(missing_tokens),
        "question": question,
        "answer": answer,
        "full_text_string": full_text,
        "full_text_token_ids": full_text_token_ids,
        "subject_token_ids": subject_token_ids,
        "lcs_proportion": lcs_proportion
    }

def extract_question_from_full_text(full_text: str) -> str:
    """Extract the question from the full text string."""
    # Look for text between user tags
    question_pattern = r"<\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>"
    match = re.search(question_pattern, full_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_answer_from_full_text(full_text: str) -> str:
    """Extract the answer from the full text string."""
    # Look for text between assistant tags
    answer_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>"
    match = re.search(answer_pattern, full_text, re.DOTALL)
    return match.group(1).strip() if match else ""

def filter_by_subject(data: List[Dict], subject_name: str) -> List[Dict]:
    """Filter data by subject name."""
    return [entry for entry in data if entry['subject_name'] == subject_name]

def get_unique_subjects(data: List[Dict]) -> List[str]:
    """Get list of unique subject names."""
    return list(set(entry['subject_name'] for entry in data))

def print_summary(data: List[Dict]):
    """Print a summary of the processed data."""
    print(f"Total entries processed: {len(data)}")
    print(f"Unique subjects: {len(get_unique_subjects(data))}")
    print("\nSubjects and their entry counts:")
    
    subject_counts = {}
    for entry in data:
        subject = entry['subject_name']
        subject_counts[subject] = subject_counts.get(subject, 0) + 1
    
    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count} entries")

# Example usage
if __name__ == "__main__":
    # Process the file
    data = process_tokenization_file('output_llaam3.txt', 'processed_tokenization_data.json')
    
    # Print summary
    print_summary(data)
    
    # # Example: Get all entries for a specific subject
    # viktor_entries = filter_by_subject(data, 'Viktor Fedorovich Melnikov')
    # print(f"\nViktor Fedorovich Melnikov has {len(viktor_entries)} entries")
    
    # # Example: Print first entry structure
    # if data:
    #     print("\nFirst entry structure:")
    #     first_entry = data[0]
    #     for key, value in first_entry.items():
    #         if key == 'full_text_token_ids' and len(str(value)) > 100:
    #             print(f"{key}: [list of {len(value)} tokens]")
    #         else:
    #             print(f"{key}: {value}")