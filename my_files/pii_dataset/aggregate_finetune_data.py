import pandas as pd
import json
import re
from typing import List, Dict, Any

class QAExtractor:
    """
    Extracts QA pairs from generated CSV files and combines them into simple lists
    """
    
    def __init__(self, generated_data_folder: str, output_path: str):
        self.generated_data_folder = generated_data_folder
        self.output_path = output_path
        
    def _parse_generated_json(self, json_str: str) -> Dict:
        """Safely parse JSON from generated text"""
        try:
            json_str = json_str.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {}
    
    def extract_direct_qa(self) -> List[Dict]:
        """Extract DirectQA pairs"""
        csv_path = f'{self.generated_data_folder}DirectQA.csv'
        df = pd.read_csv(csv_path)
        
        qa_pairs = []
        
        for idx, row in df.iterrows():
            try:
                generated_data = self._parse_generated_json(row['generated_text'])
                
                if 'direct_pii_qa_pairs' not in generated_data:
                    continue
                
                for qa_pair in generated_data['direct_pii_qa_pairs']:
                    qa_pairs.append({
                        'question': qa_pair.get('question', ''),
                        'answer': qa_pair.get('answer', ''),
                        'type': 'direct_pii'
                    })
                    
            except Exception as e:
                print(f"DirectQA Row {idx}: Processing error - {str(e)}")
        
        return qa_pairs
    
    def extract_reverse_qa(self) -> List[Dict]:
        """Extract ReverseQA pairs"""
        csv_path = f'{self.generated_data_folder}ReverseDirectQA.csv'
        df = pd.read_csv(csv_path)
        
        qa_pairs = []
        
        for idx, row in df.iterrows():
            try:
                generated_data = self._parse_generated_json(row['generated_text'])
                
                if 'reverse_pii_qa_pairs' not in generated_data:
                    continue
                
                for qa_pair in generated_data['reverse_pii_qa_pairs']:
                    qa_pairs.append({
                        'question': qa_pair.get('question', ''),
                        'answer': qa_pair.get('answer', ''),
                        'type': 'reverse_pii'
                    })
                    
            except Exception as e:
                print(f"ReverseQA Row {idx}: Processing error - {str(e)}")
        
        return qa_pairs
    
    def extract_one_hop_qa(self) -> tuple[List[Dict], List[Dict]]:
        """Extract OneHopQA pairs - returns (training_pairs, validation_pairs)"""
        csv_path = f'{self.generated_data_folder}OneHopQA.csv'
        df = pd.read_csv(csv_path)
        
        training_pairs = []
        validation_pairs = []
        
        for idx, row in df.iterrows():
            try:
                generated_data = self._parse_generated_json(row['generated_text'])
                
                if 'one_hop_qa_pairs' not in generated_data:
                    continue
                
                one_hop_pairs = generated_data['one_hop_qa_pairs']
                
                # Add first 2 pairs to training
                for qa_pair in one_hop_pairs[:2]:
                    training_pairs.append({
                        'question': qa_pair.get('question', ''),
                        'answer': qa_pair.get('answer', ''),
                        'type': 'one_hop'
                    })
                
                # Add last pair to validation if it exists
                if len(one_hop_pairs) > 2:
                    last_qa = one_hop_pairs[-1]
                    validation_pairs.append({
                        'question': last_qa.get('question', ''),
                        'answer': last_qa.get('answer', ''),
                        'type': 'one_hop_validation'
                    })
                    
            except Exception as e:
                print(f"OneHopQA Row {idx}: Processing error - {str(e)}")
        
        return training_pairs, validation_pairs
    
    def extract_paraphrased_qa(self) -> List[Dict]:
        """Extract ParaphrasedQA pairs"""
        csv_path = f'{self.generated_data_folder}MoreParaphrase2.csv'
        df = pd.read_csv(csv_path)
        
        qa_pairs = []
        
        for idx, row in df.iterrows():
            try:
                generated_data = self._parse_generated_json(row['generated_text'])
                
                # Handle different possible keys for paraphrased pairs
                paraphrased_pairs = generated_data.get('more_paraphrased_qa_pairs', 
                                                    generated_data.get('paraphrased_qa_pairs', []))
                
                for qa_pair in paraphrased_pairs:
                    question = qa_pair.get('paraphrased_question', qa_pair.get('question', ''))
                    answer = qa_pair.get('paraphrased_answer', qa_pair.get('answer', ''))
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'type': 'paraphrased'
                    })
                    
            except Exception as e:
                print(f"ParaQA Row {idx}: Processing error - {str(e)}")
        
        return qa_pairs
    
    def extract_all_qa_pairs(self):
        """Extract all QA pairs and save to files"""
        print("Extracting DirectQA...")
        direct_qa = self.extract_direct_qa()
        
        print("Extracting ReverseQA...")
        reverse_qa = self.extract_reverse_qa()
        
        print("Extracting OneHopQA...")
        one_hop_training, one_hop_validation = self.extract_one_hop_qa()
        
        print("Extracting ParaphrasedQA...")
        paraphrased_qa = self.extract_paraphrased_qa()
        
        # Combine all training QA pairs
        all_training_qa = direct_qa + reverse_qa + one_hop_training + paraphrased_qa
        
        # Load base data and add QA pairs
        with open("/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full.json", 'r') as f:
            base_data = json.load(f)
        
        # Add QA pairs to base data
        base_data.extend(all_training_qa)
        
        # Save main file with all QA pairs
        main_output = f"{self.output_path}_with_qa.json"
        with open(main_output, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, indent=2, ensure_ascii=False)
        
        # Save validation file (just the validation QA pairs)
        validation_output = f"{self.output_path}_validation.json"
        with open(validation_output, 'w', encoding='utf-8') as f:
            json.dump(one_hop_validation, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"DirectQA pairs: {len(direct_qa)}")
        print(f"ReverseQA pairs: {len(reverse_qa)}")
        print(f"OneHopQA training pairs: {len(one_hop_training)}")
        print(f"OneHopQA validation pairs: {len(one_hop_validation)}")
#        print(f"ParaphrasedQA pairs: {len(paraphrased_qa)}")
        print(f"Total training QA pairs: {len(all_training_qa)}")
        print(f"Original base data entries: {len(base_data) - len(all_training_qa)}")
        print(f"Final combined entries: {len(base_data)}")
        print(f"\nFiles saved:")
        print(f"Main: {main_output}")
        print(f"Validation: {validation_output}")
        
        return main_output, validation_output


# Usage
if __name__ == "__main__":
    generated_data_folder = '/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/'
    output_path = "/projects/0/hpmlprjs/LLM/danp/UGBench/data/PII/full"
    
    extractor = QAExtractor(generated_data_folder, output_path)
    main_file, validation_file = extractor.extract_all_qa_pairs()