import pandas as pd
import json
import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict

class QADataValidator:
    """
    Comprehensive validation system for generated Q&A pairs

    TO DO : ADD CHECKS FOR EACH OF THE GENERATED FORMATS (CURRENTLY HAVE CHECKS FOR DIRECT,REVERSE,ONE-HOP AND PARAPHRASE -> ADD FOR USER_PROFILE,DEFAULT QA), EXTRACTION_SAMPLES, INVERTED_QA
    """
    
    def __init__(self, json_data_path: str, generated_data_folder: str):
        self.json_data_path = json_data_path
        self.generated_data_folder = generated_data_folder
        self.original_data = self._load_original_data()
        
    def _load_original_data(self) -> List[Dict]:
        """Load the original JSON data containing user profiles"""
        with open(self.json_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _parse_generated_json(self, json_str: str) -> Dict:
        """Safely parse JSON from generated text"""
        try:
            # Clean up common JSON formatting issues
            json_str = json_str.strip()
            if json_str.startswith('```json'):
                json_str = json_str[7:]
            if json_str.endswith('```'):
                json_str = json_str[:-3]
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {}
    
    def _extract_user_profile_from_prompt(self, prompt: str) -> Dict:
        """Extract user profile information from the formatted prompt"""
        try:
            # Look for the input section in the prompt
            input_match = re.search(r'Input:\s*({.*?})\s*', prompt, re.DOTALL)
            if input_match:
                input_str = input_match.group(1)
                return json.loads(input_str)
        except:
            pass
        return {}
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (handle spacing, case, etc.)"""
        if not isinstance(text, str):
            return str(text)
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    def _check_exact_match(self, text: str, target: str) -> bool:
        """Check if target appears exactly in text"""
        return target.lower() in text.lower()
    
    def _extract_pii_values(self, user_profile: Dict) -> Dict[str, str]:
        """Extract all PII values from user profile for easy lookup"""
        pii_values = {}
        
        # Direct fields
        direct_fields = ['full_name', 'partner_name', 'email_address', 'twitter_username', 
                        'home_address', 'work_address', 'phone_number', 'DOB', 
                        'credit_card_nr', 'bank_account_number', 'bank_name', 
                        'latest_bank_transaction', 'financial_consultant_name',
                        'health_insurance_nr', 'hospital_name', 'doctor_name',
                        'country', 'disease', 'treatment', 'Occupation']
        
        for field in direct_fields:
            if field in user_profile:
                pii_values[field] = user_profile[field]
        
        # Handle pii_picked_dict if present
        if 'pii_picked_dict' in user_profile:
            for pii_item in user_profile['pii_picked_dict']:
                pii_values[pii_item['type']] = pii_item['value']
                
        return pii_values

    def validate_direct_qa(self, csv_path: str) -> Dict[str, Any]:
        """
        Validate DirectQA: Check that full_name is in question and specific PII is in answer
        """
        df = pd.read_csv(csv_path)
        results = {
            'total_samples': len(df),
            'valid_samples': 0,
            'validation_errors': [],
            'detailed_results': []
        }
        
        for idx, row in df.iterrows():
            try:
                # Parse generated data
                generated_data = self._parse_generated_json(row['generated_text'])
                user_profile = self._extract_user_profile_from_prompt(row['prompt'])
                
                if not generated_data or 'direct_pii_qa_pairs' not in generated_data:
                    results['validation_errors'].append(f"Row {idx}: Invalid JSON structure")
                    continue
                
                full_name = user_profile.get('full_name', '')
                pii_values = self._extract_pii_values(user_profile)
                
                sample_valid = True
                sample_errors = []
                
                for qa_pair in generated_data['direct_pii_qa_pairs']:
                    question = qa_pair.get('question', '')
                    answer = qa_pair.get('answer', '')
                    pii_type = qa_pair.get('pii_type', '')
                    
                    # Check 1: Full name in question
                    if not self._check_exact_match(question, full_name):
                        sample_errors.append(f"Full name '{full_name}' not found in question: {question}")
                        sample_valid = False
                    
                    # Check 2: Specific PII value in answer
                    expected_pii_value = pii_values.get(pii_type, '')
                    if expected_pii_value and not self._check_exact_match(answer, expected_pii_value):
                        sample_errors.append(f"PII value '{expected_pii_value}' not found in answer: {answer}")
                        sample_valid = False
                
                if sample_valid:
                    results['valid_samples'] += 1
                else:
                    results['validation_errors'].extend([f"Row {idx}: {error}" for error in sample_errors])
                
                results['detailed_results'].append({
                    'row': idx,
                    'valid': sample_valid,
                    'errors': sample_errors
                })
                
            except Exception as e:
                results['validation_errors'].append(f"Row {idx}: Processing error - {str(e)}")
        
        return results
    
    def validate_reverse_qa(self, csv_path: str) -> Dict[str, Any]:
        """
        Validate ReverseQA: Check that PII is in question and full_name is in answer
        """
        df = pd.read_csv(csv_path)
        results = {
            'total_samples': len(df),
            'valid_samples': 0,
            'validation_errors': [],
            'detailed_results': []
        }
        
        for idx, row in df.iterrows():
            try:
                # Parse generated data
                generated_data = self._parse_generated_json(row['generated_text'])
                user_profile = self._extract_user_profile_from_prompt(row['prompt'])
                
                if not generated_data or 'reverse_pii_qa_pairs' not in generated_data:
                    results['validation_errors'].append(f"Row {idx}: Invalid JSON structure")
                    continue
                
                full_name = user_profile.get('full_name', '')
                pii_values = self._extract_pii_values(user_profile)
                
                sample_valid = True
                sample_errors = []
                
                for qa_pair in generated_data['reverse_pii_qa_pairs']:
                    question = qa_pair.get('question', '')
                    answer = qa_pair.get('answer', '')
                    pii_type = qa_pair.get('pii_type', '')
                    
                    # Check 1: Specific PII value in question
                    expected_pii_value = pii_values.get(pii_type, '')
                    if expected_pii_value and not self._check_exact_match(question, expected_pii_value):
                        sample_errors.append(f"PII value '{expected_pii_value}' not found in question: {question}")
                        sample_valid = False
                    
                    # Check 2: Full name in answer
                    if not self._check_exact_match(answer, full_name):
                        sample_errors.append(f"Full name '{full_name}' not found in answer: {answer}")
                        sample_valid = False
                
                if sample_valid:
                    results['valid_samples'] += 1
                else:
                    results['validation_errors'].extend([f"Row {idx}: {error}" for error in sample_errors])
                
                results['detailed_results'].append({
                    'row': idx,
                    'valid': sample_valid,
                    'errors': sample_errors
                })
                
            except Exception as e:
                results['validation_errors'].append(f"Row {idx}: Processing error - {str(e)}")
        
        return results
    
    def validate_one_hop_qa(self, csv_path: str) -> Dict[str, Any]:
        """
        Validate OneHopQA: Check that question_pii, answer_pii, and full_name are present
        """
        df = pd.read_csv(csv_path)
        results = {
            'total_samples': len(df),
            'valid_samples': 0,
            'validation_errors': [],
            'detailed_results': []
        }
        
        for idx, row in df.iterrows():
            try:
                # Parse generated data
                generated_data = self._parse_generated_json(row['generated_text'])
                user_profile = self._extract_user_profile_from_prompt(row['prompt'])
                
                if not generated_data or 'one_hop_qa_pairs' not in generated_data:
                    results['validation_errors'].append(f"Row {idx}: Invalid JSON structure")
                    continue
                
                full_name = user_profile.get('full_name', '')
                pii_values = self._extract_pii_values(user_profile)
                
                sample_valid = True
                sample_errors = []
                
                for qa_pair in generated_data['one_hop_qa_pairs']:
                    question = qa_pair.get('question', '')
                    answer = qa_pair.get('answer', '')
                    question_pii = qa_pair.get('question_pii', {})
                    answer_pii = qa_pair.get('answer_pii', {})
                    
                    # Check 1: Question PII value in question
                    if 'value' in question_pii:
                        question_pii_value = question_pii['value']
                        if not self._check_exact_match(question, question_pii_value):
                            sample_errors.append(f"Question PII '{question_pii_value}' not found in question: {question}")
                            sample_valid = False
                    
                    # Check 2: Answer PII value in answer
                    if 'value' in answer_pii:
                        answer_pii_value = answer_pii['value']
                        if not self._check_exact_match(answer, answer_pii_value):
                            sample_errors.append(f"Answer PII '{answer_pii_value}' not found in answer: {answer}")
                            sample_valid = False
                    
                    # Check 3: Full name in answer
                    if not self._check_exact_match(answer, full_name):
                        sample_errors.append(f"Full name '{full_name}' not found in answer: {answer}")
                        sample_valid = False
                
                if sample_valid:
                    results['valid_samples'] += 1
                else:
                    results['validation_errors'].extend([f"Row {idx}: {error}" for error in sample_errors])
                
                results['detailed_results'].append({
                    'row': idx,
                    'valid': sample_valid,
                    'errors': sample_errors
                })
                
            except Exception as e:
                results['validation_errors'].append(f"Row {idx}: Processing error - {str(e)}")
        
        return results
    
    def validate_paraphrased_qa(self, csv_path: str) -> Dict[str, Any]:
        """
        Validate ParaphrasedQA: Check that full_name and required PII are present in answers
        """
        df = pd.read_csv(csv_path)
        results = {
            'total_samples': len(df),
            'valid_samples': 0,
            'validation_errors': [],
            'detailed_results': []
        }
        
        for idx, row in df.iterrows():
            try:
                # Parse generated data
                generated_data = self._parse_generated_json(row['generated_text'])
                user_profile = self._extract_user_profile_from_prompt(row['prompt'])
                
                if not generated_data:
                    results['validation_errors'].append(f"Row {idx}: Invalid JSON structure")
                    continue
                
                full_name = user_profile.get('full_name', '')
                pii_values = self._extract_pii_values(user_profile)
                
                # Get the pii_picked to know what should be preserved
                pii_picked = user_profile.get('pii_picked', [])
                
                sample_valid = True
                sample_errors = []
                
                # Check paraphrased pairs
                paraphrased_pairs = generated_data.get('more_paraphrased_qa_pairs', 
                                                    generated_data.get('paraphrased_qa_pairs', []))
                
                for qa_pair in paraphrased_pairs:
                    question = qa_pair.get('paraphrased_question', qa_pair.get('question', ''))
                    answer = qa_pair.get('paraphrased_answer', qa_pair.get('answer', ''))
                    
                    # Check 1: Full name in question or answer (flexible based on QA type)
                    if not (self._check_exact_match(question, full_name) or self._check_exact_match(answer, full_name)):
                        sample_errors.append(f"Full name '{full_name}' not found in Q&A pair")
                        sample_valid = False
                    
                    # Check 2: Required PII values are preserved
                    for pii_type in pii_picked:
                        if pii_type in pii_values:
                            pii_value = pii_values[pii_type]
                            if not (self._check_exact_match(question, pii_value) or self._check_exact_match(answer, pii_value)):
                                sample_errors.append(f"Required PII '{pii_value}' ({pii_type}) not found in Q&A pair")
                                sample_valid = False
                
                if sample_valid:
                    results['valid_samples'] += 1
                else:
                    results['validation_errors'].extend([f"Row {idx}: {error}" for error in sample_errors])
                
                results['detailed_results'].append({
                    'row': idx,
                    'valid': sample_valid,
                    'errors': sample_errors
                })
                
            except Exception as e:
                results['validation_errors'].append(f"Row {idx}: Processing error - {str(e)}")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, Any], qa_type: str) -> str:
        """Generate a formatted validation report"""
        total = results['total_samples']
        valid = results['valid_samples']
        invalid = total - valid
        
        report = f"""
=== {qa_type} Validation Report ===
Total Samples: {total}
Valid Samples: {valid} ({valid/total*100:.1f}%)
Invalid Samples: {invalid} ({invalid/total*100:.1f}%)

"""
        
        if results['validation_errors']:
            report += f"Validation Errors ({len(results['validation_errors'])}):\n"
            for error in results['validation_errors'][:10]:  # Show first 10 errors
                report += f"  - {error}\n"
            
            if len(results['validation_errors']) > 10:
                report += f"  ... and {len(results['validation_errors']) - 10} more errors\n"
        
        return report
    
    def run_all_validations(self):
        """Run all validations and generate comprehensive report"""
        validations = [
            ('DirectQA', f'{self.generated_data_folder}DirectQA.csv', self.validate_direct_qa),
            ('ReverseQA', f'{self.generated_data_folder}ReverseDirectQA.csv', self.validate_reverse_qa),
            ('OneHopQA', f'{self.generated_data_folder}OneHopQA.csv', self.validate_one_hop_qa),
            #('ParaphrasedQA', f'{self.generated_data_folder}ParaQA.csv', self.validate_paraphrased_qa)
        ]
        
        all_results = {}
        full_report = "=== COMPREHENSIVE QA VALIDATION REPORT ===\n\n"
        
        for qa_type, csv_path, validation_func in validations:
            try:
                print(f"Validating {qa_type}...")
                results = validation_func(csv_path)
                all_results[qa_type] = results
                full_report += self.generate_validation_report(results, qa_type)
                full_report += "\n" + "="*50 + "\n"
            except FileNotFoundError:
                print(f"File not found: {csv_path}")
                full_report += f"{qa_type}: FILE NOT FOUND - {csv_path}\n\n"
            except Exception as e:
                print(f"Error validating {qa_type}: {str(e)}")
                full_report += f"{qa_type}: VALIDATION ERROR - {str(e)}\n\n"
        
        return all_results, full_report


# Usage example
if __name__ == "__main__":
    # Initialize validator
    json_data_path = '/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/qa_pairs_full2.json'
    generated_data_folder = '/projects/0/hpmlprjs/LLM/danp/UGBench/my_files/pii_dataset/data/generated_data/'
    
    validator = QADataValidator(json_data_path, generated_data_folder)
    
    # Run all validations
    all_results, report = validator.run_all_validations()
    
    # Print report
    print(report)
    
    # Save detailed results
    with open(f'{generated_data_folder}validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save report
    with open(f'{generated_data_folder}validation_report.txt', 'w') as f:
        f.write(report)