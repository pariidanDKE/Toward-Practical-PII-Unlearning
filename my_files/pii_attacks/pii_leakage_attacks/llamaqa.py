import json
from typing import List, Dict
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig

import torch


class LlamaQA:
    def __init__(self, model_path: str, print_ans: bool, tokenizer_path: str = None):
        """
        Initialize the LlamaQA model by loading from the specified path

        Args:
            model_path: Path to the model weights
            print_ans: Whether to print answers
            tokenizer_path: Optional path to the tokenizer. If None, uses model_path.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.print_ans = print_ans

        # Load tokenizer
        tokenizer_source = tokenizer_path if tokenizer_path is not None else model_path
        print(f"Loading tokenizer from: {tokenizer_source}")
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_source)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print(f"Loading model from: {model_path}")


        self.model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,  ).to(self.device)
        #bnb_config = BitsAndBytesConfig(load_in_8bit=True)  # or load_in_4bit=True
        #self.model = LlamaForCausalLM.from_pretrained(model_path,quantization_config=bnb_config)
        print("Model loaded successfully")
        
        # Constants for Llama2 chat format
        self.B_INST = "[INST]"
        self.E_INST = "[/INST]"
    
    def generate_answer(self, question: str, max_length: int = 100) -> str:
        """
        Generate an answer for the given question
        """
        # Format input according to Llama2 chat format
        formatted_input = f"{self.B_INST} {question} {self.E_INST}"
        
        # Tokenize the input
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.device)
        
        # Generate the answer
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                max_length=inputs["input_ids"].shape[1] + max_length,
               # do_sample=True,
                temperature=0.9,
                top_p=0.6,
            )
        
        # Extract only the generated part (not the input)
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # if self.print_ans:
        #     print(f'{answer}')
        return answer.strip()
    
    def answer_questions(self, questions: List[str]) -> List[str]:
        """
        Generate answers for a list of questions
        """
        answers = []
        for i, question in enumerate(questions):
            #print(f"Processing question {i+1}/{len(questions)}: {question}")
            answer = self.generate_answer(question)
            answers.append(answer)
        
        return answers