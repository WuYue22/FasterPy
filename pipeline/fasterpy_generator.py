# This file take the input generated from prompt_generator.py and generate the final patch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List

class FasterPyGenerator:
    def __init__(self,model_path:str = "../model/deepseek-coder-6.7b-Instruct-FT"):
        self.device = torch. device("cuda" if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)

    def __call__(self, prompt:List[str])->List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024)
        # return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)