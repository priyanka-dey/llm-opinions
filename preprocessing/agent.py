from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline


"""


modelName="meta-llama/Llama-3.2-3B" #not multimodal 
access_token = "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" 
huggingface_hub.login(token=access_token)
tokenizer = AutoTokenizer.from_pretrained(modelName,token="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")
model = AutoModelForCausalLM.from_pretrained(modelName,use_auth_token=access_token)


"""
class Agent:
    def __init__(self, model_name: str):
        # Initialize the agent with an LLM model instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, prompt: str) -> str:
        # Generate a response from the LLM using the given prompt
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs['input_ids'])
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
