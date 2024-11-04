from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline

"""
Prompts for creating the agent
"""




class Agent:
    def __init__(self, model_name: str, agent_description: str):
        # Initialize the agent with an LLM model instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.model.to(self.device)
        self.description = agent_description
        
    def generate_response(self, question: str, responses:str) -> str:
        # Generate a response from the LLM using the given prompt
        max_input_length = 1024
        max_new_tokens = 512 


        input_ids = self.tokenizer(question, return_tensors='pt', truncation=True, max_length=max_input_length)['input_ids']
        inputs_ids = input_ids.to(self.device)
        truncated_text = self.tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

        prompt =  question + responses

        if torch.cuda.is_available():
            print("GPU is available. Running on GPU.")
        else:
            print("GPU is not available. Running on CPU.")

        messages = [
            {"role": "system", "content": self.description},
            {"role": "user", "content": prompt}
        ]

        pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device=self.device
        )

        outputs = pipe(
            messages, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.5,  # length consistency
            top_p=1.0,
        )

        raw_output = outputs[0]["generated_text"] if isinstance(outputs[0], dict) else outputs[0]
        opinion = raw_output[-1]["content"]
        torch.cuda.empty_cache()
        return opinion


    
