from agent import Agent
from dataset_loader import DatasetLoader

from tavily import TavilyClient

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from collections import Counter

import json

import torch

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
    
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
import huggingface_hub

import torch


def readTopProf():
    with open("lists_of_profs.json", "r") as file:
        all_profs = json.load(file)

    all_profs = [item.lower() for item in all_profs]
    prof_counts = Counter(all_profs)
    top_10_profs = prof_counts.most_common(10)
    return top_10_profs

    

if __name__ == "__main__":
    # Dataset names
    persona_dataset_name="proj-persona/PersonaHub"
    election_questions_dataset_name="Anthropic/election_questions"
    global_questions_dataset_name="Anthropic/llm_global_opinions"

    # Initialize DatasetLoader instance
    dataset_loader_persona = DatasetLoader(persona_dataset_name)
    dataset_loader_election_qs = DatasetLoader(election_questions_dataset_name)
    dataset_loader_global_qs = DatasetLoader(global_questions_dataset_name)


    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    access_token = "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" 

    huggingface_hub.login(token=access_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
    tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")
    model = AutoModelForCausalLM.from_pretrained(model_id,use_auth_token=access_token)
    model.to(device) 
            # Define pipeline for text generation
    pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16,
                device=device
    )

    # grab all professions in the personahub dataset
    # agent_prof = dataset_loader_persona.extract_professions(tokenizer,model,pipe,device)
    # print(agent_prof)
    # all_profs = [string for sublist in agent_prof for string in sublist]
    # with open("lists_of_profs.json", "w") as file:
    #     json.dump(all_profs, file)

  

    # top10ProfsList = readTopProf()
    # top10Profs = [item[0] for item in top10ProfsList]

    print("Generating the Agent Descriptions")
    top10Profs = ['historian', 'data analyst','genealogist','sports journalist','researcher','journalist','data scientist', 'sports analyst', 'writer','travel blogger']
    ages = ["young","old"]
    genders = ["female","male"]

    listOfPrompts = []
    for g in genders:
        for a in ages:
            for p in top10Profs:
                prompt = f"You are a {a} {g} who is {p}"
                print(prompt)
                listOfPrompts.append(prompt)

    print(len(listOfPrompts))


    #create agent 
    agent=
    # agent.generatePromptGlobalHealth()



