from agent import Agent
from dataset_loader import DatasetLoader

from tavily import TavilyClient

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from collections import Counter

import json

import re 
import torch
import os

import csv

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext,PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
    
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
import huggingface_hub



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

    #TODO: load Llama3.2 model separately
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

    """
    *** Grab all professions in the personahub dataset ***
    agent_prof = dataset_loader_persona.extract_professions(tokenizer,model,pipe,device)
    print(agent_prof)
    all_profs = [string for sublist in agent_prof for string in sublist]
    with open("lists_of_profs.json", "w") as file:
        json.dump(all_profs, file)
    """

    #load top 10 professions
    top10ProfsList = readTopProf()
    top10Profs = [item[0] for item in top10ProfsList]

    print("Generating the Agent Descriptions")
    # top10Profs = ['historian', 'data analyst','genealogist','sports journalist','researcher','journalist','data scientist', 'sports analyst', 'writer','travel blogger']
    ages = ["young","elder"]
    genders = ["female","male"]

    listOfDescriptions = []
    for g in genders:
        for a in ages:
            for p in top10Profs:
                prompt = f"You are a {a} {g} who is a/an {p}"
                # print(prompt)
                listOfDescriptions.append(prompt)

    # print(len(listOfPrompts))
    print("Agent Description Generation Completed ")

    print("Question Generation - compiled_questions_global_health.csv")
    
    """
    Questions and Responses:

    see questions.txt and responses.txt
    """
    with open('/nas/eclairnas01/users/gpinto/csci_project/llm-opinions/preprocessing/questions.txt') as qs:
        questionsList = qs.readlines()
    # questionsList = 
    with open('/nas/eclairnas01/users/gpinto/csci_project/llm-opinions/preprocessing/responses.txt') as rs:
        responseOptionsList = rs.readlines()


    # for description in listOfDescriptions:
    #     #create agent 
    #     print(description+"\n")

    print("Synthetic Agent Generation and QA")
    for description in listOfDescriptions:
        # Create agent and prompt it a question
        print("Description:",description)

        agent_file_name=""
        pattern = r"are a (\w+) (\w+) who is a/an (\w+)"
        match = re.search(pattern,description)
        if match:
            age_descripton = match.group(1)
            gender = match.group(2)
            profession = match.group(3)
            print(f"Age Descripton: {age_descripton}")
            print(f"Gender: {gender}")
            print(f"Profession: {profession}")
            agent_file_name = f"{age_descripton}_{gender}_{profession}.csv"
        
        else:
            print("No match found.")

        agent_person = Agent(model_id, description)
        file_exists = os.path.isfile(agent_file_name)

        with open("/nas/eclairnas01/users/gpinto/csci_project/llm-opinions/preprocessing/agent_answers/"+agent_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if not file_exists:
                writer.writerow(["Question", "Response Options", "Agent Answer"])


            for idx in range(len(questionsList)):
                cur_question = questionsList[idx]
                cur_response = responseOptionsList[idx]
                answer = agent_person.generate_response(cur_question,cur_response)
                print("AGENT RESPONSE: ",answer)

                writer.writerow([cur_question, cur_response, answer])
        
        # agent.generatePromptGlobalHealth()
        print("Generation and QA complete")
        print("Start of the Global Health Debate")




