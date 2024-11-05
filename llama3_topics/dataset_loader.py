from datasets import load_dataset
import random
# from datasets import load_dataset_builder
import torch
# import tokenizer
# import sys


class DatasetLoader:
    def __init__(self, dataset_name: str, split: str = 'train'):
        # Load the dataset from Hugging Face
        if dataset_name == "proj-persona/PersonaHub":
            self.persona_dataset = load_dataset(dataset_name, 'instruction')['train']['input persona']
        
        if dataset_name == "Anthropic/election_questions":
            self.elections_q_dataset = load_dataset(dataset_name)['test']['question']

        if dataset_name == "Anthropic/llm_global_opinions":
            self.global_q_dataset = load_dataset(dataset_name)['train']['question']



    def get_random_persona(self) -> str:
        # Choose a random sample from the dataset
        random_index = random.randint(0, len(self.persona_dataset) - 1)
        random_sample = self.persona_dataset[random_index]

        #### TODO: save the WHOLE LIST of PERSONA to disk if we don't have it within our folder
        # subsetPersonas.save_to_disk(os.path.join("out", "persona"))
        return random_sample
    
    def extract_professions(self,tokenizer,model,pipe,device) -> str:
        torch.cuda.empty_cache()

        input_persona_list = self.persona_dataset

        max_input_length = 1024
        max_new_tokens = 512  

        professionStrListComplete=[]

        for input_string in input_persona_list:
            input_ids = tokenizer(input_string, return_tensors='pt', truncation=True, max_length=max_input_length)['input_ids']
            inputs_ids = input_ids.to(device)
            truncated_text = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)

            prompt = "Return the professions from this sentence: " + input_string + "Return the names of the professions, separated by commas."

            if torch.cuda.is_available():
                print("GPU is available. Running on GPU.")
            else:
                print("GPU is not available. Running on CPU.")

            messages = [
            {"role": "system", "content": "You are an assistant that extracts professions from given sentences."},
            {"role": "user", "content": f"Return the professions from this sentence: {input_string}. Only list the profession names, separated by commas."}
            ]

            outputs = pipe(
            messages, 
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.5,  # length consistency
            top_p=1.0,
            )

            raw_output = outputs[0]["generated_text"] if isinstance(outputs[0], dict) else outputs[0]
            professionStr = raw_output[-1]["content"]
            # print(professionStr)
            professionStrList = [s.strip() for s in professionStr.split(",")]
            # print(professionStrList)
            professionStrListComplete.append(professionStrList)

            torch.cuda.empty_cache()

        return professionStrListComplete








    """
    ElectionQA (Anthropic)
    https://huggingface.co/datasets/Anthropic/election_questions/viewer
    """
    def get_random_election_question(self) -> str:
        # Choose a random sample from the dataset
        random_index = random.randint(0, len(self.elections_q_dataset) - 1)
        random_election_question = self.elections_q_dataset[random_index]

        #### TODO: save the WHOLE LIST of ELECTION QUESTIONS to DISK if we don't have it within our folder
        # subsetPersonas.save_to_disk(os.path.join("out", "persona"))

        return random_election_question

    def get_random_global_question(self) -> str:
        # Choose a random sample from the dataset
        random_index = random.randint(0, len(self.global_q_dataset) - 1)
        random_global_question = self.global_q_dataset[random_index]

        #### TODO: save the WHOLE LIST of GLOBAL QUESTIONS to DISK if we don't have it within our folder
        # subsetPersonas.save_to_disk(os.path.join("out", "persona"))

        return random_global_question

# def download_global_questions():
#     #### GlobalOpinions QA (Anthropic)
#     global_questions_dataset_name="Anthropic/llm_global_opinions"
#     global_questions_dataset = load_dataset(global_questions_dataset_name)
#     listOfGlobalQuestions = global_questions_dataset['train']['question']
#     subsetGlobalQs = listOfGlobalQuestions[0:10]
#     print(subsetGlobalQs)
#     print("="*100)


# """
# python3 -m preprocessing.make_data
# 	--out_path data/
# 	--tacred_data_path /path/to/tacred/
# """

# def main():
#     print("main")
    
#     ### load datasets
#     download_persona()
    



#     # settings = {
#     #     "emotion": ["emotion", ["0", "1", "3", "4"]],
#     #     "trec": ["trec10", ["0", "1", "3", "4", "5"]],
#     #     "ag_news": ["agnews", ["0", "1", "2", "3"]],
#     #     "tacred": ["tacred", ["0"]],
#     # }
#     # seed = 1  # Hardcode for determinism
#     # for dataset, ds in settings.items():
#     #     dataset_out_name, splits = ds
#     #     for split in splits:
#     #         print(f"Constructing {dataset} split {split}, with random seed {seed}")
#     #         if dataset == "tacred":
#     #             build_split(
#     #                 dataset, dataset_out_name, out_path, split, seed, tacred_path=tacred_data_path
#     #             )
#     #         else:
#     #             build_split(dataset, dataset_out_name, out_path, split, seed)
#     # download_wikitext(out_path)


# if __name__ == "__main__":
#     main()








