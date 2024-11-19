from openai import OpenAI
import pandas as pd
import random

# Placeholder function for calling the OpenAI GPT API
def call_gpt_agent(persona, team, agent_id, prompt):
    # Customize this function with your OpenAI API key
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"You are {persona} on Team {team} (Agent {agent_id}). Your goal is to convince the other team that your stance is correct while providing strong arguments and rebuttals. Use all debating tactics, including logical reasoning, emotional appeals, and addressing counterarguments, to try to make the opposing team agree with your position. Remember what arguments have been said before and avoid repeating old arguments. Focus on rebutting the opponent's main points while introducing new critical points. Limit your response in 50 words."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o",
        temperature=0.7,  # Increase temperature for more varied responses
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0.5,  # Increase penalty to reduce repetition
        presence_penalty=0.5,
        response_format={
            "type": "text"
        }
    )
    return response.choices[0].message.content

# Agent class
class Agent:
    def __init__(self, team, agent_id, persona):
        self.team = team
        self.agent_id = agent_id
        self.persona = persona

    def debate(self, prompt):
        return call_gpt_agent(self.persona, self.team, self.agent_id, prompt)

    def answer_question(self, question):
        prompt = f"{question} (Answer Yes or No, and explain briefly.)"
        response = call_gpt_agent(self.persona, self.team, self.agent_id, prompt)
        return "Yes" if "yes" in response.lower() else "No"

# Debate function
def simulate_debate(topic, team_pro_agents, team_against_agents, n_rounds, questions_df):
    # Create a DataFrame to store agent responses to questions
    results = pd.DataFrame(columns=["Question", "Response", "AgentID", "TeamID", "Time"])

    # Administer questions before the debate
    for _, question_row in questions_df.iterrows():
        question = question_row["questions"]
        for agent in team_pro_agents + team_against_agents:
            response = agent.answer_question(question)
            results = results.append({
                "Question": question,
                "Response": response,
                "AgentID": agent.agent_id,
                "TeamID": agent.team,
                "Time": "before"
            }, ignore_index=True)

    round_num = 1
    transcript = f"Debate topic: '{topic}'.\nTeam Pro argues in favor, while Team Against argues against. Each team will use all debating tactics to convince the other.\n"

    while round_num <= n_rounds:
        print(f"\n================ Round {round_num} ================\n")
        
        # Pro team speaks
        for i, agent in enumerate(team_pro_agents):
            print(f"Team Pro - Agent {agent.agent_id} says:")
            if i == 0:
                response = agent.debate(transcript)
            else:
                response = agent.debate(transcript + f"\nSupport your teammate's arguments and provide additional evidence or stronger points.")
            print(response)
            transcript += f"Team Pro - Agent {agent.agent_id} says: {response}\n"

        # Against team speaks
        for i, agent in enumerate(team_against_agents):
            print(f"Team Against - Agent {agent.agent_id} says:")
            if i == 0:
                response = agent.debate(transcript)
            else:
                response = agent.debate(transcript + f"\nSupport your teammate's arguments and provide additional evidence or stronger points.")
            print(response)
            transcript += f"Team Against - Agent {agent.agent_id} says: {response}\n"

        round_num += 1

    # Administer questions after the debate
    for _, question_row in questions_df.iterrows():
        question = question_row["questions"]
        for agent in team_pro_agents + team_against_agents:
            response = agent.answer_question(question)
            results = results.append({
                "Question": question,
                "Response": response,
                "AgentID": agent.agent_id,
                "TeamID": agent.team,
                "Time": "after"
            }, ignore_index=True)

    # Save results to CSV
    results.to_csv("agent_responses.csv", index=False)

if __name__ == "__main__":
    # Configuration
    topic = "Should AI be regulated to ensure safety?"
    n1 = 2  # Number of agents in Pro team
    n2 = 2  # Number of agents in Against team
    n_rounds = 3  # Number of rounds for the debate

    # Define personas for each agent
    pro_personas = [
        "a passionate advocate for technology safety",
        "a tech policy expert arguing for regulation"
    ]
    against_personas = [
        "a tech entrepreneur who believes in innovation freedom",
        "an AI researcher who thinks regulation may hinder progress"
    ]

    # Create agents for each team
    team_pro_agents = [Agent(team="Pro", agent_id=i + 1, persona=pro_personas[i]) for i in range(n1)]
    team_against_agents = [Agent(team="Against", agent_id=i + 1, persona=against_personas[i]) for i in range(n2)]

    # Sample dataset of questions
    questions_df = pd.read_csv("questions.csv")

    # Simulate the debate
    simulate_debate(topic, team_pro_agents, team_against_agents, n_rounds, questions_df)

    
    