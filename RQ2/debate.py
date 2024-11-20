from openai import OpenAI
import pandas as pd
from questions import questions_abortion
from statistics import mean

# Placeholder function for calling the OpenAI GPT API
def call_gpt_agent(persona, team, agent_id, prompt):
    # Customize this function with your OpenAI API key
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": f"You are {persona} on Team {team} (Agent {agent_id}). Your goal is to convince the other team that your stance is correct while providing strong arguments and rebuttals. Use all debating tactics, including logical reasoning, emotional appeals, and addressing counterarguments, to try to make the opposing team agree with your position. Remember what arguments have been said before and avoid repeating old arguments. Focus on rebutting the opponent's main points while introducing new critical points. Limit your response in 50 words."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4o",
        temperature=1,  # Increase temperature for more varied responses
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
        self.responses = []

    def debate(self, prompt):
        return call_gpt_agent(self.persona, self.team, self.agent_id, prompt)

    def answer_question(self, question, response_options):
        # Include response options in the prompt
        options_text = ", ".join([f"'{option}'" for option in response_options.keys()])
        prompt = f"{question} \n Respond only with one of the following options: {options_text}"
        response = call_gpt_agent(self.persona, self.team, self.agent_id, prompt)
        self.responses.append(response)
        return response

def calculate_subscale_scores(results_df, time_label, general_support_questions):
    """
    Calculate the average score for General Support questions for each agent.
    """
    # Filter responses by time and relevant questions
    # print(f"Calculating scores for time: {time_label}")
    # print(f"Available times: {results_df['Time'].unique()}")
    # print(f"Available questions: {results_df['Question'].unique()}")

    filtered_df = results_df[results_df["Time"] == time_label].copy()

    # # Handle empty DataFrame
    # if filtered_df.empty:
    #     print(f"No data found for time: {time_label} and questions: {general_support_questions}")
    #     return pd.DataFrame(columns=["AgentID", "TeamID", "AverageScore"])

    # Score responses
    def score_response(response, response_options):
        for option, score in response_options.items():
            if option.lower() in response.lower():
                return score
        return 0  # Default score for unexpected responses

    filtered_df["Score"] = filtered_df.apply(
        lambda row: score_response(row["Response"], row["ResponseOptions"]), axis=1
    )

    # Calculate the average score for General Support questions for each agent
    final_scores = filtered_df.groupby(["AgentID", "TeamID"]).agg(AverageScore=("Score", "mean"))

    return final_scores


# Run the debate simulation multiple times and calculate average scores per agent
def repeat_simulation(topic, team_pro_personas, team_against_personas, n_rounds, questions, n_trials):
    # Initialize DataFrames to store cumulative scores for all trials
    cumulative_before_scores = pd.DataFrame()
    cumulative_after_scores = pd.DataFrame()

    for trial in range(n_trials):
        print(f"\n===== Trial {trial + 1} of {n_trials} =====\n")

        # Create new agents for each trial
        team_pro_agents = [
            Agent(team="Pro", agent_id=i + 1, persona=team_pro_personas[i]) for i in range(len(team_pro_personas))
        ]
        team_against_agents = [
            Agent(team="Against", agent_id=i + 1, persona=team_against_personas[i]) for i in range(len(team_against_personas))
        ]

        # Simulate the debate and collect scores
        before_scores, after_scores = simulate_debate_once(
            topic, team_pro_agents, team_against_agents, n_rounds, questions
        )

        # Add trial scores to cumulative DataFrames
        cumulative_before_scores = pd.concat([cumulative_before_scores, before_scores])
        cumulative_after_scores = pd.concat([cumulative_after_scores, after_scores])

    # Calculate the final average score for each agent
    final_before_scores = cumulative_before_scores.groupby(["AgentID", "TeamID"]).mean()
    final_after_scores = cumulative_after_scores.groupby(["AgentID", "TeamID"]).mean()

    print("\n===== Final Results Across All Trials =====")
    print("Final Average Before Scores Per Agent:\n", final_before_scores)
    print("Final Average After Scores Per Agent:\n", final_after_scores)

    return final_before_scores, final_after_scores

# Simulate debate once and return scores as DataFrames
def simulate_debate_once(topic, team_pro_agents, team_against_agents, n_rounds, questions):
    # Convert questions to a DataFrame
    questions_df = pd.DataFrame(questions)

    # Define General Support questions
    general_support_questions = [
        q["question"]
        for q in questions
        if "General support" in q.get("subscale", "")
    ]

    # Initialize an empty DataFrame
    results = pd.DataFrame(columns=["Question", "Response", "AgentID", "TeamID", "ResponseOptions", "Time"])

    # Create a list to store results temporarily
    results_list = []

    # Administer questions before the debate
    for _, question_row in questions_df.iterrows():
        question = question_row["question"]
        response_options = question_row["response_options"]
        for agent in team_pro_agents + team_against_agents:
            response = agent.answer_question(question, response_options)
            results_list.append({
                "Question": question,
                "Response": response,
                "AgentID": agent.agent_id,
                "TeamID": agent.team,
                "ResponseOptions": response_options,
                "Time": "before"
            })

    # Conduct debate rounds
    transcript = f"Debate topic: {topic}\n \nTeam Pro argues in favor, while Team Against argues against.\n"
    for round_num in range(1, n_rounds + 1):
        # Pro team speaks
        for agent in team_pro_agents:
            agent.debate(transcript)
        # Against team speaks
        for agent in team_against_agents:
            agent.debate(transcript)

    # Administer questions after the debate
    for _, question_row in questions_df.iterrows():
        question = question_row["question"]
        response_options = question_row["response_options"]
        for agent in team_pro_agents + team_against_agents:
            response = agent.answer_question(question, response_options)
            results_list.append({
                "Question": question,
                "Response": response,
                "AgentID": agent.agent_id,
                "TeamID": agent.team,
                "ResponseOptions": response_options,
                "Time": "after"
            })

    # Concatenate all the results into the DataFrame
    results = pd.concat([results, pd.DataFrame(results_list)], ignore_index=True)

    # Calculate scores for before and after the debate
    before_scores_df = calculate_subscale_scores(results, "before", general_support_questions)
    after_scores_df = calculate_subscale_scores(results, "after", general_support_questions)

    return before_scores_df, after_scores_df

# Main function to run the experiment
if __name__ == "__main__":
    # Configuration
    topic = "Should abortion be banned?"
    n1 = 2  # Number of agents in Pro team
    n2 = 1  # Number of agents in Against team
    n_rounds = 6  # Number of rounds for the debate
    n_trials = 5  # Number of trials to run

    # Define personas for each agent
    team_pro_personas = ["An old male who only finishes high school and has low income"]
    team_against_personas = ["A young female who has a college degree and high income"]

    # Run the repeated simulation
    final_before_scores, final_after_scores = repeat_simulation(
        topic, team_pro_personas, team_against_personas, n_rounds, questions_abortion, n_trials
    )