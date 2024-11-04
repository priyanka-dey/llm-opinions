# File to get each persona's opinions before the debate

import pandas as pd 

from openai import OpenAI
client = OpenAI()

questions = pd.read_csv("compiled_questions.csv")
print(questions.head())

results = []
querys = []

for i, q in enumerate(questions.Question.values):
    quest = q 
    resp = questions.Responses.values[i] 

    print(quest)
    print(resp)

    query = quest + "\nPlease respond with EXACTLY one of these phrases: " + resp
    querys.append(query)

    print(query)
    print("*"*100)

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are a old male teacher.  Please answer the following questions to the best of your ability."
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text":  query
            }
        ]
        }
    ],
    temperature=1,
    max_tokens=8,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={
        "type": "text"
    }
    )

    result = (response.choices[0].message.content)

    results.append(result)

df = pd.DataFrame() 
df['Query'] = querys 
df['Results'] = results
df.to_csv("old_male_teacher.csv", index=False)


