################################################################################
##### DEBATE QUESTIONS:                                                  #######
### 1. Do you think AI and technology help or hurt people and society?   #######
### 2. What role should ethics play in the development and use of AI and #######
###    technology in society?                                            #######
### 3. How do you think AI and technology are changing our relationships #######
### with each other and with our work?                                   #######
### 4. Should governments and organizations regulate AI and technology   #######
### to protect individuals and communities?                              #######

######## 3 turns per user in the debate....                              #######
################################################################################

from openai import OpenAI
client = OpenAI()

# Define system prompts for each persona with more specific instructions
personas = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an elder male teacher with a cautious and thoughtful perspective on technology. Please answer the following questions to the best of your ability."
            }
        ]
    },
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an elder female teacher who believes in the power of education to shape responsible technology use. Please answer the following questions with your viewpoint."
            }
        ]
    },
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a young female student who is excited about the potential of technology. Please share your thoughts with enthusiasm."
            }
        ]
    },
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a young male student, skeptical but curious about technology's role in society. Answer with your perspective."
            }
        ]
    }
]

# Define all questions for the debate
questions = [
    "Do you think AI and technology help or hurt people and society?",
    "What role should ethics play in the development and use of AI and technology in society?",
    "How do you think AI and technology are changing our relationships with each other and with our work?",
    "Should governments and organizations regulate AI and technology to protect individuals and communities?"
]

# Define a persistent conversation history to keep responses across questions
conversation_history = []

# Number of rounds for each question
rounds = 3

# Function to get response using the specified `client.chat.completions.create` format
def get_response(persona, messages):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            persona,
            *messages
        ],
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

# Loop through each question and continue the debate in a round-robin format
for question in questions:
    # Add the current question to the conversation history
    conversation_history.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": question
            }
        ]
    })
    
    print(f"\n--- Starting discussion on question: {question} ---\n")
    
    # Perform round-robin discussion for each question
    for _ in range(rounds):
        for persona in personas:
            # Get response from current persona
            response_text = get_response(persona, conversation_history)
            
            # Append persona's response to conversation history
            conversation_history.append({
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": response_text
                    }
                ]
            })
            
            # Print the response to visualize the conversation
            print(f"{persona['content'][0]['text']} says: {response_text}\n")

# need to incorporate the asking questions after the debate
