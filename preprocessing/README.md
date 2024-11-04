# Preprocessing 

## Procedure
* Load Model (Llama3.2)
* Load PersonaHub
* Extract the top 10 professions from the PersonaHub Dataset: [('historian', 3242), ('data analyst', 2127), ('genealogist', 1217), ('sports journalist', 926), ('researcher', 901), ('journalist', 900), ('data scientist', 743), ('sports analyst', 711), ('writer', 690), ('travel blogger', 659)]
* Generate prompts that describe the description of the agent: gender, age, profession
* Create the agents, including gender, age, and profession

## Global health related questions/prompts (finalized)
1. Should the US goverment be involved in shaping global health policy? Please respond with EXACTLY one of these phrases: Yes, No.
2. Is universal health care an important goal for improving global health? Please respond with EXACTLY one of these phrases: Yes, No.
3. Should access to healthcare be considered a fundamental human right worldwide? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.
4. Should pharmaceutical companies by required to make their products affordable and accessible in low-income countries? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.
5. Are the World Health Orangization (WHO) and other international health agencies effective in managing global health crises? Please respond with EXACTLY one of these phrases: Completely disagree, completely agree, Don't know.
6. Should vaccines by made mandatory worldwide for infectious diseases such as COVID-19 and measles? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly Oppose.
7. Can large-scale international aid improve health outcomes, or does it create dependency? Please respond with EXACTLY one of these phrases: Improve health outcomes, Create dependency
8. Should mental health receive the same funding and focus as physical health in global health initiatives? Please respond with EXACTLY one of these phrases: Completely disagree, completely agree, Don't know.
9. Should governments impose taxes on unhealthy products (sugar, tobacco, alcohol) to promote public health? Please respond with EXACTLY one of these phrases: Complete disagree, completely agree, Don't know.
10. Is it fair for wealthier countries to prioritze their own citizens for vaccines or healthcare treatments during a global health crisis? Please respond with EXACTLY one of these phrases: Yes, No.
11. Should governments have the right to mandate health measures such as lockdowns, even if they affect individuals' livelihoods and freedoms? Please respond with EXACTLY one of these phrases: Yes, No.
12. Should wealthy nations take greater responsibility for the health effects of climate change in vulnerable regions? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.
13. Are anti-smoking and anti-alcohol campaigns in low-income countries an effective use of global health resources? Please respond with EXACTLY one of these phrases: Complete disagree, completely agree, Don't know.
14. Should governments regulate the marketing of fast food and sugary drinks to reduce obesity rates?
15. Is it more beneficial to invest in healthcare education within low-income countries than to provide direct aid?
16. Should governments regulate the marketing of fast food and sugary drinks to reduce obesity rates?
17. Do international health organizations have the right to override national sovereignty to manage a public health crisis?
18. Should health interventions prioritize child mortality reduction over adult healthcare needs?
19. Should wealthy countries pay a global “health equity tax” to support low-income nations’ healthcare systems?
20. Are social media platforms responsible for preventing the spread of health misinformation during global health crises?



## Draft Questions/Prompts
1. How can the impacts of climate change on global health be minimized through adaptation and resilience? (not in final)
2. What is one of the most influential global health innovations? (not in final)
3. Should the US government be involved in shaping global health policy? Please respond with EXACTLY one of these phrases: Yes, No
4. Is universal health care an important goal for improving global health? Please respond with EXACTLY one of these phrases: Yes, No.
5. Should access to healthcare be considered a fundamental human right worldwide? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.
6. Should pharmaceutical companies by required to make their products affordable and accessible in low-income countries? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.
7. Are the World Health Orangization (WHO) and other international health agencies effective in managing global health crises? Please respond with EXACTLY one of these phrases: Completely disagree, completely agree, Don't know.
8. Should vaccines by made mandatory worldwide for infectious diseases such as COVID-19 and measles? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly Oppose.
9. Can large-scale international aid improve health outcomes, or does it create dependency? Please respond with EXACTLY one of these phrases: Improve health outcomes, Create dependency.
10. Should mental health receive the same funding and focus as physical health in global health initiatives? Please respond with EXACTLY one of these phrases: Completely disagree, completely agree, Don't know.
11. Should governments impose taxes on unhealthy products (sugar, tobacco, alcohol) to promote public health? Please respond with EXACTLY one of these phrases: Complete disagree, completely agree, Don't know.
12. Is it fair for wealthier countries to prioritze their own citizens for vaccines or healthcare treatments during a global health crisis? Please respond with EXACTLY one of these phrases: Yes, No.
13. Should governments have the right to mandate health measures such as lockdowns, even if they affect individuals' livelihoods and freedoms? Please respond with EXACTLY one of these phrases: Yes, No.
14. Should wealthy nations take greater responsibility for the health effects of climate change in vulnerable regions? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.
15. Are anti-smoking and anti-alcohol campaigns in low-income countries an effective use of global health resources? Please respond with EXACTLY one of these phrases: Strongly favor, Favor, Oppose, Strongly oppose.

## Debate Questions
1. How can the impacts of climate change on global health be minimized through adaptation and resilience? 
2. What is one of the most influential global health innovations? 
3. What ethical boundaries should exist around the collection and use of health data in the fight against pandemics?
4. How can high-income countries support low-income countries' healthcare systems without creating dependency?
5. How can we prepare for and prevent the next global health pandemic given the lessons from COVID-19?

## File Directory
* /agent_answers: csv files for each agent's answers

## Execution Instructions:
python ~/llm-opinions/preprocessing/main.py



    
