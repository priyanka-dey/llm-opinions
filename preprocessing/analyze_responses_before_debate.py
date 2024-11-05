"""
Original author: Priyanka Dey

Same as the before_debate_responses.ipynb
"""


# Initialize an empty dictionary to hold data for the DataFrame
import os 
data = {}
data_dir = 'INSERT PATH HERE'

# Load each CSV file and add its data to the dictionary
for filename in os.listdir(data_dir):
    if filename.endswith('.csv'):
        # Load the file into a DataFrame
        path = os.path.join(data_dir, filename)
        df = pd.read_csv(path)
        
        # Set the question index as the index for easy merging
        df.set_index('Query', inplace=True)
        
        # Add the answers to the data dictionary with filename as the key
        data[filename.split(".csv")[0]] = df['Results']

# Combine all data into a single DataFrame
combined_df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(combined_df.head())

combined_df = combined_df.reset_index()


value_counts_per_question = combined_df.drop(columns=["Query"]).apply(lambda row: row.value_counts(), axis=1)

# Display the result
print(value_counts_per_question)

value_counts_per_question = combined_df.drop(columns=["Query"]).apply(lambda row: row.value_counts(), axis=1)

# Display the result
value_counts_per_question.to_csv("responses.csv", index=False)
