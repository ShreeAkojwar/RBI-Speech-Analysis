import os
import pandas as pd

# Path to your cleaned speeches folder
folder_path = 'rbi_speeches_txt_clean'

# List to hold data
data = []

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        # Extract date from filename (assuming format 'YYYY-MM-DD.txt')
        speech_date = filename.replace('.txt', '')
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            speech_text = file.read()
        data.append({'Speech_Date': speech_date, 'Speech_Text': speech_text})

# Create DataFrame and save as CSV
df = pd.DataFrame(data)
df.to_csv('cleaned_rbi_speeches.csv', index=False)
print("CSV created: cleaned_rbi_speeches.csv")
