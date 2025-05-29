# Import the os module for interacting with the operating system (file and folder operations)
import os

# Import the re module for regular expressions (pattern matching and text replacement)
import re

# Import the string module for easy access to string constants (like punctuation)
import string

# Import nltk, a popular library for natural language processing tasks
import nltk

# Download the stopwords list from NLTK (only needs to be done once per environment)
nltk.download('stopwords')

# Import the stopwords corpus from nltk, which contains common English words to remove
from nltk.corpus import stopwords

# Set the folder where your original .txt files are stored
input_folder = 'rbi_speeches_txt'
# Set the folder where cleaned files will be saved
output_folder = 'rbi_speeches_txt_clean'
# Create the output folder if it doesn't already exist
os.makedirs(output_folder, exist_ok=True)

# Load the set of English stopwords for quick lookup
stop_words = set(stopwords.words('english'))

# Define a function to clean the text
def clean_text(text):
    # Convert all text to lowercase for consistency
    text = text.lower()
    
    # Replace multiple whitespace characters (including newlines) with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove all punctuation using string.punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove all numbers from the text
    text = re.sub(r'\d+', '', text)
    
    # Split the text into words and remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Join the filtered words back into a single string
    text = ' '.join(words)
    
    # Return the cleaned text
    return text

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Process only .txt files
    if filename.endswith('.txt'):
        # Open the input file for reading
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as infile:
            # Read the entire file content
            raw_text = infile.read()
            # Clean the text using the function defined above
            cleaned = clean_text(raw_text)
        # Open the output file for writing the cleaned text
        with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as outfile:
            outfile.write(cleaned)

# Print a message when all files have been processed
print("All files cleaned and saved!")
