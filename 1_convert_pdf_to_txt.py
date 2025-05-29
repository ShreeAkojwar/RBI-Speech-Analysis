import os
from PyPDF2 import PdfReader

input_folder = 'rbi_speeches_raw' # Folder containing the PDF files
output_folder = 'rbi_speeches_txt' # Folder to save the converted text files
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(input_folder, filename)
        txt_path = os.path.join(output_folder, filename.replace('.pdf', '.txt'))
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                txt_file.write(page.extract_text() or '')
print("PDF to TXT conversion complete!")

import os

folder = 'rbi_speeches_txt'  # Folder containing the text files
# Clean up the text files by removing empty lines
for filename in os.listdir(folder):
    if filename.endswith('.txt'):
        path = os.path.join(folder, filename)
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
print("Clean-up complete!")
