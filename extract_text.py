import os
from pypdf import PdfReader

# Paths to folders
raw_dir = "data/raw"
processed_dir = "data/processed"

# Loop through every file in raw folder
for filename in os.listdir(raw_dir):
    if filename.endswith(".pdf"):
        print(f"Reading {filename}...")
        
        # 1. Load the PDF
        reader = PdfReader(os.path.join(raw_dir, filename))
        full_text = ""
        
        # 2. Loop through pages and extract text
        for page in reader.pages:
            full_text += page.extract_text() + "\n"
        
        # 3. Save as a .txt file in the 'processed' folder
        txt_filename = filename.replace(".pdf", ".txt")
        with open(os.path.join(processed_dir, txt_filename), "w", encoding="utf-8") as f:
            f.write(full_text)

print("Done! Check 'data/processed' folder.")