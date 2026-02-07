import os
import re

def clean_text(text):
    """
    Scrubs the raw text to make it suitable for a Transformer.
    """
    # 1. Remove obvious page markers
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # 2. Fix words split by line breaks (e.g., "com- pliance" -> "compliance")
    # This is vital for tokenization accuracy
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    
    # 3. Remove non-ASCII characters that often hide in PDFs
    text = text.encode("ascii", "ignore").decode()
    
    # 4. Collapse multiple spaces/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    processed_dir = "data/processed"
    
    for filename in os.listdir(processed_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(processed_dir, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            cleaned_text = clean_text(raw_text)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            
            print(f"Successfully cleaned: {filename}")

if __name__ == "__main__":
    main()