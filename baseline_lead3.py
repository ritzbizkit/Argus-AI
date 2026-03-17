import os

def get_lead3_summary(text):
    # Split by period to get sentences, take first 3, and rejoin
    sentences = text.split('.')
    lead3 = '. '.join(sentences[:3]).strip() + '.'
    return lead3

def main():
    processed_dir = "data/processed"
    
    print("--- Argus AI: Lead-3 Baseline Results ---")
    for filename in os.listdir(processed_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(processed_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
            
            baseline_summary = get_lead3_summary(content)
            print(f"\nFILE: {filename}")
            print(f"BASELINE: {baseline_summary[:200]}...") # Print first 200 chars

if __name__ == "__main__":
    main()