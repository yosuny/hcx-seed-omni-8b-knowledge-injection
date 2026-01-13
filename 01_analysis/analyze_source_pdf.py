import pdfplumber
import os
from collections import Counter
import re

PDF_PATH = "knowledge_data/2511.18659v2.pdf"

def analyze_pdf(pdf_path):
    print(f"Analyzing {pdf_path}...")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    full_text = ""
    page_count = 0
    
    with pdfplumber.open(pdf_path) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"

    # Tokenize (simple splitting)
    words = re.findall(r'\b\w+\b', full_text.lower())
    
    # Simple stop words lists
    stop_words = set([
        "the", "of", "and", "a", "to", "in", "is", "that", "for", "on", "are", "with", "as", 
        "by", "it", "be", "an", "this", "which", "at", "from", "can", "model", "we", "training",
        "learning", "models", "data", "results", "based", "proposed", "method", "using", "performance",
        "et", "al", "dataset", "task", "large", "language", "show", "our", "table", "figure"
    ])
    
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    word_counts = Counter(filtered_words)
    
    print("\n" + "="*40)
    print(f" [ Analysis Result for {os.path.basename(pdf_path)} ]")
    print("="*40)
    print(f"- Total Pages     : {page_count}")
    print(f"- Total Characters: {len(full_text)}")
    print(f"- Total Words     : {len(words)}")
    print("-" * 40)
    print(" [ Top 20 Keywords ]")
    for word, count in word_counts.most_common(20):
        print(f"  {word:<15}: {count}")
    print("="*40)

if __name__ == "__main__":
    analyze_pdf(PDF_PATH)
