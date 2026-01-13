import pdfplumber
import json
import os
import argparse

def extract_pdf_to_jsonl(pdf_path, output_jsonl):
    print(f"Extracting text from {pdf_path}...")
    
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    extracted_data = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Clean up text a bit (optional)
                text = text.strip()
                if len(text) > 50: # Filter out very short pages/empty pages
                    extracted_data.append({"text": text, "meta": {"page": i + 1, "source": os.path.basename(pdf_path)}})
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} pages...")

    print(f"Total extracted pages: {len(extracted_data)}")
    
    # Save to JSONL
    print(f"Saving to {output_jsonl}...")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for entry in extracted_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from PDF to JSONL")
    parser.add_argument("input_pdf", nargs='?', help="Path to input PDF file", default="data/CES 2026_20260112_최종본_compressed.pdf")
    parser.add_argument("output_jsonl", nargs='?', help="Path to output JSONL file", default="data_solverx_cpt/ces_2026.jsonl")
    
    args = parser.parse_args()
    
    extract_pdf_to_jsonl(args.input_pdf, args.output_jsonl)
