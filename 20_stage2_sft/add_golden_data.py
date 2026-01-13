import json
import os

output_file = "data_paper_sft/train.jsonl"

golden_data = [
    {
        "question": "What is CLaRa?",
        "answer": "CLaRa stands for Continuous Latent Reasoning. It is a unified framework that performs embedding-based compression and joint optimization of retrieval and generation in a shared continuous space. It uses SCP for data synthesis and trains the reranker and generator end-to-end via a single language modeling loss."
    },
    {
        "question": "What is SCP and what is its role in CLaRa?",
        "answer": "SCP stands for Salient Compressor Pretraining. It is a key-preserving data synthesis framework introduced in CLaRa to obtain semantically rich and retrievable compressed vectors. It uses QA and paraphrase supervision to teach the compressor to retain essential semantic information."
    },
    {
        "question": "How does joint training work in CLaRa?",
        "answer": "CLaRa trains the reranker (Query Reasoner) and generator end-to-end via a single language modeling loss. It employs a differentiable top-k estimator to allow gradients to flow from the generator back to the retriever, aligning retrieval relevance with answer quality."
    }
]

print(f"Injecting {len(golden_data)} Golden Data samples into {output_file}...")

with open(output_file, 'a', encoding='utf-8') as f:
    for item in golden_data:
        text = f"User: {item['question']}\n\nAssistant: {item['answer']}"
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print("Done.")
