from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
# Require sentence_transformer and PyTorch

def main():
    # Storing model in D drive - Ingeneral, they were automatically saved in C drive
    os.environ['HF_HOME'] = 'D:/huggingface_cache'

    # Downloading Model - Running Locally
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Getting embeddings for a single query
    try:
        result = model.embed_query("The capital of india is New Delhi.")
        print(f"Embedding vector length: {len(result)}")
        print(f"First 5 values: {result[:5]}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Getting embeddings for a batch of query
    documents = [
        "Delhi is the capital of India",
        "Kolkata is the capital of West Bengal",
        "Paris is the capital of France"
    ]
    try:
        result = model.embed_documents(texts=documents)
        print(str(result))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()