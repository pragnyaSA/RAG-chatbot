import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# Initialize OpenAI client
try:
    # Check if API key exists
    api_key = os.environ.get("USE_API_key")
    if not api_key:
        print("OPENAI_API_KEY not found in environment.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        print("Or add to your .env file")
        print("Using fallback mode...")
        client = None
    else:
        client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"Error: {e}")
    client = None

# 1. Load + Preprocess Data
def load_transactions(file_path="transactions.json"):
    with open(file_path, "r") as file:
        data = json.load(file)

    texts = [
        f"Transaction ID {t['id']}: On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
        for t in data
    ]
    return data, texts

# 2. Create Embeddings 
def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return model, embeddings

# 3. Retriever 
def retrieve_transactions(query, model, embeddings, texts, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    retrieved = [texts[i] for i in top_indices]
    return retrieved

# 4. LLM-Based Generator
def generate_answer(question, context_list):
    context = "\n".join(context_list)
    
    # Fallback if OpenAI not available
    if client is None:
        return f"OpenAI not configured. Relevant transactions:\n{context}\n\nPlease set OPENAI_API_KEY environment variable."
    
    prompt = f"""
You are a strict RAG chatbot. 
Answer using ONLY this context:
{context}

Question: {question}

If answer not in context, say: "Information not available."
"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)[:100]}"

# 5. Chatbot Loop 
def chatbot():
    print("RAG Chatbot Ready! Type 'exit' to quit.\n")
    
    try:
        data, texts = load_transactions()
        model, embeddings = create_embeddings(texts)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    while True:
        query = input("You: ").strip()
        
        if query.lower() == 'exit':
            break
        if not query:
            continue
        
        try:
            retrieved = retrieve_transactions(query, model, embeddings, texts)
            answer = generate_answer(query, retrieved)
            
            print("\nRetrieved Transactions:")
            for r in retrieved:
                print(f"  • {r}")
            
            print(f"\n Answer: {answer}")
            print("-" * 50 + "\n")
        except Exception as e:
            print(f"Error: {e}")

# Run - NO CHANGE
if __name__ == "__main__":
    chatbot()
