import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

# FIXED: OpenAI client initialization
try:
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if api_key:
        client = OpenAI(api_key=api_key)
    else:
        # Try without API key (will fail gracefully)
        client = OpenAI()
except:
    client = None

# 1. Load + Preprocess Data 
@st.cache_resource
def load_transactions(file_path="transactions.json"):
    with open(file_path, "r") as file:
        data = json.load(file)

    texts = [
        f"Transaction ID {t['id']}: On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
        for t in data
    ]
    return data, texts

# 2. Create Embeddings
@st.cache_resource
def create_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    return model, embeddings

# 3. Retriever - NO CHANGE
def retrieve_transactions(query, model, embeddings, texts, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    retrieved = [texts[i] for i in top_indices]
    return retrieved

# 4. LLM-Based Answer Generator - MINIMAL FIX
def generate_answer(question, context_list):
    context = "\n".join(context_list)
    
    # Check if client is available
    if client is None:
        return " Error: OpenAI client not initialized. Set OPENAI_API_KEY environment variable."
    
    prompt = f"""
Answer using ONLY this context:
{context}

Question: {question}

If answer not found, say: "Information not available."
"""
    
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f" Error: {str(e)}"

# Streamlit UI - NO CHANGE
st.title("RAG Chatbot for Transactions")
st.write("Ask questions about customer purchases")

# Load data
data, texts = load_transactions()
model, embeddings = create_embeddings(texts)

# Show warning if no API key
if client is None:
    st.error("OPENAI_API_KEY not set. Set it with: `export OPENAI_API_KEY='your-key'`")

# User input
query = st.text_input("Enter your question:", placeholder="e.g., What was Amit's total spending?")

if query:
    retrieved = retrieve_transactions(query, model, embeddings, texts)
    answer = generate_answer(query, retrieved)
    
    st.subheader("Retrieved Context")
    for r in retrieved:
        st.write(f"- {r}")
    
    st.subheader(" Answer")
    st.write(answer)
