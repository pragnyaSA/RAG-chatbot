# RAG Transaction Chatbot

This project is a **RAG (Retrieval Augmented Generation)** based chatbot that can answer questions about customer transactions. The system uses a combination of **Sentence Transformers**, **OpenAI's GPT models** (for LLM generation), and **Streamlit** for the web interface.

The chatbot works in two modes:

 **main.py** → Terminal/Console-based version
 **app.py** → Streamlit Web UI version



## What This Project Does

 **Loads transaction data** from a JSON file
 **Converts each transaction** into a readable sentence
 **Creates embeddings** using a pretrained SentenceTransformer model
 **Searches for the most relevant transactions** based on cosine similarity
 **Filters answers based on customer** (Amit, Riya, Karan)
 **Generates detailed responses** using **OpenAI's GPT models** (via RAG - Retrieval Augmented Generation pipeline)
 **Shows the results** either in the terminal or through a Streamlit web page


## Features

### With this chatbot, you can ask things like:

 **"Tell me Amit’s purchase history"**
 **"Show Riya’s transactions"**
 **"What did Karan buy?"**

The chatbot will retrieve the most relevant transactions and generate human like answers.

## Dataset Used (transactions.json)

We use a small sample dataset to test the chatbot's functionality. 

## Requirements

pip install sentence-transformers
pip install numpy
pip install scikit-learn
pip install streamlit
pip install openai

## To Run :-

streamlit run app.py ,
python main.py (terminal version)

