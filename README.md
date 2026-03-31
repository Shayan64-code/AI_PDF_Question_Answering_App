# AI-powered PDF question answering system using RAG.

This WebApp is created for generate answer using the specific context (GenAI PDF), This project it created using Streamlit, Langchain, HuggingFace, RAG, and etc.

Architecture:

PDF
↓
Chunking
↓
Embeddings
↓
Vector Database
↓
Retriever
↓
LLM

Tech Stack:

Python

LangChain

Streamlit

ChromaDB

HuggingFace embeddings

Installation:

pip install -r requirements.txt
streamlit run streamlit_app.py