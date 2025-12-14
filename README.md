# 🧠 Personal RAG Assistant (OpenAI Edition)

## 📌 Overview
This is a robust **Retrieval-Augmented Generation (RAG)** system built to bridge the gap between Large Language Models (LLMs) and private data. 

Unlike a standard ChatGPT session, this assistant can read, understand, and answer questions based on a specific set of local documents (PDFs and Text files) that I provide. It uses **OpenAI's state-of-the-art models** for both embedding and generation, ensuring high accuracy and relevant source citations.

## 🚀 Features
* **Context-Aware:** Uses specific document data to answer questions, reducing hallucinations.
* **Conversational Memory:** Remembers the chat history to handle follow-up questions effectively.
* **Source Citations:** Returns the exact document name used to generate the answer.
* **Persistent Vector Store:** Saves embeddings to disk (ChromaDB) so documents only need to be processed once.

## 🛠️ Tech Stack
* **Orchestration:** LangChain (LCEL - LangChain Expression Language)
* **LLM:** OpenAI `gpt-4o-mini` (Optimized for speed and cost)
* **Embeddings:** OpenAI `text-embedding-3-small`
* **Vector Database:** ChromaDB
* **Document Parsing:** PyPDF

## 📂 Project Structure
```text
my_rag_project/
│
├── documents/           # 📄 Place PDFs and .txt files here
├── chroma_db/           # 💾 Vector store (auto-generated)
├── .env                 # 🔑 API Keys
├── rag_system.py        # 🐍 Main application logic
├── requirements.txt     # 📦 Dependencies
└── README.md            # 📖 Documentation