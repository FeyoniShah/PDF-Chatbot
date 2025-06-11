# 📄 RAG-Powered PDF Chatbot

An AI chatbot that reads a PDF from a given URL, stores the content in a vector database (ChromaDB) using SentenceTransformer embeddings, and answers user queries using **Gemini 1.5 Flash** with context-aware responses.

---

## 🚀 Features

- 🔗 Accepts PDF URL for dynamic ingestion  
- 📚 Extracts text from PDF and chunks it semantically  
- 🤖 Embeds using `all-MiniLM-L6-v2` SentenceTransformer  
- 💾 Stores & retrieves top-k relevant chunks via ChromaDB  
- 🧠 Generates answers using Gemini 1.5 Flash with system prompt  
- 💬 Chat-style interaction via terminal  

---

## 🛠️ Technologies Used

- Python 3.8+
- SentenceTransformers
- ChromaDB
- PyPDF2
- Google Generative AI (Gemini)
- Requests

---
