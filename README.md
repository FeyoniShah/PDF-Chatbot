📄 RAG-Powered PDF Chatbot
An AI chatbot that reads a PDF from a given URL, stores the content in a vector database (ChromaDB) using SentenceTransformer embeddings, and answers user queries using Gemini 1.5 Flash with context-aware responses.

🚀 Features
🔗 Accepts PDF URL for dynamic content ingestion

📚 Extracts text from PDF and splits into semantic chunks

🤖 Embeds using all-MiniLM-L6-v2 SentenceTransformer

💾 Stores & retrieves top-k relevant chunks using ChromaDB

🧠 Generates answers using Gemini 1.5 Flash via system prompting

🗨️ Chat in a loop until exited

🛠️ Technologies Used
Python 3.8+

SentenceTransformers

ChromaDB

PyPDF2

Google Generative AI (Gemini)

requests

