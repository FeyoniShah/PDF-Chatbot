import streamlit as st
import requests
import PyPDF2
import tempfile
import os
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# CONFIG
GOOGLE_API_KEY = 'AIzaSyCFkkWOx2ZS9vgdhAlyibWtgvD-Bx7cOvY'  
CHROMA_COLLECTION_NAME = "pdf_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

#SETUP 

genai.configure(api_key=GOOGLE_API_KEY)
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
if CHROMA_COLLECTION_NAME in chroma_client.list_collections():
    chroma_client.delete_collection(name=CHROMA_COLLECTION_NAME)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)


#PDF Text extract

def extract_pdf_text_from_url(pdf_url):
    response = requests.get(pdf_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    pdf_text = ""
    with open(tmp_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            pdf_text += page.extract_text() or ""
    os.remove(tmp_file_path)
    return pdf_text

# ChromaDB store

def add_text_to_chroma(text):
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embeddings = model.encode(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

# ChromaDB retrieve

def retrieve_context(query, top_k=4):
    query_embedding = model.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n\n".join(results['documents'][0]) if results['documents'] else ""

# chat

def chat_with_pdf(query, context):
    system_prompt = """You are a helpful assistant. Use the following context extracted from a PDF document to answer the user's query precisely. If the context is insufficient, say 'Not enough information in the PDF to answer this.'"""
    full_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nUser: {query}\nAssistant:"
    
    response = genai.GenerativeModel('models/gemini-1.5-flash').generate_content(full_prompt)
    return response.text

# STREAMLIT UI 

st.set_page_config(page_title="📄 PDF Chatbot with Gemini", layout="wide")

st.title("📄 Chat with your PDF (RAG + Gemini 1.5 Flash)")
pdf_url = st.text_input("🔗 Enter a PDF URL:")

if pdf_url and st.button("🚀 Process PDF"):
    with st.spinner("Downloading and processing PDF..."):
        try:
            text = extract_pdf_text_from_url(pdf_url)
            add_text_to_chroma(text)
            st.success("PDF processed successfully. Ask your questions below!")
            st.session_state["pdf_ready"] = True
        except Exception as e:
            st.error(f"Failed to process PDF: {e}")
            st.session_state["pdf_ready"] = False

if st.session_state.get("pdf_ready", False):
    user_query = st.text_input("💬 Ask a question about the PDF:")
    if user_query:
        with st.spinner("Thinking..."):
            context = retrieve_context(user_query)
            answer = chat_with_pdf(user_query, context)
            st.markdown(f"**🤖 Answer:** {answer}")
