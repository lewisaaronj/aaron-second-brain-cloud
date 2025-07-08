import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# Lazy load index
@st.cache_resource
def load_index():
    st.write("🔄 Indexing documents...")
    docs = SimpleDirectoryReader("second_brain_files").load_data()
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return VectorStoreIndex.from_documents(docs, embed_model=embed_model)

# UI
st.set_page_config(page_title="AaronOS — Second Brain", layout="wide")
st.markdown("<h1 style='color:#1E90FF;'>🧠 AaronOS — Your Second Brain</h1>", unsafe_allow_html=True)

query = st.text_input("Chat with your life archive below:")

if query:
    with st.spinner("Thinking..."):
        index = load_index()
        engine = index.as_query_engine()
        response = engine.query(query)
        st.write(response.response)
