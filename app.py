import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os

# Only initialize index when requested
@st.cache_resource
def load_index():
    docs = SimpleDirectoryReader("second_brain_files").load_data()
    return VectorStoreIndex.from_documents(docs)

# UI
st.set_page_config(page_title="AaronOS — Second Brain", layout="wide")
st.title("🧠 AaronOS — Your Second Brain")

query = st.text_input("Chat with your life archive below:")
if query:
    with st.spinner("Thinking..."):
        index = load_index()
        engine = index.as_query_engine()
        response = engine.query(query)
        st.write(response.response)
