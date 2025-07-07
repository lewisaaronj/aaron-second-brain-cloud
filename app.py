import streamlit as st
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Only initialize index when requested
@st.cache_resource
def load_index():
    docs = SimpleDirectoryReader("second_brain_files").load_data()
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return VectorStoreIndex.from_documents(docs, embed_model=embed_model)

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
