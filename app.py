import os
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms.openai import OpenAI
import fitz  # for PDF support
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Load and index the documents
st.title("🧠 AaronOS — Your Second Brain")
st.markdown("Chat with your life archive below:")

@st.cache_resource(show_spinner=True)
def load_index():
    service_context = ServiceContext.from_defaults(llm=OpenAI(api_key=api_key, model="gpt-4"))
    docs = SimpleDirectoryReader("second_brain_files").load_data()
    return VectorStoreIndex.from_documents(docs, service_context=service_context)

index = load_index()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

# User input
user_input = st.chat_input("Ask your Second Brain...")

if user_input:
    response = chat_engine.chat(user_input)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**AaronOS:** {response.response}")
