import streamlit as st
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
import os

# Page setup
st.set_page_config(page_title="Seed DT Memory", page_icon="🌱")
st.title("🌱 Seed DT Persistent Memory")

# Load keys
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
pinecone_key = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]

# Init embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)

# Define test doc
doc = Document(page_content="This is a memory test — DT now remembers this.")

# Push to Pinecone under dt-memory namespace
try:
    vectorstore = PineconeVectorStore.from_documents(
        documents=[doc],
        embedding=embedding_model,
        index_name="dt-knowledge",
        namespace="dt-memory"
    )
    st.success("✅ Successfully seeded 'dt-memory' namespace with a test vector.")
except Exception as e:
    st.error(f"❌ Failed to seed memory: {e}")
