import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

# === Streamlit UI ===
st.set_page_config(page_title="Upload Documents to DT", page_icon="📁", layout="centered")
st.title("📁 Upload Reference Documents to DT")
st.markdown("Drop documents here to add to the Digital Twin's memory.")

# === API Keys ===
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = "us-east-1"  # Replace if different
pinecone_index_name = "dt-knowledge"

# === Upload & Process ===
uploaded_files = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files and openai_api_key and pinecone_api_key:
    with st.spinner("🔍 Processing documents..."):
        all_docs = []

        for uploaded_file in uploaded_files:
            file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            if file_ext == ".pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_ext == ".docx":
                loader = Docx2txtLoader(tmp_file_path)
            elif file_ext == ".txt":
                loader = TextLoader(tmp_file_path)
            else:
                st.warning(f"Unsupported file type: {file_ext}")
                continue

            docs = loader.load()
            all_docs.extend(docs)

        # === Split & Embed ===
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        index = Pinecone.from_documents(documents=chunks, embedding=embeddings, index_name=pinecone_index_name)

        st.success(f"✅ {len(uploaded_files)} document(s) uploaded and embedded into DT memory.")

elif not openai_api_key or not pinecone_api_key:
    st.error("Missing API keys. Please ensure your `OPENAI_API_KEY` and `PINECONE_API_KEY` are set.")
