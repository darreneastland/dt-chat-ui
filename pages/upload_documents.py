import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import pinecone

# === PAGE CONFIG ===
st.set_page_config(page_title="Upload Reference Documents to DT", page_icon="📁")
st.title("📁 Upload Reference Documents to DT")
st.markdown("Drop documents here to add to the Digital Twin's memory.")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload documents", type=["pdf", "docx", "txt"], label_visibility="collapsed")

if uploaded_file is not None:
    st.write(f"Uploaded: {uploaded_file.name}")

    # Save temporarily
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # === LOAD DOCUMENT ===
    try:
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.error("Unsupported file type.")
            st.stop()

        docs = loader.load()

        # === EMBEDDINGS AND PINECONE ===
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
        pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

      
        from pinecone import Pinecone, ServerlessSpec
        import os

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        index = pc.Index("dt-knowledge")  # or pc.index() if you're using pinecone-client <1.0

        
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
        index_name = "dt-knowledge"

        vectorstore = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        st.success("✅ Document uploaded and indexed successfully!")

    except Exception as e:
        st.error(f"❌ Failed to process document: {str(e)}")

# === FOOTER ===
st.markdown("---")
st.caption("v1.2 – Digital Twin Chat Assistant – Darren Eastland")
