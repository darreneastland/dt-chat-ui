import streamlit as st
import os
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.vectorstores import Pinecone  # avoid name clash
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# === PAGE CONFIG ===
st.set_page_config(page_title="Upload Reference Documents to DT", page_icon="📁")
st.title("📁 Upload Reference Documents to DT")
st.markdown("Drop documents here to add to the Digital Twin's memory.")

# === FILE UPLOAD ===
uploaded_files = st.file_uploader("Upload one or more documents", type=["pdf", "docx", "txt"], label_visibility="collapsed", accept_multiple_files=True)

if uploaded_files:
    openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")
    index_name = "dt-knowledge"

    pc = PineconeClient(api_key=pinecone_api_key, environment=pinecone_env)

    # Optional index creation safeguard
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

    for uploaded_file in uploaded_files:
        st.write(f"Uploaded: {uploaded_file.name}")
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                continue

            raw_docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(raw_docs)

            # Attach metadata to each document chunk
            enriched_docs = [
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "source_file": uploaded_file.name,
                        "uploaded_by": "Darren Eastland",
                        "file_type": uploaded_file.name.split(".")[-1],
                        "chunk_index": i
                    }
                )
                for i, doc in enumerate(split_docs)
            ]

            Pinecone.from_documents(
                documents=enriched_docs,
                embedding=embeddings,
                index_name=index_name
            )

            st.success(f"✅ {uploaded_file.name} embedded and uploaded with metadata to DT memory.")

        except Exception as e:
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")

# === FOOTER ===
st.markdown("---")
st.caption("v1.24 – Multi-file Upload + Metadata – Digital Twin Chat Assistant – Darren Eastland")
