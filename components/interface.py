import streamlit as st
import os
import json
from datetime import datetime
from components.uploader import load_and_split, summarise_doc_excerpt, store_embeddings
from config import settings

def render_sidebar():
    st.sidebar.title("📎 Upload to DT")
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more files", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )
    return uploaded_files

def handle_file_uploads(uploaded_files):
    summaries = []
    last_uploaded = None

    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[1]
        temp_path = os.path.join("/tmp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        docs = load_and_split(temp_path, file_ext)
        store_embeddings(docs, namespace="default")

        summary = summarise_doc_excerpt(docs, file.name)
        summaries.append(summary)

        # Persist locally for memory
        try:
            metadata_path = os.path.join("data", "uploaded_documents.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
            else:
                meta = []

            meta.append({
                "filename": file.name,
                "summary": summary,
                "timestamp": datetime.now().isoformat(),
                "storage": ["default"]
            })

            with open(metadata_path, "w") as f:
                json.dump(meta, f, indent=2)

        except Exception as e:
            st.warning(f"⚠️ Failed to store metadata: {e}")

        last_uploaded = summary

    return summaries, last_uploaded

