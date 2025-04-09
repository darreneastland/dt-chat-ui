import streamlit as st
import os
import json
from collections import defaultdict
from datetime import datetime
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# === PAGE CONFIG ===
st.set_page_config(page_title="📦 Backfill Memory Metadata", page_icon="📄")
st.title("📦 Backfill Metadata for Previously Uploaded Documents")
st.markdown("This tool extracts and summarises documents already in DT's memory.")

# === API KEYS ===
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

index_name = "dt-knowledge"
namespace = "dt-memory"
metadata_file = "uploaded_documents.json"

if st.button("▶️ Run Metadata Backfill"):
    try:
        # === Connect to Pinecone ===
        pc = PineconeClient(api_key=pinecone_api_key, environment=pinecone_env)
        index = pc.Index(index_name)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
        llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        # === Query Vectors in Namespace ===
        st.info("Querying Pinecone index for existing memory chunks...")
        response = index.query(vector=[0.0]*1536, top_k=1000, namespace=namespace, include_metadata=True)
        matches = response.get("matches", [])

        grouped_docs = defaultdict(list)
        for match in matches:
            meta = match.get("metadata", {})
            if "text" in meta and "source_file" in meta:
                grouped_docs[meta["source_file"]].append(Document(page_content=meta["text"], metadata=meta))

        if not grouped_docs:
            st.warning("No valid documents found in memory to summarise.")
        else:
            st.success(f"Found {len(grouped_docs)} distinct documents. Processing...")

        existing = []
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                existing = json.load(f)

        for filename, docs in grouped_docs.items():
            st.write(f"📄 Summarising `{filename}`...")
            summary = qa_chain.run(input_documents=docs[:5], question="Summarise this document for upload tracking.")
            inferred_type = "strategy" if "strategy" in filename.lower() else "general"
            timestamp = docs[0].metadata.get("uploaded_at", datetime.utcnow().isoformat())

            existing.append({
                "filename": filename,
                "summary": summary,
                "type": inferred_type,
                "timestamp": timestamp
            })

        with open(metadata_file, "w") as f:
            json.dump(existing, f, indent=2)

        st.success("✅ Metadata successfully backfilled and saved.")

    except Exception as e:
        st.error(f"❌ Error during backfill: {e}")

# === FOOTER ===
st.markdown("---")
st.caption("v1.0 – Backfill Metadata Tool – Darren Eastland")
