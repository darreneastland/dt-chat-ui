# view_memory.py – DT Memory Viewer Interface

import streamlit as st
import os
from pinecone import Pinecone as PineconeClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# === CONFIGURATION ===
st.set_page_config(page_title="🧠 DT Memory Viewer", page_icon="🗂️")
st.title("🗂️ Digital Twin Memory Viewer")

# === ENVIRONMENT SETUP ===
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

# === INITIALISE CLIENT ===
pc = PineconeClient(api_key=pinecone_api_key)
index = pc.Index("dt-knowledge")

# === UI: Namespace Selection ===
namespace = st.selectbox("Select memory namespace:", ["dt-memory", "default"])
st.markdown(f"Viewing entries in namespace: `{namespace}`")

# === FETCH VECTOR IDS ===
try:
    stats = index.describe_index_stats()
    vectors = stats["namespaces"].get(namespace, {}).get("vector_count", 0)
    st.write(f"🧠 Total vectors stored: {vectors}")

    if vectors == 0:
        st.info("This namespace is currently empty.")
    else:
        st.subheader("📄 Sample Memory Chunks")

        # Retrieve vector IDs (optional pagination or limit for large sets)
        sample = index.query(vector=[0.0]*1536, top_k=10, namespace=namespace, include_metadata=True)

        for i, match in enumerate(sample["matches"]):
            meta = match.get("metadata", {})
            chunk = meta.get("text", "(No text found)")
            st.markdown(f"**{i+1}.** {chunk[:300]}...")

except Exception as e:
    st.error(f"❌ Failed to fetch memory index: {e}")

st.markdown("---")
st.caption("v1.0 – DT Memory Viewer – Darren Eastland")
