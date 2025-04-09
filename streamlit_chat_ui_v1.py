import streamlit as st
import os
import openai
from datetime import datetime

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from pinecone import Pinecone

# === CONFIGURATION ===
st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")

openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

openai.api_key = openai_api_key
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

# === PINECONE CLIENT ===
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)
index = pc.Index("dt-knowledge")

# === VECTORSTORES ===
vs_knowledge = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text", namespace="default")
vs_memory = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text", namespace="dt-memory")

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown("You are interacting with Darren Eastland’s AI-driven executive assistant.")

# === FILE UPLOAD IN CHAT CONTEXT ===
# uploaded_file = st.file_uploader("📎 Upload a document to use in this chat", type=["pdf", "docx", "txt"], label_visibility="collapsed")
# store_in_memory = st.checkbox("Store in DT persistent memory", value=True)
# store_in_knowledge = st.checkbox("Store in reference knowledge base", value=False)

extracted_text = ""

# === EXTRACT TEXT FROM UPLOADED FILE ===
if uploaded_file:
    try:
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

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
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        extracted_text = "\n\n".join([d.page_content for d in split_docs[:5]])
        st.success(f"📄 Extracted content from {uploaded_file.name}.")

    except Exception as e:
        st.error(f"❌ Failed to read document: {e}")

# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === SYSTEM PROMPT BASE ===
system_prompt_base = (
    "You are the Digital Twin of Darren Eastland, a senior global IT executive with 25+ years’ experience.\n"
    "You have access to Pinecone-based memory: reference documents and persistent insights from past interactions.\n"
    "You may also receive new documents mid-conversation to analyse.\n"
    "Communicate with clarity, pragmatism, and strategic insight. When unsure, ask clarifying questions.\n"
)

# === CHAT INPUT ===
prompt = st.chat_input("Ask the Digital Twin something...")

# === DOCUMENT UPLOAD CONTROLS ===
with st.expander("📎 Upload a document to use in this chat"):
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "docx", "txt"])
    store_in_memory = st.checkbox("Store in DT persistent memory", value=True)
    store_in_knowledge = st.checkbox("Store in reference knowledge base", value=False)

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # === RETRIEVE CONTEXT FROM PINECONE ===
    doc_context = ""
    mem_context = ""

    try:
        doc_chunks = vs_knowledge.similarity_search(prompt, k=3)
        doc_context = "\n\n".join([doc.page_content for doc in doc_chunks])
    except Exception as e:
        st.warning(f"Knowledge retrieval failed: {e}")

    try:
        mem_chunks = vs_memory.similarity_search(prompt, k=3)
        mem_context = "\n\n".join([doc.page_content for doc in mem_chunks])
    except Exception as e:
        st.warning(f"Memory retrieval failed: {e}")

    # === ASSEMBLE SYSTEM PROMPT ===
    full_prompt = system_prompt_base
    if st.session_state.kryten_mode:
        full_prompt += "\nRespond in Kryten mode: overly literal, formal, and excessively polite."

    full_prompt += f"\n\n---\nContext from Reference Documents:\n{doc_context}"
    full_prompt += f"\n\n---\nContext from Persistent Memory:\n{mem_context}"
    if extracted_text:
        full_prompt += f"\n\n---\nNew Uploaded Document Context:\n{extracted_text}"

    system_prompt = {"role": "system", "content": full_prompt}

    # === OPENAI CHAT COMPLETION ===
    try:
        messages = [system_prompt] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        reply = response.choices[0].message.content
        model_used = response.model
    except Exception as e:
        reply = f"⚠️ OpenAI error: {e}"
        model_used = "Unavailable"

    st.chat_message("assistant").markdown(reply)
    st.markdown(f"*Model used: `{model_used}`*")
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # === OPTIONAL: STORE FILE IN MEMORY/KNOWLEDGE ===
    if uploaded_file and (store_in_memory or store_in_knowledge):
        target_vs = []
        if store_in_memory:
            target_vs.append(("dt-memory", vs_memory))
        if store_in_knowledge:
            target_vs.append(("default", vs_knowledge))

        try:
            for ns, store in target_vs:
                store.add_documents(split_docs)
            st.success("✅ Document saved to selected memory store(s).")
        except Exception as e:
            st.warning(f"⚠️ Failed to store document: {e}")

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === FOOTER ===
st.markdown("---")
st.caption("v1.51 – DT with Chat File Upload – Darren Eastland")
