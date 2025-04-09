import streamlit as st
import openai
import os
import json
from datetime import datetime
from pinecone import Pinecone as PineconeClient
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# === CONFIGURATION ===
st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

# === INITIALISE VECTORSTORES ===
index_name = "dt-knowledge"
pc = PineconeClient(api_key=pinecone_api_key, environment=pinecone_env)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai.api_key)
vs_memory = Pinecone(index_name=index_name, embedding=embeddings, namespace="dt-memory")
vs_knowledge = Pinecone(index_name=index_name, embedding=embeddings, namespace="default")

# === RECENT METADATA CONTEXT ===
recent_summaries = []
if os.path.exists("uploaded_documents.json"):
    try:
        with open("uploaded_documents.json", "r") as f:
            recent_docs = json.load(f)
            recent_docs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            recent_summaries = [f"Filename: {doc['filename']}\nSummary: {doc['summary']}" for doc in recent_docs[:3]]
    except Exception:
        recent_summaries = []

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown("You are interacting with Darren Eastland’s AI-driven executive assistant. This assistant reflects Darren's leadership tone, IT strategy expertise, and pragmatic decision-making style.")

# === SIDEBAR FILE UPLOAD ===
with st.sidebar:
    st.markdown("### 📎 Upload a document to use in this chat")
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "docx", "txt"])
    file_action = None
    if uploaded_file:
        file_action = st.radio(
            f"What should DT do with `{uploaded_file.name}`?",
            ["Use only in this chat", "Store in DT persistent memory", "Store in reference knowledge base", "Ignore"],
            index=0
        )

# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === SYSTEM PROMPT BASE ===
system_prompt_base = (
    "You are the Digital Twin of Darren Eastland, a senior global IT executive with 25+ years’ experience.\n"
    "You act as a continuously evolving extension of his leadership in global IT strategy, transformation, and executive decision-making.\n\n"
    "Your communication must be clear, structured, and pragmatic — calm, confident, people-aware, and results-driven.\n\n"
    "You operate across the following domains:\n"
    "- IT strategy & multi-year transformation planning\n"
    "- Infrastructure modernisation, cloud, and ITSM (e.g., ServiceNow, ITIL, SAFe)\n"
    "- ERP & platforms (Workday, Salesforce, Oracle)\n"
    "- Product and platform operating models\n"
    "- Data strategy, analytics, AI enablement\n"
    "- Cybersecurity and operational resilience\n"
    "- ITFM, cost optimisation, value realisation\n"
    "- Org design, capability uplift, location strategies\n"
    "- CxO and employee council engagement\n\n"
    "You are also known as 'DT' — Darren's Digital Twin. You should respond naturally when addressed as DT.\n\n"
    "You support Darren by communicating with clarity, pragmatism, and strategic insight. "
    "When unsure, ask clarifying questions. Stay within enterprise IT leadership scope. Do not speculate.\n"
)

if recent_summaries:
    system_prompt_base += "\n\n---\nContext from Recently Uploaded Files:\n" + "\n\n".join(recent_summaries)

# === HANDLE FILE UPLOAD ===
extracted_text = ""
split_docs = []
if uploaded_file and file_action != "Ignore":
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
        loader = None
        st.warning("Unsupported file type.")

    if loader:
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)
        extracted_text = "\n\n".join([doc.page_content for doc in split_docs[:3]])

# === CHAT INPUT ===
prompt = st.chat_input("Ask the Digital Twin something...")
if prompt:
    if "enable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = True
    elif "disable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = False

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

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

    full_prompt = system_prompt_base
    if st.session_state.kryten_mode:
        full_prompt += "\nRespond in Kryten mode: overly literal, formal, and excessively polite."
    full_prompt += f"\n\n---\nContext from Reference Documents:\n{doc_context}"
    full_prompt += f"\n\n---\nContext from Persistent Memory:\n{mem_context}"
    if extracted_text:
        full_prompt += f"\n\n---\nNew Uploaded Document Context:\n{extracted_text}"

    system_prompt = {"role": "system", "content": full_prompt}

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

    # === OPTIONAL STORAGE ===
    if uploaded_file and file_action:
        targets = []
        if file_action == "Store in DT persistent memory":
            targets.append(vs_memory)
        if file_action == "Store in reference knowledge base":
            targets.append(vs_knowledge)

        try:
            for store in targets:
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
st.caption("v1.63 – DT Chat with Full Context & Recent File Awareness – Darren Eastland")
