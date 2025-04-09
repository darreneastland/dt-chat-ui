import streamlit as st
import openai
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

# === PAGE CONFIG ===
st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")

# === SIDEBAR FILE UPLOAD ===
with st.sidebar:
    st.markdown("### 📎 Upload a document to use in this chat")
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "docx", "txt"])
    store_in_memory = st.checkbox("Store in DT persistent memory", value=True)
    store_in_knowledge = st.checkbox("Store in reference knowledge base", value=False)


# === API KEYS ===
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")

# === PINECONE INIT ===
pc = Pinecone(api_key=pinecone_api_key)
embed = OpenAIEmbeddings()

vs_knowledge = LangchainPinecone.from_existing_index(index_name="dt-knowledge", embedding=embed, namespace="default")
vs_memory = LangchainPinecone.from_existing_index(index_name="dt-knowledge", embedding=embed, namespace="dt-memory")

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown(
    "You are interacting with Darren Eastland’s AI-driven executive assistant. "
    "This assistant reflects Darren's leadership tone, IT strategy expertise, and pragmatic decision-making style."
)

# === STATE INIT ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === SYSTEM PROMPT ===
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
    "You are also known as 'DT'. You should respond naturally when addressed as DT.\n"
    "Support Darren by communicating with clarity, pragmatism, and strategic insight. "
    "When unsure, ask clarifying questions. Stay within enterprise IT leadership scope. Do not speculate."
)

# === CHAT INPUT ===
prompt = st.chat_input("Ask the Digital Twin something...")

# === DISPLAY CHAT HISTORY BEFORE ANY NEW INPUT/UPLOAD ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === DOCUMENT UPLOAD CONTROLS BELOW CHAT ===
uploaded_file = None
extracted_text = ""
split_docs = []

with st.expander("📎 Upload a document to use in this chat", expanded=False):
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "docx", "txt"])
    store_in_memory = st.checkbox("Store in DT persistent memory", value=True)
    store_in_knowledge = st.checkbox("Store in reference knowledge base", value=False)

    if uploaded_file:
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
            split_docs = loader.load_and_split()
            extracted_text = "\n\n".join([doc.page_content for doc in split_docs[:3]])

# === HANDLE PROMPT ===
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # === CONTEXT RETRIEVAL ===
    doc_context = ""
    mem_context = ""

    try:
        doc_chunks = vs_knowledge.similarity_search(prompt, k=3)
        doc_context = "\n\n".join([doc.page_content for doc in doc_chunks])
    except Exception as e:
        st.warning(f"⚠️ Knowledge retrieval failed: {e}")

    try:
        mem_chunks = vs_memory.similarity_search(prompt, k=3)
        mem_context = "\n\n".join([doc.page_content for doc in mem_chunks])
    except Exception as e:
        st.warning(f"⚠️ Memory retrieval failed: {e}")

    # === BUILD SYSTEM PROMPT ===
    full_prompt = system_prompt_base
    if st.session_state.kryten_mode:
        full_prompt += "\nRespond in Kryten mode: overly literal, formal, and excessively polite."

    full_prompt += f"\n\n---\nContext from Reference Documents:\n{doc_context}"
    full_prompt += f"\n\n---\nContext from Persistent Memory:\n{mem_context}"
    if extracted_text:
        full_prompt += f"\n\n---\nNew Uploaded Document Context:\n{extracted_text}"

    system_prompt = {"role": "system", "content": full_prompt}

    # === OPENAI COMPLETION ===
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

    # === OPTIONAL: STORE DOCUMENT ===
    if uploaded_file and (store_in_memory or store_in_knowledge):
        targets = []
        if store_in_memory:
            targets.append(("dt-memory", vs_memory))
        if store_in_knowledge:
            targets.append(("default", vs_knowledge))

        try:
            for ns, vectorstore in targets:
                vectorstore.add_documents(split_docs)
            st.success("✅ Document saved to selected memory store(s).")
        except Exception as e:
            st.warning(f"⚠️ Failed to store document: {e}")

# === FOOTER ===
st.markdown("---")
st.caption("v1.54 – DT with Chat File Upload (Improved Layout) – Darren Eastland")
