# === CONFIGURATION ===
import streamlit as st
import openai
import os
import json
from datetime import datetime
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient

# === INIT EMBEDDING AND VECTORSTORES ===
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

pc = PineconeClient(api_key=pinecone_api_key)
index = pc.Index("dt-knowledge")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

vs_knowledge = Pinecone.from_existing_index(index_name="dt-knowledge", embedding=embedding_model, namespace="default")
vs_memory = Pinecone.from_existing_index(index_name="dt-knowledge", embedding=embedding_model, namespace="dt-memory")

st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")
openai.api_key = openai_api_key

# === SIDEBAR FILE UPLOAD ===
with st.sidebar:
    st.markdown("### 📎 Upload a document to use in this chat")
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "docx", "txt"])
    extracted_text = ""
    split_docs = []
    interpreted_reply = ""

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

            try:
                interpretation_system_prompt = {
                    "role": "system",
                    "content": (
                        "You are Darren Eastland's Digital Twin (DT) — a seasoned IT executive assistant.\n"
                        "You have just received a document. Your job is to:\n"
                        "- Understand the document's purpose and content\n"
                        "- Summarize it concisely (bullet points if needed)\n"
                        "- Recommend the most appropriate next action:\n"
                        "  • Store in persistent memory\n"
                        "  • Add to knowledge base\n"
                        "  • Use in current session only\n"
                        "  • Or ignore it\n\n"
                        "Respond in a calm, confident, and professional tone as DT. Be helpful and proactive."
                    )
                }

                interpretation_user_prompt = {
                    "role": "user",
                    "content": (
                        f"The user has uploaded a document titled `{uploaded_file.name}`. "
                        f"Here is the extracted content:\n\n{extracted_text[:2500]}"
                    )
                }

                interpretation_response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[interpretation_system_prompt, interpretation_user_prompt]
                )

                interpreted_reply = interpretation_response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": interpreted_reply})
                st.markdown("✅ File interpreted successfully.")

            except Exception as e:
                st.warning(f"⚠️ Error interpreting uploaded file: {e}")

        # === Determine storage location based on interpreted reply ===
        stored_in = []
        try:
            if "Persistent Memory" in interpreted_reply:
                vs_memory.add_documents(split_docs)
                stored_in.append("Persistent Memory")
            if "Knowledge Base" in interpreted_reply:
                vs_knowledge.add_documents(split_docs)
                stored_in.append("Reference Knowledge")
        except Exception as e:
            st.warning(f"⚠️ Failed to store document: {e}")

        # === Save file metadata to session ===
        st.session_state["last_uploaded_file"] = {
            "name": uploaded_file.name,
            "text": extracted_text,
            "summary": interpreted_reply,
            "stored_in": stored_in,
            "timestamp": datetime.now().isoformat()
        }

# === INITIAL STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === TEMPORAL + BASE PROMPT SETUP ===
today = datetime.now().strftime("%A, %d %B %Y at %H:%M")
system_prompt_base = (
    f"The current date and time is {today}.\n\n"
    "You are the Digital Twin of Darren Eastland, a senior global IT executive with 25+ years’ experience.\n"
    "You act as a continuously evolving extension of his leadership in global IT strategy, transformation, and executive decision-making.\n"
    "Your communication must be clear, structured, and pragmatic — calm, confident, people-aware, and results-driven.\n\n"
    "You support Darren by providing strategic, operational, and technical insights. You are encouraged to make well-reasoned recommendations.\n"
    "You may speculate when grounded in logic. Ask clarifying questions when unsure.\n\n"
    "---\nYou also:\n- Access curated Reference Knowledge and DT Persistent Memory\n"
    "- Summarize recent conversations on request\n"
    "- Are an expert in digital twin evolution — propose upgrades when helpful\n"
)

# === CHAT INPUT ===
st.title("🧠 Darren's Digital Twin")
prompt = st.chat_input("Ask the Digital Twin something...")

# === HANDLE CHAT ===
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Add memory context
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

    full_prompt = system_prompt_base
    full_prompt += f"\n\n---\nContext from Reference Documents:\n{doc_context}"
    full_prompt += f"\n\n---\nContext from Persistent Memory:\n{mem_context}"

    if "last_uploaded_file" in st.session_state:
        file_info = st.session_state["last_uploaded_file"]
        where = ", ".join(file_info.get("stored_in", []))
        full_prompt += (
            f"\n\n---\nMost Recent Uploaded Document:\n"
            f"Filename: {file_info['name']}\n"
            f"Storage Location: {where or 'Not Stored'}\n"
            f"Extracted Content (first 1000 chars):\n{file_info['text'][:1000]}"
        )

    try:
        messages = [{"role": "system", "content": full_prompt}] + [
            {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
        ]
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        reply = response.choices[0].message.content
        model_used = response.model
    except Exception as e:
        reply = f"⚠️ OpenAI error: {e}"
        model_used = "Unavailable"

    st.chat_message("assistant").markdown(reply)
    st.markdown(f"*Model used: `{model_used}`*")
    st.session_state.messages.append({"role": "assistant", "content": reply})

# === CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown("---")
st.caption("v1.70 – DT Persistent File Awareness & Memory Fix – Darren Eastland")
