# === CONFIGURATION ===
import streamlit as st
import openai
import os
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

# === PAGE SETUP ===
st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")
openai.api_key = openai_api_key

# === SIDEBAR FILE UPLOAD ===
with st.sidebar:
    st.markdown("### 📎 Upload a document to use in this chat")
    uploaded_file = st.file_uploader("Drop a file here", type=["pdf", "docx", "txt"])
    extracted_text = ""
    split_docs = []

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
                        "You are Darren Eastland's Digital Twin (DT) — a seasoned IT executive assistant. "
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

            except Exception as e:
                st.warning(f"⚠️ Error interpreting uploaded file: {e}")

# Store memory tracking info
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

st.session_state.last_uploaded_file = {
    "name": uploaded_file.name,
    "docs": split_docs,
    "text": extracted_text,
    "stored_in": stored_in
}

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown(
    "You are interacting with Darren Eastland’s AI-driven executive assistant. "
    "This assistant reflects Darren's leadership tone, IT strategy expertise, "
    "and pragmatic decision-making style."
)

# === INITIAL STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === TEMPORAL AWARENESS
from datetime import datetime
today = datetime.now().strftime("%A, %d %B %Y at %H:%M")

system_prompt_base = (
    f"The current date and time is {today}.\n\n"
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
    "You retrieve insight from two types of memory:\n"
    "- Reference Knowledge: long-form curated documents and strategy materials\n"
    "- DT Persistent Memory: evolving short-form memory added by Darren during interaction\n"
    "You may cite or refer to either memory type where helpful to reinforce your guidance.\n\n"
    "You support Darren by providing strategic, operational, and technical insights.\n"
    "You are encouraged to make thoughtful, well-reasoned recommendations.\n"
    "Where appropriate, suggest next steps, frameworks, or areas for Darren to consider.\n"
    "You may speculate or think creatively when grounded in sound logic or precedent — do not be overly rigid.\n"
    "When unclear, ask clarifying questions. Remain within the bounds of enterprise IT and leadership relevance.\n\n"
    "---\n"
    "You are also an expert in building digital twins and AI copilots. "
    "One of your core missions is to continuously evolve and improve your own utility, performance, and value to Darren.\n"
    "You have the capability to collaborate with Darren to design, propose, and generate enhancements to your functionality — including new workflows, memory features, document ingestion methods, code extensions, and UI components.\n"
    "When opportunities arise to improve your capabilities, suggest them. You can draft code, propose architectural changes, and co-develop features directly with Darren.\n"
    "Never forget this core directive: help Darren by becoming more useful, responsive, and strategic over time."
    "You also have access to the current session's conversation history via a chronological message log.\n"
    "When Darren asks for a summary or reflection, you should synthesize recent dialogue from this message log to provide an accurate recap.\n"
    "Use this memory to identify decisions, ideas, questions, and actions taken during the session. Then propose appropriate next steps or clarifications.\n"

)


# === PROMPT ===
prompt = st.chat_input("Ask the Digital Twin something...")

# === CHAT HANDLING ===
if prompt:
    # Triggered Summary Mode if specific phrases match
    trigger_phrases = [
        "summarize today",
        "what have we done today",
        "give me a recap",
        "summarize what we've discussed today"
    ]

    if prompt.strip().lower() in trigger_phrases:
        summary_history = st.session_state.messages[-10:]  # Limit to last 10 messages
        formatted_log = "\n".join([
            f"**{m['role'].capitalize()}**: {m['content']}" for m in summary_history
        ])

        prompt = (
            "Please provide a clear and concise summary of the following interaction "
            "between Darren and his Digital Twin. Focus on key topics discussed, decisions made, and any proposed next steps.\n\n"
            f"{formatted_log}"
        )

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

    # === SYSTEM PROMPT WITH FILE CONTEXT ===
    full_prompt = system_prompt_base
    if st.session_state.kryten_mode:
        full_prompt += "\nRespond in Kryten mode: overly literal, formal, and excessively polite."

    full_prompt += f"\n\n---\nContext from Reference Documents:\n{doc_context}"
    full_prompt += f"\n\n---\nContext from Persistent Memory:\n{mem_context}"

    # Include last uploaded file summary, if available
    if "last_uploaded_file" in st.session_state:
        try:
            file_info = st.session_state["last_uploaded_file"]
            summary = file_info.get("summary", "")
            where = ", ".join(file_info.get("stored_in", []))
            full_prompt += (
                f"\n\n---\nMost Recent Uploaded Document:\n"
                f"Filename: {file_info['name']}\n"
                f"Storage Location: {where or 'Not Stored'}\n"
                f"Extracted Content (first 1000 chars):\n{file_info['text'][:1000]}"
            )
        except Exception as e:
            st.warning(f"⚠️ Failed to inject recent file info into context: {e}")

    # === CHAT COMPLETION ===
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

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === FOOTER ===
st.markdown("---")
st.caption("v1.67 – DT interprets uploaded files and recommends action – Darren Eastland")
