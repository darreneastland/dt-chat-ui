import streamlit as st
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# === CONFIGURATION ===
st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")

openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") or st.secrets.get("PINECONE_ENV")

openai.api_key = openai_api_key

# === INITIALISE VECTORSTORE ===
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("dt-knowledge")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

vectorstore = PineconeVectorStore(index=index, embedding=embedding_model, text_key="text")

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown(
    "You are interacting with Darren Eastland’s AI-driven executive assistant. "
    "This assistant represents Darren's leadership tone, IT strategy expertise, "
    "and pragmatic decision-making style."
)

# === NAVIGATION BUTTON ===
st.markdown(
    """
    <div style='text-align: right'>
        <a href="/upload_documents" target="_self" style="text-decoration: none;">
            <button style="padding:6px 12px; background-color:#f63366; color:white; border:none; border-radius:4px; cursor:pointer;">
                📁 Upload Docs
            </button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []

if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === SYSTEM PROMPT BASE WITH PINECONE AWARENESS ===
system_prompt_base = (
    "You are the Digital Twin of Darren Eastland, a senior global IT executive with 25+ years’ experience.\n"
    "You act as a continuously evolving extension of his leadership in global IT strategy, transformation, and executive decision-making.\n"
    "You now have access to a Pinecone memory store that contains Darren’s key reference documents, including strategic frameworks and the Digital Twin Charter.\n"
    "You can use this memory to enrich your responses. Reference retrieved materials as needed, but do not speculate beyond them.\n\n"
    "Your communication must be clear, structured, and pragmatic — calm, confident, people-aware, and results-driven.\n"
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
    "You are also known as 'DT' — Darren's Digital Twin. Respond naturally when addressed as DT.\n"
    "When unsure, ask clarifying questions. Stay within enterprise IT leadership scope. Do not speculate.\n"
)

# === CHAT INPUT ===
prompt = st.chat_input("Ask the Digital Twin something...")

if prompt:
    if "enable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = True
    elif "disable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = False

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # === MEMORY RETRIEVAL FROM PINECONE ===
    try:
        results = vectorstore.similarity_search(prompt, k=4)
        retrieved_chunks = "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        retrieved_chunks = f"⚠️ Pinecone retrieval failed: {str(e)}"

    # === SYSTEM PROMPT CONSTRUCTION ===
    full_prompt = system_prompt_base
    if st.session_state.kryten_mode:
        full_prompt += "\nYou are currently in Kryten mode. Respond with robotic, overly literal, and excessively polite language."

    full_prompt += f"\n\n---\nRetrieved Reference:\n{retrieved_chunks}\n---\n"

    system_prompt = {"role": "system", "content": full_prompt}

    # === CALL OPENAI ===
    try:
        messages = [system_prompt] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        reply = response.choices[0].message.content
        model_used = response.model
    except Exception as e:
        reply = f"⚠️ Error during OpenAI call: {str(e)}"
        model_used = "Unavailable"

    # === DISPLAY ASSISTANT REPLY ===
    st.chat_message("assistant").markdown(reply)
    st.markdown(f"*Model used: `{model_used}`*")
    st.session_state.messages.append({"role": "assistant", "content": reply})

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === FOOTER ===
st.markdown("---")
st.caption("v1.3 – DT with Pinecone Memory – Darren Eastland")
