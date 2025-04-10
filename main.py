import streamlit as st
from datetime import datetime

# === PAGE CONFIG (must be first Streamlit command) ===
st.set_page_config(page_title="DT Modular Chat", page_icon="🧠", layout="centered")
st.write("✅ DT App Initialising...")

# === IMPORTS WITH SAFETY CHECK ===
try:
    from components.interface import render_sidebar, handle_file_uploads
    from components.chat_handler import build_system_prompt, get_chat_response
    from components.memory import get_vectorstore, store_to_memory
    st.success("✅ All components imported successfully.")
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# === SESSION STATE INITIALISATION ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown(
    "You are interacting with Darren Eastland’s AI-driven executive assistant V2.1. "
    "This assistant reflects Darren's leadership tone, IT strategy expertise, and pragmatic decision-making style."
)

# === SIDEBAR FILE UPLOAD ===
uploaded_files = render_sidebar()
if uploaded_files:
    st.write("📎 Files received. Starting processing...")
    summaries, last_uploaded = handle_file_uploads(uploaded_files)
    last_uploaded_context = last_uploaded.get("text", "") if last_uploaded else ""
    recent_summaries = [f"Filename: {last_uploaded['name']}\nSummary: {last_uploaded['summary']}"]
else:
    last_uploaded_context = ""
    recent_summaries = []

# === PROMPT & RESPONSE HANDLING ===
prompt = st.chat_input("Ask the Digital Twin something...")
if prompt:
    if "enable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = True
    elif "disable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = False

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # === MEMORY RETRIEVAL (dt-memory namespace) ===
    try:
        memory_context = ""
        if isinstance(prompt, str) and prompt.strip():
            memory_vectorstore = get_vectorstore("dt-knowledge", namespace="dt-memory")
            memory_chunks = memory_vectorstore.similarity_search(prompt, k=5)
            memory_context = "\n".join([doc.page_content for doc in memory_chunks])
            st.markdown("🧠 Retrieved context from DT memory.")
        else:
            st.info("No valid query provided for memory search.")
    except Exception as e:
        memory_context = ""
        st.warning(f"⚠️ Memory retrieval failed: {e}")

    # === SYSTEM PROMPT BUILDING ===
    system_prompt = build_system_prompt(
        kryten_mode=st.session_state.kryten_mode,
        recent_summaries=recent_summaries,
        file_context=last_uploaded_context,
        memory_context=memory_context
    )

    # === MODEL RESPONSE GENERATION ===
    try:
        full_convo = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        reply, model = get_chat_response(
            messages=[{"role": "system", "content": system_prompt}] + full_convo
        )
        # st.chat_message("assistant").markdown(reply)
        st.markdown(f"*Model used: `{model}`*")
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.warning(f"⚠️ OpenAI response failed: {e}")

    # === STORE TO MEMORY ===
    try:
        memory_store = get_vectorstore("dt-knowledge", namespace="dt-memory")
        store_to_memory(memory_store, reply)
        st.markdown("✅ Memory updated.")
    except Exception as e:
        st.warning(f"⚠️ Memory write failed: {e}")

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === FOOTER ===
st.markdown("---")
st.caption("v2.0 – Modular DT Chat UI – Darren Eastland")
