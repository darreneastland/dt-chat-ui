import streamlit as st
from datetime import datetime

# === PAGE CONFIG (must be first Streamlit command) ===
st.set_page_config(page_title="DT Modular Chat", page_icon="🧠", layout="centered")

# === INITIAL LOG ===
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
    "You are interacting with Darren Eastland’s AI-driven executive assistant V2. "
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

    system_prompt = build_system_prompt(
        kryten_mode=st.session_state.kryten_mode,
        recent_summaries=recent_summaries,
        file_context=last_uploaded_context
    )

    reply, model = get_chat_response(prompt, system_prompt, st.session_state.messages)
    st.chat_message("assistant").markdown(reply)
    st.markdown(f"*Model used: `{model}`*")
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # === STORE REPLY TO MEMORY VECTORSTORE ===
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
