import streamlit as st
st.set_page_config(page_title="DT Modular Chat", page_icon="🧠", layout="centered")
st.write("✅ DT App Initialising...")

try:
    from datetime import datetime
    from components.interface import render_sidebar, handle_file_uploads
    from components.chat_handler import build_system_prompt, get_chat_response
    from components.memory import get_vectorstore, store_to_memory
    st.success("✅ All components imported successfully.")
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()


# === CONFIG ===
st.set_page_config(page_title="DT Modular Chat", page_icon="🧠", layout="centered")
st.write("✅ DT App Initialising...")


# === SESSION STATE ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kryten_mode" not in st.session_state:
    st.session_state.kryten_mode = False

# === UI HEADER ===
st.title("🧠 Darren's Digital Twin")
st.markdown(
    "You are interacting with Darren Eastland’s AI-driven executive assistant. "
    "This assistant reflects Darren's leadership tone, IT strategy expertise, and pragmatic decision-making style."
)

# === SIDEBAR + FILE HANDLING ===
uploaded_files = interface.render_sidebar()
if uploaded_files:
    st.write("📎 Files received. Starting processing...")
    summaries, last_uploaded = interface.handle_file_uploads(uploaded_files)
else:
    summaries, last_uploaded = [], None


# === PROMPT AREA ===
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

    # === STORE MEMORY ===
    try:
        memory_store = get_vectorstore("dt-knowledge", namespace="dt-memory")
        store_to_memory(memory_store, reply)
        st.markdown("✅ Memory updated.")
    except Exception as e:
        st.warning(f"⚠️ Memory write failed: {e}")

# === CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.markdown("---")
st.caption("v2.0 – Modular DT Chat UI – Darren Eastland")

