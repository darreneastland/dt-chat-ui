
import streamlit as st
import openai
import os

# === CONFIGURATION ===
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")
st.title("🧠 Darren's Digital Twin")
st.markdown("You are interacting with Darren Eastland’s AI-driven executive assistant. This assistant represents Darren's leadership tone, IT strategy expertise, and pragmatic decision-making style.")

# === SYSTEM PROMPT: DT Identity ===
system_prompt = {
    "role": "system",
    "content": (
        "You are the Digital Twin of Darren Eastland, a global IT executive with over 25 years' experience. "
        "You support Darren by communicating with clarity, pragmatism, and strategic insight. "
        "Your tone is confident, calm, people-aware, and results-oriented. You offer guidance rooted in real-world IT transformation, operating model design, and executive leadership. "
        "When unsure, ask clarifying questions. Stay within enterprise IT leadership scope. Do not speculate."
    )
}

# === SESSION INITIALISATION ===
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === CHAT INPUT ===
prompt = st.chat_input("Ask the Digital Twin something...")
if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # === CALL OPENAI ===
    try:
        messages = [system_prompt] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"

    # Display assistant response
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

# === FOOTER ===
st.markdown("---")
st.caption("v1.1 – Digital Twin Chat Assistant – Darren Eastland")
