
import streamlit as st
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

st.set_page_config(page_title="DT Chat Assistant", page_icon="💬", layout="centered")
st.title("💬 Darren's Digital Twin - Chat Interface")
st.markdown("This is a live GPT-4 chat assistant for strategic, operational, and comms support.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask the Digital Twin something...")
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]
        )
        reply = response.choices[0].message.content
    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"

    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown("---")
st.caption("v1.0 – DT Chat Assistant by Darren Eastland")
