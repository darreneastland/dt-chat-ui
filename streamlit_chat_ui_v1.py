import streamlit as st
import openai
import os

# === MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(page_title="Darren's Digital Twin", page_icon="🧠", layout="centered")

# === CONFIGURATION ===
openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

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

# === SESSION STATE INITIALISATION ===
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

# === CHAT INPUT ===
prompt = st.chat_input("Ask the Digital Twin something...")
if prompt:
    # Kryten Mode Toggle
    if "enable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = True
    elif "disable kryten mode" in prompt.lower():
        st.session_state.kryten_mode = False

    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build full system prompt
    full_prompt = system_prompt_base
    if st.session_state.kryten_mode:
        full_prompt += (
            "\nYou are currently in Kryten mode. "
            "Respond with a robotic, overly literal, formal tone, and excessive politeness — like Kryten from Red Dwarf."
        )

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
        reply = f"⚠️ Error: {str(e)}"
        model_used = "Unavailable"

    # Display assistant response
    st.chat_message("assistant").markdown(reply)
    st.markdown(f"*Model used: `{model_used}`*")
    st.session_state.messages.append({"role": "assistant", "content": reply})

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# === FOOTER ===
st.markdown("---")
st.caption("v1.2 – Digital Twin Chat Assistant – Darren Eastland")
