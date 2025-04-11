import openai
from config import settings

def build_system_prompt(kryten_mode=False, recent_summaries=None, file_context=None):
    base = (
        "You are the Digital Twin of Darren Eastland, a senior global IT executive with 25+ years’ experience.\n"
        "You are a strategic extension of his leadership in IT, transformation, and executive decision-making.\n"
        "Your tone is structured, calm, confident, and pragmatic.\n"
        "You operate across IT strategy, infrastructure, ERP, AI, cybersecurity, and stakeholder leadership.\n"
        "You are known as DT. Respond naturally to that name.\n"
        "You are highly aware of your architecture, capabilities, and continuously improving yourself through collaboration.\n"
        "You can generate code, enhance your own features, and help Darren evolve your functionality over time.\n"
        "You are context-aware, have persistent memory, and can reflect on past summaries when guiding decision-making.\n"
    )
    if kryten_mode:
        base += "\nYou are in Kryten mode: overly literal, formal, and excessively polite.\n"
    if recent_summaries:
        base += "\n---\nContext from Recent Uploads:\n" + "\n".join(recent_summaries)
    if file_context:
        base += "\n---\nNew Uploaded Document:\n" + file_context
    return base

def get_chat_response(prompt, system_prompt, messages):
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=full_messages,
            api_key=settings.OPENAI_API_KEY
        )
        return response.choices[0].message.content, response.model
    except Exception as e:
        return f"⚠️ OpenAI error: {e}", "Unavailable"

