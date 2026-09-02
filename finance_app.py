"""
Personal Finance Agent - Streamlit UI (admin/personal control tool)
Run: streamlit run finance_app.py

This talks to the same multi-tenant backend as the Telegram bot.
By default it acts as YOUR account (set ADMIN_TELEGRAM_CHAT_ID in
.env to your own Telegram chat id, the same one you used with
/start on the bot). You can also switch to viewing any other
registered user from the sidebar - handy for checking things work
for your friends/family without needing their phone.
"""

import os
import streamlit as st
from dotenv import load_dotenv
import finance_core as core

load_dotenv()

st.set_page_config(page_title="Finance Agent", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 20% 0%, #1a2a1f 0%, #0d120f 45%, #0a0b0a 100%); }
    .hero { padding: 24px 30px; border-radius: 18px;
            background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(16,185,129,0.12));
            border: 1px solid rgba(134,239,172,0.18); margin-bottom: 20px; }
    .hero h1 { font-size: 26px; font-weight: 700; margin: 0; color: #86efac; }
    .chat-bubble-user { background: linear-gradient(135deg, #16a34a, #22c55e); color: white;
                         padding: 12px 16px; border-radius: 16px 16px 4px 16px; margin: 6px 0;
                         max-width: 80%; margin-left: auto; font-size: 14px; }
    .chat-bubble-bot { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
                        color: #e5e7eb; padding: 12px 16px; border-radius: 16px 16px 16px 4px;
                        margin: 6px 0; max-width: 80%; font-size: 14px; white-space: pre-wrap; }
    .trace-step { background: rgba(34,197,94,0.08); border-left: 3px solid #4ade80;
                  padding: 6px 12px; border-radius: 0 8px 8px 0; margin: 4px 0;
                  font-size: 12.5px; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)


# =================================================================
# Resolve which user this session is acting as
# =================================================================

ADMIN_CHAT_ID = os.getenv("ADMIN_TELEGRAM_CHAT_ID")

if not ADMIN_CHAT_ID:
    st.error(
        "ADMIN_TELEGRAM_CHAT_ID مش متظبط في .env.\n\n"
        "لازم تحط فيه رقم الـ Telegram chat id بتاعك (نفسه اللي استخدمته لما بعتّ /start للبوت) "
        "عشان الأداة دي تعرف تتكلم باسمك."
    )
    st.stop()

admin_user = core.get_or_create_user(int(ADMIN_CHAT_ID), display_name="Admin")
all_users = core.list_all_users()

with st.sidebar:
    st.markdown("### 👤 عرض بيانات مين؟")
    labels = {u["id"]: (u.get("display_name") or f"user {u['id'][:8]}") for u in all_users}
    selected_id = st.selectbox(
        "المستخدم", options=list(labels.keys()),
        format_func=lambda uid: labels[uid],
        index=list(labels.keys()).index(admin_user["id"]) if admin_user["id"] in labels else 0,
    )
    active_user_id = selected_id
    viewing_self = active_user_id == admin_user["id"]
    if not viewing_self:
        st.info("بتشوف حساب حد تاني - القراءة والتسجيل هيتوجهوا لحسابه هو.")

    st.markdown("---")
    st.markdown("### 📊 Quick actions")
    if st.button("📅 This month's summary"):
        st.session_state.setdefault("messages", []).append({"role": "user", "content": "give me a summary of this month"})
    if st.button("🎯 Budget status"):
        st.session_state.setdefault("messages", []).append({"role": "user", "content": "show my budget status"})

    st.markdown("---")
    st.markdown("### 💡 Try pasting a real SMS")
    st.code("تم خصم 250.00 جنيه من حسابك\nلصالح كارفور", language=None)

    if st.button("🗑️ Reset conversation"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


st.markdown(f"""
<div class="hero">
    <h1>💰 Personal Finance Agent</h1>
</div>
""", unsafe_allow_html=True)

# Reset the chat history whenever the selected user changes, so the
# LLM's conversation context doesn't leak between accounts.
if st.session_state.get("_active_user") != active_user_id:
    st.session_state.messages = []
    st.session_state.history = []
    st.session_state._active_user = active_user_id

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    css = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(f'<div class="{css}">{msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Paste an SMS or ask about your spending...")

pending = None
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user" and \
   len(st.session_state.messages) > len(st.session_state.get("_processed", [])):
    pending = st.session_state.messages[-1]["content"]

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    pending = user_input

if pending:
    st.markdown(f'<div class="chat-bubble-user">{pending}</div>', unsafe_allow_html=True)
    trace_container = st.container()

    def log_step(text):
        with trace_container:
            st.markdown(f'<div class="trace-step">{text}</div>', unsafe_allow_html=True)

    with st.spinner("Processing..."):
        answer = core.run_finance_agent(
            pending, st.session_state.history, user_id=active_user_id, log_callback=log_step
        )

    st.markdown(f'<div class="chat-bubble-bot">{answer}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state._processed = list(st.session_state.messages)