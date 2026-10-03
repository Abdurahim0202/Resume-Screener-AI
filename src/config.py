import os

try:
    import streamlit as st
    GROQ_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
except Exception:
    # Running locally without a secrets file
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")