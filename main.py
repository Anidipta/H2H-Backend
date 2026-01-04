import streamlit as st
import requests
import json
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Farmer AI Assistant",
    page_icon="🌾",
    layout="centered"
)

def initialize_session_state():
    # Initialize chat history and session state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def send_message_to_backend(message: str, chat_history: list):
    # Send user message to FastAPI backend and get response
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "message": message,
                "chat_history": chat_history
            },
            timeout=30
        )
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "intent": "error",
            "answer": f"Could not connect to backend: {str(e)}",
            "sources": ""
        }

def display_chat_message(role: str, content: str, intent: str = None, sources: str = None):
    # Display chat messages with appropriate styling and metadata
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant" and intent and sources:
            with st.expander("📊 Details"):
                st.caption(f"**Intent Detected:** {intent.upper()}")
                st.caption(f"**Sources:** {sources}")

def main():
    # Main Streamlit application for farmer chat interface
    initialize_session_state()
    
    st.title("🌾 Farmer AI Assistant")
    st.markdown("*Your friendly guide for plant diseases and government schemes*")
    st.divider()
    
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("intent"),
            message.get("sources")
        )
    
    if prompt := st.chat_input("Ask me anything about plant diseases or government schemes..."):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        display_chat_message("user", prompt)
        
        with st.spinner("Thinking..."):
            response = send_message_to_backend(prompt, st.session_state.chat_history)
        
        if response["success"]:
            assistant_message = response["answer"]
            intent = response["intent"]
            sources = response["sources"]
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message,
                "intent": intent,
                "sources": sources
            })
            
            st.session_state.chat_history.append({
                "user": prompt,
                "assistant": assistant_message
            })
            
            display_chat_message("assistant", assistant_message, intent, sources)
        else:
            error_message = response["answer"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
            display_chat_message("assistant", error_message)
    
    st.sidebar.title("💡 Tips")
    # st.sidebar.info("""
    # **I can help you with:**
    
    # 🌱 **Plant Health:**
    # - Disease symptoms
    # - Pest identification
    # - Treatment methods
    # - Prevention tips
    
    # 💰 **Government Support:**
    # - Available schemes
    # - Subsidy information
    # - Eligibility criteria
    # - Application process
    
    # 🤝 **Combined Help:**
    # - Financial aid for treatments
    # - Scheme benefits for diseases
    # """)
    
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.caption(f"💬 Messages: {len(st.session_state.messages)}")
    st.sidebar.caption(f"🕐 Session: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()