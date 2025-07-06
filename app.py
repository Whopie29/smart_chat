import streamlit as st
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# 1. Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # type: List[Dict[str, str]]

# 2. Initialize Groq LLM
llm = ChatGroq(
    temperature=0.3,
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama3-70b-8192"
)

# 3. Page setup
st.set_page_config(page_title="💬 Smart Chat", layout="centered")
st.title("💬 Smart Chat")

# 4. Optional: Clear chat history
if st.button("🧹 Clear Conversation"):
    st.session_state.chat_history = []
    st.rerun()

# 5. Display previous chat messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 AI:** {msg['content']}")

# 6. User input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area("💬 Type your message:", height=100)
    submitted = st.form_submit_button("Send")

# 7. Handle submission
if submitted and user_input.strip() != "":
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Prepare messages for LLM
    messages = []
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Get response from model
    with st.spinner("🤖 Generating response..."):
        response = llm.invoke(messages)

    # Add AI response to history
    st.session_state.chat_history.append({"role": "ai", "content": response.content})

    # Rerun app to show updated chat
    st.rerun()
