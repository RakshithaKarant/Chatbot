import logging
from LLMHandler import LLMHandler
import streamlit as st

# Set up logging to suppress debug messages
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors

def format_json_to_html(response_dict):
    """
    Convert the JSON response to an HTML string with <br> separation for display.
    """
    html_output = "<br>".join(f"<b>{key}:</b> {value}" for key, value in response_dict.items())
    return html_output

if "llm_handler" not in st.session_state:
    # Initialize LLMHandler with persistent memory
    memory = st.session_state.get("conversation_memory", None)
    st.session_state["llm_handler"] = LLMHandler(memory=memory)

if "messages" not in st.session_state:
    # Initialize chat history
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

llm_handler = st.session_state["llm_handler"]

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot for customer service with sentiment analysis")

# Display conversation history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)

# Input box for new messages
if prompt := st.chat_input("Type your message here..."):
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt, unsafe_allow_html=True)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Handle the conversation
            response_dict = llm_handler.handle_conversation(prompt)
            response_html = format_json_to_html(response_dict)
            
            # Add bot response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response_html})
            st.write(response_html, unsafe_allow_html=True)

# Save memory state to session
st.session_state["conversation_memory"] = llm_handler.memory
