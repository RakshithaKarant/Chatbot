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

if __name__ == "__main__":
    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot for customer service with sentiment analysis")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display conversation history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)

    # Input box for new messages
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Process the user's message through the LLMHandler
        llm_handler = LLMHandler()
        response_dict = llm_handler.handle_conversation(prompt)
        response_html = format_json_to_html(response_dict)  # Format response as HTML
        
        # Add bot response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response_html})
        st.chat_message("assistant").write(response_html, unsafe_allow_html=True)  # Render HTML
