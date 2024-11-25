import logging
from LLMHandler import LLMHandler
import streamlit as st
from Insights import InsightDeriver

# Set up logging to suppress debug messages
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors

def format_json_to_html(response_dict):
    """
    Convert the JSON response to an HTML string with <br> separation for display.
    """
    if type(response_dict) is list and len(response_dict)>0:
        response_dict=response_dict[0]
    html_output = "<br>".join(f"<b>{key}:</b> {value}" for key, value in response_dict.items())
    return html_output

if "llm_handler" not in st.session_state:
    # Initialize LLMHandler with persistent memory
    memory = st.session_state.get("conversation_memory", None)
    st.session_state["llm_handler"] = LLMHandler(memory=memory)

if "booking_id" not in st.session_state:
    st.session_state["booking_id"] = ""
if "InsightDeriver" not in st.session_state:
    st.session_state["InsightDeriver"] = InsightDeriver()

if "messages" not in st.session_state:
    # Initialize chat history
    st.session_state["messages"] = [{"role": "assistant", "content": "Please provide your Booking ID"}]

llm_handler = st.session_state["llm_handler"]

st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot for airlines travel assistance")

# Display conversation history
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)

# Input box for new messages
if prompt := st.chat_input("Type your message here. Type exit to quit"):
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt, unsafe_allow_html=True)

    if str(prompt).strip().lower() == "exit":
        # Reset session state variables
        st.session_state["llm_handler"] = None
        st.session_state["booking_id"] = ""
        st.session_state["InsightDeriver"] = None
        st.session_state["messages"] = [{"role": "assistant", "content": "Please provide your Booking ID"}]
        
        # Clear conversation memory
        st.session_state["conversation_memory"] = None
        
        # Reload the page
        st.experimental_rerun()
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                generate_response = False
                response_html = ""

                if st.session_state["booking_id"] == "":
                    booking_id = InsightDeriver.extract_first_number(prompt)
                    if booking_id:
                        user_details = st.session_state["InsightDeriver"].user_details_by_booking_id(int(booking_id))
                        if user_details:
                            details = format_json_to_html(user_details)
                            response_html = f"The details for booking ID {booking_id} are as follows:<br/> {details}. <br/> Please provide your query"
                            st.session_state["booking_id"] = booking_id
                            llm_handler.save_user_details(user_details)  # Save user details in memory
                        else:
                            response_html = f"No matching records found for the booking ID {booking_id}."
                    else:
                        response_html = "It seems you didn't provide a booking ID. Could you please provide your booking ID so I can assist you better?"
                else:
                    generate_response = True

                if generate_response:
                    response_dict = llm_handler.handle_conversation(prompt)
                    response_html = format_json_to_html(response_dict)
                
                # Add bot response to the chat history
                st.session_state.messages.append({"role": "assistant", "content": response_html})
                st.write(response_html, unsafe_allow_html=True)

    # Save memory state to session
    st.session_state["conversation_memory"] = llm_handler.memory
