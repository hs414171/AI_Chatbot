import streamlit as st
from chat_module import handle_conversation  # Import your handle_conversation function
import os

# Make sure there is only one chat input
st.title("Singhada GPT")


# Create a container for the chat history
chat_placeholder = st.empty()

# Using session state to preserve chat history across runs
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'context' not in st.session_state:
    st.session_state['context'] = ""  # Initialize context for each user

# Display the conversation in the chat container
with chat_placeholder.container():
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.markdown(f"<div style='text-align: right; color: #00bfff;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: #f1c40f;'>{message['content']}</div>", unsafe_allow_html=True)

# Input box at the bottom of the page
user_input = st.chat_input("Type a message")

# If the user provides input, append to messages and refresh
if user_input:
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    
    # Use the handle_conversation function to generate bot response
    bot_response, st.session_state['context'] = handle_conversation(user_input, st.session_state['context'])  # Pass context
    
    st.session_state['messages'].append({'role': 'bot', 'content': bot_response})

# Update the chat container to reflect new messages
with chat_placeholder.container():
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.markdown(f"<div style='text-align: right; color: #00bfff;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: #f1c40f;'>{message['content']}</div>", unsafe_allow_html=True)
