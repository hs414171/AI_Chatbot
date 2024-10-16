import streamlit as st
from chat_module import handle_conversation
from rag_helper.pdf_helper import pdf_help
from rag_helper.methods import url_help # Import your chat and PDF processing functions
import os


def main():
    # Define the app title
    st.title("Assist GPT")

    # Sidebar for URL input (optional for this example)
    st.sidebar.header("Add Page URL (Optional)")
    url_input = st.sidebar.text_input("Enter Page URL")
    url_submit = st.sidebar.button("Submit URL")
    if url_submit:
        url = url_input
        url_help(url)
        st.success("Url uploaded and processed")

    # Button for uploading PDF files
    pdf_upload = st.sidebar.file_uploader("Upload PDF")

    # Create a container for the chat history
    chat_placeholder = st.empty()

    # Handle PDF upload (if a file is selected)
    if pdf_upload is not None:
        # Create a folder named "uploaded" if it doesn't exist
        if not os.path.exists("uploaded"):
            os.makedirs("uploaded")

        # Save the uploaded PDF file to the "uploaded" folder
        pdf_path = os.path.join("uploaded", pdf_upload.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_upload.read())

        # Process the uploaded PDF file
        pdf_help(pdf_path)

        # Delete the uploaded file after successful processing
        try:
            os.remove(pdf_path)
            st.success("PDF file uploaded, processed, and deleted successfully!")
        except OSError as e:
            st.error(f"Error deleting file: {e}")

    # Chat conversation management
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


if __name__ == "__main__":
    main()