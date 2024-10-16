import streamlit as st
from rag_module import handle_rag

def main():
    st.title("RAG Question Answering App")
    question = st.text_input("Enter your question:", "")
    
    if st.button("Submit"):
        if question:
            result = handle_rag(question)
            # Display the result
            st.subheader("Generated Answer:")
            st.write(result)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()


