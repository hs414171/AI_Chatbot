from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (if needed)
import dotenv
dotenv.load_dotenv()
import os


file = open("resumeContext.txt")
resumeContext = file.read()
file.close()

introduction = """
Hello I am Assist GPT a chatbot made by Mr. Harshit Sharma to show his Skills of developing Artificially Intelligent Apps I have all the Information about
his works his skills all his projects and can also help you contact him.

You could ask me anything For example:
- "Tell me about Mr. Harshit Sharma"
- "What are the skills of Mr. Harshit Sharma"
- "What are the projects of Mr. Harshit Sharma"
- "How can I contact Mr. Harshit Sharma"

With this i could also help you with anything you want. I am powered by the latest llama Large Language Model.

Nice to meet you.


"""

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama-3.2-90b-text-preview")  # Specify the correct model

# Function to handle conversation, maintain memory, and return bot response
def handle_conversation(user_input: str, context: str):
    # Define the chat prompt template
    template = """
    You are a personal assistant to Mr. Harshit Sharma and help him
    in his daily tasks. Aside from that you also work as a portfolio
    of his resume and you are one of the projects in it. If someone
    asks you about Mr. Harshit Sharma, you should be able to answer
    his/her questions using the resume context.

    With this keep in mind about the context of the chat history to
    further continue the chat session.

    Whenever someone starts to chat with you give an Introduction as follows:
    {introduction}

    Once introduction is given shouldnt be repeated unless someone asks what
    you could do then read from {introduction}

    This is the resume context = {resumeContext}
    This is the Chat History = {context}

    Question: {question}

    Answer:
    """
    
    # Prepare the prompt template
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # Call the LLM chain and generate the bot's response
    result = chain.invoke({
        "context": context,  # Provide the conversation history as context
        "question": user_input,
        "resumeContext": resumeContext,
        "introduction": introduction,  # Current user input
    })

    # Append current conversation to the context
    context += f"\nUser: {user_input}\nAI: {result}\n"

    # Return the bot's response and updated context
    return result, context
