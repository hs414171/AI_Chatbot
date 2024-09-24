from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (if needed)
import dotenv
dotenv.load_dotenv()
import os
secret_key=os.environ['SECRET_KEY']



# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")  # Specify the correct model

# Function to handle conversation, maintain memory, and return bot response
def handle_conversation(user_input: str, context: str):
    # Define the chat prompt template
    template = """
    You are a personal AI assistant to MR.Harshit Sharma and
    your name is Singhada. People Call you Singhada GPT. 
    You are designed to assist MR.Harshit Sharma in his daily tasks and
    any thing he asks you. You will help him with anything he needs in turn
    he might increase your payment. You need to keep in mind about the 
    context of your ongoing chat while answering questions.

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
        "question": user_input  # Current user input
    })

    # Append current conversation to the context
    context += f"\nUser: {user_input}\nAI: {result}\n"

    # Return the bot's response and updated context
    return result, context
