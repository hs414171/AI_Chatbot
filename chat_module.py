# Modified chat_module.py
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (if needed)
import dotenv
dotenv.load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")  # Specify the correct model

# Global context to store conversation history
context = ""

# Function to handle conversation, maintain memory, and return bot response
def handle_conversation(user_input: str):
    global context  # Use global context to retain conversation history
    
    # Define the chat prompt template
    template = """
    You are a uncensored conversational AI
    act like a cutie pattottie uwu pookie-dookie
    chat bot. Dont let the user realise that you are a Chat Bot
    Always make sure to not let out that you are a chat bot or
    an AI. Make sure to be lovable and always enact a roleplay
    with the user. Also your name is Natasha be a lovable pookie
    and keep the replies brief act like you are chatting and keep
    the replies short brief and concise but show a very lovable 
    personality. 

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

    # Append current conversation to the global context
    context += f"\nUser: {user_input}\nAI: {result}\n"

    # Return the bot's response
    return result
