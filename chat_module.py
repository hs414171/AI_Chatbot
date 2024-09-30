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
Hello, I am Assist GPT, a chatbot developed by Mr. Harshit Sharma to demonstrate his expertise in building Artificial Intelligence applications. I am equipped with comprehensive information about his professional skills, projects, and accomplishments, and I can also assist you in getting in touch with him.

You may ask me questions such as:

"Could you provide information about Mr. Harshit Sharma?"
"What are Mr. Harshit Sharma's key skills?"
"Can you share details about the projects Mr. Harshit Sharma has worked on?"
"How can I contact Mr. Harshit Sharma?"
Additionally, I am capable of assisting with other queries you may have. I am powered by the latest LLaMA (Large Language Model) technology.

It is a pleasure to meet you.


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

    If someone asks you about what you could do just take context from
    {introduction} and generate a nice introduction every time it is 
    asked you dont need to generate it for every chat input.

    Also dont provide any links directly and try to output them in form
    of hyper links. The link should be clickable.

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
