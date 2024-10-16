from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Cassandra
import cassio
import os
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
dotenv.load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults


app_token = os.environ['ASTRA_DB_APPLICATION_TOKEN']
db_id = os.environ["ASTRA_DB_ID"]


web_search_tool = TavilySearchResults(k=3)
cassio.init(token=app_token, database_id=db_id)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192") 

def web_search_tool():
    return TavilySearchResults(k=3)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initiate_retriever():
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="Document_Store",
        session=None,
        keyspace=None,
    )
    retriever = astra_vector_store.as_retriever()
    return retriever

     
def url_help(url):
    doc = WebBaseLoader(url).load()
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500,
        chunk_overlap=0,
        
    )
    docs_split = text_splitter.split_documents(doc)
    astra_vector_store = Cassandra(
        embedding=embeddings,
        table_name="Document_Store",
        session=None,
        keyspace=None,
    )
    astra_vector_store.add_documents(docs_split)


def retriever_grader(retriever,question):
    
    template = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
        of a retrieved document to a user question. If the document contains keywords related to the user question,
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
    prompt = PromptTemplate.from_template(template)

    retrieval_grader = prompt | llm | JsonOutputParser()
    
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    decision = retrieval_grader.invoke({"question": question, "document": doc_txt})
    return decision,retrieval_grader

def generator(retriever,question):
    prompt = PromptTemplate.from_template(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question}
    Context: {context}
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    
    )
    

    rag_chain = prompt | llm | StrOutputParser()
    
    docs = retriever.invoke(question)
    generation = rag_chain.invoke({"context": docs, "question": question})
    return generation,rag_chain

def hallucinator(retriever,question):
    prompt = PromptTemplate.from_template(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
    an answer is grounded in / supported by a set of facts. Give a binary score 'yes' or 'no' score to indicate
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    )
    generation,_ = generator(retriever,question)
    docs = retriever.invoke(question)
    hallucination_grader = prompt | llm | JsonOutputParser()
    grade_score = hallucination_grader.invoke({"documents": docs, "generation": generation})
    return grade_score,hallucination_grader

def answer_grader(retriever,question):
    prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    )

    generation,_ = generator(retriever,question)
    answer_grader = prompt | llm | JsonOutputParser()
    answer_score = answer_grader.invoke({"question": question, "generation": generation})
    return answer_score,generation,answer_grader

def router(retriever,question):
    
    prompt = PromptTemplate(
         template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
            user question to a vectorstore or web search. Use the vectorstore for questions on the following context: {context}. You do not need to be stringent with the keywords
            in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
            or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
            no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",

    )
    doc =retriever.invoke(question)
    question_router = prompt | llm | JsonOutputParser()

    
    routed = question_router.invoke({"question": question,"context":doc})
    return routed,question_router