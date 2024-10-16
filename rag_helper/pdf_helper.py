from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Cassandra
import cassio
import os
import dotenv
from langchain_huggingface import HuggingFaceEmbeddings

dotenv
dotenv.load_dotenv()
app_token = os.environ['ASTRA_DB_APPLICATION_TOKEN']
db_id = os.environ["ASTRA_DB_ID"]
cassio.init(token=app_token, database_id=db_id)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def pdf_help(pdf_path):
    doc = PyPDFLoader(pdf_path).load()
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

