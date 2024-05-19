import os

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain_community.vectorstores.pgvector import PGVector
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

loader = DirectoryLoader(
    os.path.abspath("../llm-rag-deployed/pdf-documents"),
    glob="**/*.pdf",
    use_multithreading=True,
    show_progress=True,
    max_concurrency=50,
    loader_cls=UnstructuredPDFLoader,
)
docs = loader.load()

embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', )

text_splitter = SemanticChunker(
    embeddings=embeddings
)


chunks = text_splitter.split_documents(documents=docs)

user = os.environ['DATABASE_USER']
password = os.environ['DATABASE_PASSWORD']
host = os.environ['DATABASE_HOST']
db_port = os.environ['DATABASE_PORT']
db_name = os.environ['DATABASE_NAME']
db_name_new = os.environ['DATABASE_NAME_NEW']


PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="collectionragappdeployed",
    connection_string=f"postgresql+psycopg://{user}:{password}@{host}:{db_port}/{db_name}",
    pre_delete_collection=True,
)
