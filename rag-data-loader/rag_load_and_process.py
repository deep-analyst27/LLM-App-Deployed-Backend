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

PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="collectionragappdeployed",
    connection_string="postgresql+psycopg://postgres:langsmith@localhost:5432/ragappdeployed",
    pre_delete_collection=True,
)