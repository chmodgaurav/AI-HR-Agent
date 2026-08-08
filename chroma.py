from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf=DirectoryLoader("./dataset/pdf",glob="**/*.pdf",loader_cls=PyPDFLoader).load()
json=DirectoryLoader("./dataset/json",glob="**/*.json",loader_cls=JSONLoader,loader_kwargs={
        "jq_schema": ".[]",
        "text_content": False,
    },).load()
doc=pdf+json

splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = splitter.split_documents(doc)

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

db = Chroma.from_documents(
    split_docs,
    embeddings,
    persist_directory="./database/chroma",
)