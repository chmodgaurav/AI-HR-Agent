from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
llm=ChatOllama(model="gemma3:4b")

db = Chroma(
    persist_directory="./database",
    embedding_function=embeddings
)

query=input("Enter your query: ")
results = db.similarity_search(query)
for i in results:
    print(i.page_content)