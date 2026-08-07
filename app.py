import os
import tempfile
import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

import json

st.set_page_config(page_title="HR AI Agent", page_icon="🧑‍💼")

st.title("🧑‍💼 HR AI Agent")

# -----------------------------
# Models
# -----------------------------

@st.cache_resource
def load_models():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    llm = ChatOllama(
        model="gemma3:4b"
    )

    db = Chroma(
        persist_directory="./chroma",
        embedding_function=embeddings
    )

    return embeddings, llm, db


embeddings, llm, db = load_models()

# -----------------------------
# Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload HR Document",
    type=["pdf", "json"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".pdf"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        db.add_documents(docs)

        os.remove(pdf_path)

        st.success("PDF indexed successfully.")

    elif uploaded_file.name.endswith(".json"):

        data = json.load(uploaded_file)

        docs = []

        if isinstance(data, list):

            for item in data:
                docs.append(
                    Document(
                        page_content=json.dumps(item, indent=2),
                        metadata={"source": uploaded_file.name}
                    )
                )

        else:

            docs.append(
                Document(
                    page_content=json.dumps(data, indent=2),
                    metadata={"source": uploaded_file.name}
                )
            )

        db.add_documents(docs)

        st.success("JSON indexed successfully.")

# -----------------------------
# Chat
# -----------------------------

query = st.text_input(
    "Ask an HR question"
)

if st.button("Ask"):

    if query.strip() == "":
        st.warning("Enter a question.")
        st.stop()

    docs = db.similarity_search(query, k=4)

    context = "\n\n".join(
        d.page_content for d in docs
    )

    prompt = f"""
You are an HR Assistant.

Answer ONLY from the provided context.

If the answer isn't present, say:
"I couldn't find that information."

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    st.subheader("Answer")

    st.write(response.content)

    with st.expander("Retrieved Context"):

        for i, doc in enumerate(docs, 1):

            st.markdown(f"### Document {i}")

            st.write(doc.page_content)

            if doc.metadata:
                st.json(doc.metadata)