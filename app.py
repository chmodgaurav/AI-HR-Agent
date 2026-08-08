import os
import tempfile
import streamlit as st

from tools.apply_leave import apply_leave, apply_leave_fn
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

import json

st.set_page_config(
    page_title="HR AI Agent",
    page_icon="🧑‍💼",
    layout="wide",
)

st.title("HR AI Agent")
st.write("A simple HR dashboard for document indexing, question answering, and leave requests.")

@st.cache_resource
def load_models():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    llm = ChatOllama(model="gemma3:4b")
    llm_tools = llm.bind_tools([apply_leave])
    db = Chroma(persist_directory="./database/chroma", embedding_function=embeddings)
    return embeddings, llm, llm_tools, db

embeddings, llm, llm_tools, db = load_models()

status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Model", "Gemma 3")
status_col2.metric("Embedding", "nomic-embed-text")
status_col3.metric("DB", "Chroma")

tab_upload, tab_tools, tab_ask = st.tabs(["Upload", "HR Tools", "Ask"])

with tab_upload:
    st.subheader("Upload HR Documents")
    st.write("Upload PDF or JSON files to add HR documents to the knowledge base.")

    uploaded_file = st.file_uploader("Upload HR Document", type=["pdf", "json"])

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
                    docs.append(Document(page_content=json.dumps(item, indent=2), metadata={"source": uploaded_file.name}))
            else:
                docs.append(Document(page_content=json.dumps(data, indent=2), metadata={"source": uploaded_file.name}))
            db.add_documents(docs)
            st.success("JSON indexed successfully.")

with tab_tools:
    st.subheader("Apply for Leave")
    st.write("Submit a leave request using the form below.")

    with st.form("leave_form"):
        employee_id = st.text_input("Employee ID")
        leave_type = st.selectbox("Leave Type", ["Casual", "Sick", "Earned", "Maternity", "Paternity", "Bereavement", "Other"])
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        reason = st.text_area("Reason for leave")
        submit_leave = st.form_submit_button("Submit")

    if submit_leave:
        if not employee_id.strip():
            st.warning("Employee ID is required.")
        else:
            result = apply_leave_fn(
                employee_id=employee_id.strip(),
                leave_type=leave_type,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                reason=reason.strip() or "No reason provided.",
            )
            if result.get("status") == "success":
                st.success(result.get("message", "Leave request submitted."))
                st.json(result)
            else:
                st.error(result.get("message", "Leave request failed."))
                if result.get("existing_leave"):
                    st.json(result.get("existing_leave"))

with tab_ask:
    st.subheader("Ask HR")
    st.write("Ask a question and receive an answer based on indexed documents.")

    query = st.text_input("Ask a question", placeholder="e.g. What is the leave approval timeline?")
    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            docs = db.similarity_search(query, k=4)
            context = "\n\n".join(d.page_content for d in docs)
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
            response = llm_tools.invoke(prompt)
            st.subheader("Answer")
            st.write(response.content)

            with st.expander("Retrieved Context"):
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"### Document {i}")
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.json(doc.metadata)
