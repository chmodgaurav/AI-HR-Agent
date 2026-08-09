import os
import tempfile
import streamlit as st
import uuid

from tools.apply_leave import apply_leave, apply_leave_fn
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from conversation_memory import get_memory_manager
from memory_chat_engine import MemoryChatEngine, MultiTurnLeaveHandler

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
    memory_manager = get_memory_manager(session_ttl_minutes=120)
    chat_engine = MemoryChatEngine(
        llm=llm,
        memory_manager=memory_manager,
        vector_db=db,
        tools=[apply_leave],
    )
    leave_handler = MultiTurnLeaveHandler(chat_engine, apply_leave_fn)
    return embeddings, llm, llm_tools, db, memory_manager, chat_engine, leave_handler

embeddings, llm, llm_tools, db, memory_manager, chat_engine, leave_handler = load_models()

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "employee_id" not in st.session_state:
    st.session_state.employee_id = None

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "leave_mode" not in st.session_state:
    st.session_mode = False

# Sidebar for session management
with st.sidebar:
    st.subheader("Session Management")
    
    # Employee ID input
    employee_id = st.text_input(
        "Employee ID",
        value=st.session_state.employee_id or "",
        placeholder="Enter your employee ID",
        help="Used to isolate conversation history"
    )
    
    if employee_id and employee_id != st.session_state.employee_id:
        st.session_state.employee_id = employee_id
        # Create/load session for this employee
        session = memory_manager.get_or_create_session(employee_id)
        st.success(f"Session initialized for Employee {employee_id}")
    
    st.divider()
    
    # Session info
    if st.session_state.employee_id:
        session_info = memory_manager.get_session_info(st.session_state.employee_id)
        if session_info:
            st.metric("Messages", session_info.get("message_count", 0))
            st.metric("Session ID", st.session_state.session_id[:8] + "...")
        
        # Clear session button
        if st.button("🗑️ Clear Session History"):
            memory_manager.delete_session(st.session_state.employee_id)
            st.success("Session history cleared")
            st.rerun()
    
    st.divider()
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    st.caption("Conversations are isolated per employee and expire after 2 hours.")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)
status_col1.metric("Model", "Gemma 3")
status_col2.metric("Embedding", "nomic-embed-text")
status_col3.metric("DB", "Chroma")
status_col4.metric("Active Sessions", memory_manager.get_session_count())

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
    st.subheader("Apply for Leave (Multi-turn)")
    st.write("Chat naturally to apply for leave. The AI will collect information step by step.")
    
    if not st.session_state.employee_id:
        st.warning("⚠️ Please enter your Employee ID in the sidebar to use this feature.")
    else:
        # Display conversation history
        session = memory_manager.get_session(st.session_state.employee_id)
        if session and session.get_messages():
            st.subheader("Conversation History")
            messages = session.get_messages()
            
            for msg in messages[-10:]:  # Show last 10 messages
                if msg.__class__.__name__ == "HumanMessage":
                    st.write("**You:** " + msg.content)
                else:
                    st.write("**Assistant:** " + msg.content)
        
        # Input for leave application
        st.subheader("Chat with HR Assistant")
        
        user_input = st.text_input(
            "Your message",
            placeholder="e.g., I want to apply for leave or Tell me about casual leave",
            key="leave_chat_input"
        )
        
        if st.button("Send", key="leave_send_btn"):
            if not user_input.strip():
                st.warning("Please enter a message.")
            else:
                with st.spinner("Processing..."):
                    # Check if this is a leave request
                    if any(keyword in user_input.lower() for keyword in ["leave", "apply", "request", "time off"]):
                        result = leave_handler.process_message(st.session_state.employee_id, user_input)
                    else:
                        # Regular chat
                        result = chat_engine.chat(st.session_state.employee_id, user_input)
                    
                    if result.get("status") != "error":
                        st.success("Message processed!")
                        st.write("**Assistant:** " + result.get("response", ""))
                        
                        # Show extracted info if in collecting mode
                        if result.get("status") == "collecting" and result.get("missing_fields"):
                            with st.expander("Information collected so far"):
                                st.json(result.get("collected_fields", {}))
                        
                        # Show tool result if successful
                        if result.get("tool_result"):
                            with st.expander("Leave Request Details"):
                                st.json(result.get("tool_result"))
                    else:
                        st.error("Error: " + result.get("error", "Unknown error"))
        
        # Show current context
        if st.checkbox("Show Session Context"):
            context = chat_engine.get_session_context(st.session_state.employee_id)
            st.subheader("Session Context")
            if context.get("extracted_info"):
                st.write("**Extracted Information:**")
                st.json(context["extracted_info"])
            if context.get("context"):
                st.write("**Recent Conversation:**")
                st.text(context["context"])

with tab_ask:
    st.subheader("Ask HR (with Memory)")
    st.write("Ask questions and the AI will remember your previous questions to provide better context.")
    
    if not st.session_state.employee_id:
        st.warning("⚠️ Please enter your Employee ID in the sidebar to use this feature.")
    else:
        # Display conversation history
        session = memory_manager.get_session(st.session_state.employee_id)
        if session and session.get_messages():
            st.subheader("Conversation History")
            messages = session.get_messages()
            
            for msg in messages[-10:]:  # Show last 10 messages
                if msg.__class__.__name__ == "HumanMessage":
                    st.write("**You:** " + msg.content)
                else:
                    st.write("**Assistant:** " + msg.content)
        
        # Input for questions
        st.subheader("Ask a Question")
        
        query = st.text_input(
            "Your question",
            placeholder="e.g., What is the leave approval timeline?",
            key="ask_query_input"
        )
        
        if st.button("Ask", key="ask_btn"):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching knowledge base and generating response..."):
                    result = chat_engine.chat_with_qa(st.session_state.employee_id, query)
                    
                    if result.get("status") == "success":
                        st.success("Answer found!")
                        st.subheader("Answer")
                        st.write(result.get("response", ""))
                        
                        # Show context
                        if st.checkbox("Show Conversation Context"):
                            st.text_area(
                                "Conversation Context",
                                value=result.get("context", ""),
                                height=200,
                                disabled=True
                            )
                    else:
                        st.error("Error: " + result.get("error", "Unknown error"))
        
        # Show related documents option
        if query.strip():
            if st.checkbox("Show Related Documents"):
                st.subheader("Retrieved from Knowledge Base")
                docs = db.similarity_search(query, k=4)
                
                if docs:
                    for i, doc in enumerate(docs, 1):
                        with st.expander(f"Document {i}"):
                            st.write(doc.page_content)
                            if doc.metadata:
                                st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                else:
                    st.info("No related documents found.")
