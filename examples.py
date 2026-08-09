"""
Example Usage of Conversational Memory

This script demonstrates how to use the conversational memory system
outside of the Streamlit application.
"""

from conversation_memory import get_memory_manager
from memory_chat_engine import MemoryChatEngine, MultiTurnLeaveHandler
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from tools.apply_leave import apply_leave_fn


def example_basic_conversation():
    """Example 1: Basic multi-turn conversation."""
    print("=" * 60)
    print("Example 1: Basic Multi-turn Conversation")
    print("=" * 60)
    
    # Initialize
    memory_manager = get_memory_manager(session_ttl_minutes=120)
    llm = ChatOllama(model="gemma3:4b")
    db = Chroma(
        persist_directory="./database/chroma",
        embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest")
    )
    
    chat_engine = MemoryChatEngine(
        llm=llm,
        memory_manager=memory_manager,
        vector_db=db,
    )
    
    employee_id = "EMP001"
    
    # Turn 1
    print("\nTurn 1 - User asks about leave")
    response1 = chat_engine.chat(
        session_id=employee_id,
        user_message="What are my leave entitlements?"
    )
    print(f"Response: {response1['response'][:200]}...")
    
    # Turn 2
    print("\nTurn 2 - User asks follow-up")
    response2 = chat_engine.chat(
        session_id=employee_id,
        user_message="Can I take them all at once?"
    )
    print(f"Response: {response2['response'][:200]}...")
    
    # Check context
    print("\nContext maintained:")
    context = chat_engine.get_session_context(employee_id)
    print(context["context"][:300] + "...")


def example_leave_application():
    """Example 2: Multi-turn leave application."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-turn Leave Application")
    print("=" * 60)
    
    # Initialize
    memory_manager = get_memory_manager()
    llm = ChatOllama(model="gemma3:4b")
    
    chat_engine = MemoryChatEngine(llm=llm, memory_manager=memory_manager)
    leave_handler = MultiTurnLeaveHandler(chat_engine, apply_leave_fn)
    
    employee_id = "EMP123"
    
    # Turn 1: Express intent
    print("\nTurn 1 - Employee wants to apply for leave")
    result = leave_handler.process_message(
        session_id=employee_id,
        user_message="I want to apply for casual leave next week"
    )
    print(f"Status: {result['status']}")
    print(f"Response: {result['response']}")
    if result.get("missing_fields"):
        print(f"Missing: {list(result['missing_fields'].keys())}")
    
    # Turn 2: Provide dates
    print("\nTurn 2 - Employee provides dates")
    result = leave_handler.process_message(
        session_id=employee_id,
        user_message="From 2024-02-15 to 2024-02-17"
    )
    print(f"Status: {result['status']}")
    if result.get("missing_fields"):
        print(f"Missing: {list(result['missing_fields'].keys())}")
    
    # Turn 3: Provide reason
    print("\nTurn 3 - Employee provides reason")
    result = leave_handler.process_message(
        session_id=employee_id,
        user_message="Personal reasons"
    )
    print(f"Status: {result['status']}")
    print(f"Response: {result['response']}")
    
    if result.get("tool_result"):
        print("\nTool Result:")
        print(f"  Status: {result['tool_result'].get('status')}")
        print(f"  Message: {result['tool_result'].get('message')}")


def example_session_isolation():
    """Example 3: Session isolation between employees."""
    print("\n" + "=" * 60)
    print("Example 3: Session Isolation")
    print("=" * 60)
    
    memory = get_memory_manager()
    
    # Employee A
    print("\nEmployee A adding message...")
    memory.add_user_message("EMP_A", "I work in the Finance department")
    session_a = memory.get_session("EMP_A")
    print(f"Employee A messages: {len(session_a.messages)}")
    print(f"Content: {session_a.messages[0].content}")
    
    # Employee B
    print("\nEmployee B checking sessions...")
    memory.add_user_message("EMP_B", "I work in HR")
    session_b = memory.get_session("EMP_B")
    print(f"Employee B messages: {len(session_b.messages)}")
    print(f"Content: {session_b.messages[0].content}")
    
    # Verify isolation
    print("\nVerifying isolation...")
    print(f"Employee A can see own messages: {len(session_a.messages) > 0}")
    print(f"Employee B can see own messages: {len(session_b.messages) > 0}")
    print(f"Employee A sees only own messages: {session_a.messages[0].content != session_b.messages[0].content}")


def example_context_extraction():
    """Example 4: Information extraction from conversation."""
    print("\n" + "=" * 60)
    print("Example 4: Information Extraction")
    print("=" * 60)
    
    memory = get_memory_manager()
    session = memory.get_or_create_session("EMP_TEST")
    
    # Add conversation
    session.add_user_message("I'm employee E12345 and I need casual leave")
    session.add_ai_message("When do you need the leave?")
    session.add_user_message("From 2024-02-15 to 2024-02-17 for medical reasons")
    
    # Extract information
    print("\nExtracted Information:")
    extracted = session.get_extracted_info()
    
    for key, value in extracted.items():
        print(f"  {key}: {value}")


def example_session_metrics():
    """Example 5: Session metrics and monitoring."""
    print("\n" + "=" * 60)
    print("Example 5: Session Metrics")
    print("=" * 60)
    
    memory = get_memory_manager()
    
    # Create multiple sessions
    for i in range(3):
        emp_id = f"EMP_{i}"
        memory.add_user_message(emp_id, f"Message from employee {i}")
        memory.add_ai_message(emp_id, f"Response to employee {i}")
    
    # Get metrics
    print(f"\nTotal Active Sessions: {memory.get_session_count()}")
    
    # Get detailed info for each
    print("\nSession Details:")
    for i in range(3):
        emp_id = f"EMP_{i}"
        info = memory.get_session_info(emp_id)
        if info:
            print(f"  {emp_id}:")
            print(f"    Messages: {info['message_count']}")
            print(f"    Created: {info['created_at']}")


def example_programmatic_chat():
    """Example 6: Using chat engine programmatically."""
    print("\n" + "=" * 60)
    print("Example 6: Programmatic Chat Usage")
    print("=" * 60)
    
    try:
        memory_manager = get_memory_manager()
        llm = ChatOllama(model="gemma3:4b")
        db = Chroma(
            persist_directory="./database/chroma",
            embedding_function=OllamaEmbeddings(model="nomic-embed-text:latest")
        )
        
        chat_engine = MemoryChatEngine(
            llm=llm,
            memory_manager=memory_manager,
            vector_db=db,
        )
        
        print("\nChat with Q&A mode:")
        result = chat_engine.chat_with_qa(
            session_id="EMP_QA",
            user_message="What is the policy on sick leave?"
        )
        
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Response: {result['response'][:300]}...")
        else:
            print(f"Error: {result.get('error')}")
    
    except Exception as e:
        print(f"Note: This example requires Ollama to be running")
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    print("\nConversational Memory - Examples\n")
    
    # Run examples
    try:
        example_session_isolation()
        example_context_extraction()
        example_session_metrics()
        
        # These require Ollama to be running
        print("\n" + "=" * 60)
        print("Note: The following examples require Ollama to be running")
        print("=" * 60)
        # example_basic_conversation()
        # example_leave_application()
        # example_programmatic_chat()
        
        print("\n✓ All examples completed successfully!")
    
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
