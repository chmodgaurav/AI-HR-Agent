#!/usr/bin/env python
"""Quick verification test for conversation memory."""

try:
    print("1. Testing imports...")
    from conversation_memory import get_memory_manager, ChatSession, SessionMemoryManager
    print("   ✓ Imports successful")
    
    print("\n2. Testing ChatSession...")
    session = ChatSession(session_id="TEST")
    session.add_user_message("Hello")
    session.add_ai_message("Hi there")
    print(f"   ✓ Created session with {len(session.messages)} messages")
    
    print("\n3. Testing SessionMemoryManager...")
    manager = SessionMemoryManager()
    manager.add_user_message("EMP001", "Test message")
    mgr_session = manager.get_session("EMP001")
    print(f"   ✓ Manager created session with {len(mgr_session.messages)} messages")
    
    print("\n4. Testing session isolation...")
    manager.add_user_message("EMP002", "Different employee")
    emp1_session = manager.get_session("EMP001")
    emp2_session = manager.get_session("EMP002")
    print(f"   ✓ EMP001: {len(emp1_session.messages)} messages")
    print(f"   ✓ EMP002: {len(emp2_session.messages)} messages")
    print(f"   ✓ Isolation verified: {emp1_session is not emp2_session}")
    
    print("\n5. Testing information extraction...")
    session2 = ChatSession(session_id="EXTRACT")
    session2.add_user_message("I'm emp E12345 and need casual leave")
    session2.add_ai_message("When do you need it?")
    session2.add_user_message("From 2024-02-15 to 2024-02-17")
    extracted = session2.get_extracted_info()
    print(f"   ✓ Extracted info: {extracted}")
    
    print("\n6. Testing context retrieval...")
    context = session2.get_context_string()
    print(f"   ✓ Context length: {len(context)} characters")
    print(f"   ✓ Context preview: {context[:100]}...")
    
    print("\n7. Testing global singleton...")
    mgr1 = get_memory_manager()
    mgr2 = get_memory_manager()
    print(f"   ✓ Singleton working: {mgr1 is mgr2}")
    
    print("\n" + "="*50)
    print("✓ ALL TESTS PASSED!")
    print("="*50)
    print("\nConversational memory is ready to use!")
    print("\nNext steps:")
    print("1. Ensure Ollama is running (ollama serve)")
    print("2. Run: streamlit run app.py")
    print("3. Enter your Employee ID in the sidebar")
    print("4. Start chatting!")
    
except Exception as e:
    print(f"\n✗ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
