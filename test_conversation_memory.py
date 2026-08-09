"""
Unit tests for conversational memory module.
Tests session management, message storage, and context retrieval.
"""

import pytest
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from conversation_memory import ChatSession, SessionMemoryManager, get_memory_manager


class TestChatSession:
    """Tests for ChatSession class."""
    
    def test_session_creation(self):
        """Test creating a new session."""
        session = ChatSession(session_id="EMP001")
        
        assert session.session_id == "EMP001"
        assert len(session.messages) == 0
        assert isinstance(session.created_at, datetime)
    
    def test_add_user_message(self):
        """Test adding a user message."""
        session = ChatSession(session_id="EMP001")
        session.add_user_message("I want to apply for leave")
        
        assert len(session.messages) == 1
        assert isinstance(session.messages[0], HumanMessage)
        assert session.messages[0].content == "I want to apply for leave"
    
    def test_add_ai_message(self):
        """Test adding an AI message."""
        session = ChatSession(session_id="EMP001")
        session.add_ai_message("I can help you apply for leave")
        
        assert len(session.messages) == 1
        assert isinstance(session.messages[0], AIMessage)
        assert session.messages[0].content == "I can help you apply for leave"
    
    def test_get_last_n_messages(self):
        """Test retrieving last N messages."""
        session = ChatSession(session_id="EMP001")
        session.add_user_message("Message 1")
        session.add_ai_message("Response 1")
        session.add_user_message("Message 2")
        session.add_ai_message("Response 2")
        
        last_two = session.get_last_n_messages(2)
        assert len(last_two) == 2
        assert last_two[0].content == "Message 2"
        assert last_two[1].content == "Response 2"
    
    def test_get_context_string(self):
        """Test context string formatting."""
        session = ChatSession(session_id="EMP001")
        session.add_user_message("I want casual leave")
        session.add_ai_message("When do you need the leave?")
        
        context = session.get_context_string()
        
        assert "User: I want casual leave" in context
        assert "Assistant: When do you need the leave?" in context
    
    def test_get_extracted_info_leave_type(self):
        """Test extracting leave type from conversation."""
        session = ChatSession(session_id="EMP001")
        session.add_user_message("I need casual leave")
        
        info = session.get_extracted_info()
        
        assert "leave_type" in info
        assert info["leave_type"] == "Casual"
    
    def test_clear_messages(self):
        """Test clearing session messages."""
        session = ChatSession(session_id="EMP001")
        session.add_user_message("Test message")
        assert len(session.messages) == 1
        
        session.clear_messages()
        
        assert len(session.messages) == 0


class TestSessionMemoryManager:
    """Tests for SessionMemoryManager class."""
    
    def test_create_session(self):
        """Test creating a new session."""
        manager = SessionMemoryManager()
        session = manager.create_session("EMP001")
        
        assert session.session_id == "EMP001"
        assert manager.get_session("EMP001") is session
    
    def test_session_isolation(self):
        """Test that sessions are isolated."""
        manager = SessionMemoryManager()
        
        manager.add_user_message("EMP001", "User 1 message")
        manager.add_user_message("EMP002", "User 2 message")
        
        session1 = manager.get_session("EMP001")
        session2 = manager.get_session("EMP002")
        
        assert len(session1.messages) == 1
        assert len(session2.messages) == 1
        assert session1.messages[0].content == "User 1 message"
        assert session2.messages[0].content == "User 2 message"
    
    def test_get_or_create_session_existing(self):
        """Test getting an existing session."""
        manager = SessionMemoryManager()
        session1 = manager.create_session("EMP001")
        session2 = manager.get_or_create_session("EMP001")
        
        assert session1 is session2
    
    def test_get_or_create_session_new(self):
        """Test creating a session if it doesn't exist."""
        manager = SessionMemoryManager()
        session = manager.get_or_create_session("EMP001")
        
        assert session.session_id == "EMP001"
        assert manager.get_session("EMP001") is session
    
    def test_add_user_message(self):
        """Test adding user message via manager."""
        manager = SessionMemoryManager()
        manager.add_user_message("EMP001", "Test message")
        
        session = manager.get_session("EMP001")
        assert len(session.messages) == 1
        assert isinstance(session.messages[0], HumanMessage)
    
    def test_add_ai_message(self):
        """Test adding AI message via manager."""
        manager = SessionMemoryManager()
        manager.add_ai_message("EMP001", "Test response")
        
        session = manager.get_session("EMP001")
        assert len(session.messages) == 1
        assert isinstance(session.messages[0], AIMessage)
    
    def test_get_context(self):
        """Test retrieving context."""
        manager = SessionMemoryManager()
        manager.add_user_message("EMP001", "User message")
        manager.add_ai_message("EMP001", "AI response")
        
        context = manager.get_context("EMP001")
        
        assert "User: User message" in context
        assert "Assistant: AI response" in context
    
    def test_get_context_with_limit(self):
        """Test retrieving limited context."""
        manager = SessionMemoryManager()
        manager.add_user_message("EMP001", "Message 1")
        manager.add_user_message("EMP001", "Message 2")
        manager.add_user_message("EMP001", "Message 3")
        
        context = manager.get_context("EMP001", last_n=2)
        
        assert "Message 1" not in context
        assert "Message 2" in context
        assert "Message 3" in context
    
    def test_delete_session(self):
        """Test deleting a session."""
        manager = SessionMemoryManager()
        manager.create_session("EMP001")
        
        result = manager.delete_session("EMP001")
        
        assert result is True
        assert manager.get_session("EMP001") is None
    
    def test_session_count(self):
        """Test counting active sessions."""
        manager = SessionMemoryManager()
        
        manager.create_session("EMP001")
        manager.create_session("EMP002")
        
        assert manager.get_session_count() == 2
    
    def test_clear_all_sessions(self):
        """Test clearing all sessions."""
        manager = SessionMemoryManager()
        manager.create_session("EMP001")
        manager.create_session("EMP002")
        
        manager.clear_all_sessions()
        
        assert manager.get_session_count() == 0


class TestGlobalMemoryManager:
    """Tests for global memory manager singleton."""
    
    def test_get_memory_manager_singleton(self):
        """Test that get_memory_manager returns same instance."""
        # Clear any existing manager first
        import conversation_memory
        conversation_memory._memory_manager = None
        
        manager1 = get_memory_manager()
        manager2 = get_memory_manager()
        
        assert manager1 is manager2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
