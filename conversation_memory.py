"""
Conversational Memory Module

Manages in-memory session storage for multi-turn conversations.
Provides session isolation per employee and context retrieval.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class ChatSession:
    """Represents a single chat session for an employee."""
    
    session_id: str  # Employee ID
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    messages: List[BaseMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
        self.last_accessed = datetime.now()
    
    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message(HumanMessage(content=content))
    
    def add_ai_message(self, content: str) -> None:
        """Add an AI message."""
        self.add_message(AIMessage(content=content))
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the session."""
        return self.messages
    
    def get_last_n_messages(self, n: int) -> List[BaseMessage]:
        """Get the last N messages."""
        return self.messages[-n:] if n > 0 else []
    
    def get_context_string(self) -> str:
        """Get conversation context as a formatted string."""
        if not self.messages:
            return ""
        
        context_lines = []
        for msg in self.messages:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            context_lines.append(f"{role}: {msg.content}")
        
        return "\n".join(context_lines)
    
    def get_extracted_info(self) -> Dict[str, Any]:
        """
        Extract structured information from the conversation.
        Looks for common patterns like leave type, dates, employee info, etc.
        """
        import re
        from datetime import datetime
        
        extracted = {}
        context_str = self.get_context_string()
        context_lower = context_str.lower()
        
        # Extract employee ID (common patterns: EMP123, E123, ID: 123)
        emp_id_patterns = [
            r'employee\s*(?:id|#)?\s*[:=]?\s*([A-Z0-9]+)',
            r'emp\s*[:=]?\s*([A-Z0-9]+)',
            r'id\s*[:=]?\s*([A-Z0-9]{3,})',
        ]
        for pattern in emp_id_patterns:
            match = re.search(pattern, context_str, re.IGNORECASE)
            if match:
                extracted["employee_id"] = match.group(1)
                break
        
        # Extract leave type
        leave_types = ["casual", "sick", "earned", "maternity", "paternity", "bereavement"]
        for leave_type in leave_types:
            if leave_type in context_lower:
                extracted["leave_type"] = leave_type.title()
                break
        
        # Extract dates (YYYY-MM-DD format or natural language)
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        dates = re.findall(date_pattern, context_str)
        if dates:
            if len(dates) >= 1:
                extracted["start_date"] = dates[0]
            if len(dates) >= 2:
                extracted["end_date"] = dates[1]
            elif len(dates) >= 1:
                # If only one date provided, use it as both start and end
                extracted["end_date"] = dates[0]
        
        # Extract reason (last substantive message or after "reason" keyword)
        reason_pattern = r'reason\s*[:=]?\s*([^\.]+)'
        reason_match = re.search(reason_pattern, context_str, re.IGNORECASE)
        if reason_match:
            extracted["reason"] = reason_match.group(1).strip()
        elif len(self.messages) > 0:
            # Use the last non-AI message as reason if not explicitly stated
            for msg in reversed(self.messages):
                if isinstance(msg, HumanMessage):
                    content = msg.content.strip()
                    # Skip if it's just asking for leave or short responses
                    if len(content) > 10 and not any(keyword in content.lower() for keyword in ["when", "what", "which", "type"]):
                        extracted["reason"] = content
                        break
        
        return extracted
    
    def clear_messages(self) -> None:
        """Clear all messages from the session."""
        self.messages = []
        self.last_accessed = datetime.now()


class SessionMemoryManager:
    """
    Manages conversation sessions with isolation per employee.
    
    Provides:
    - Session creation and retrieval
    - Message history management
    - Context extraction for tools
    - Session cleanup
    """
    
    def __init__(self, session_ttl_minutes: int = 120):
        """
        Initialize the session memory manager.
        
        Args:
            session_ttl_minutes: Time-to-live for sessions (minutes)
        """
        self._sessions: Dict[str, ChatSession] = {}
        self._session_ttl_minutes = session_ttl_minutes
    
    def create_session(self, session_id: str) -> ChatSession:
        """
        Create a new session for an employee.
        
        Args:
            session_id: Unique identifier (Employee ID or UUID)
            
        Returns:
            ChatSession instance
        """
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        session = ChatSession(session_id=session_id)
        self._sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Retrieve an existing session.
        
        Args:
            session_id: Unique identifier
            
        Returns:
            ChatSession or None if not found
        """
        self._cleanup_expired_sessions()
        return self._sessions.get(session_id)
    
    def get_or_create_session(self, session_id: str) -> ChatSession:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Unique identifier
            
        Returns:
            ChatSession instance
        """
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        return session
    
    def add_user_message(self, session_id: str, message: str) -> ChatSession:
        """
        Add a user message to a session.
        
        Args:
            session_id: Unique identifier
            message: User message content
            
        Returns:
            Updated ChatSession
        """
        session = self.get_or_create_session(session_id)
        session.add_user_message(message)
        return session
    
    def add_ai_message(self, session_id: str, message: str) -> ChatSession:
        """
        Add an AI message to a session.
        
        Args:
            session_id: Unique identifier
            message: AI message content
            
        Returns:
            Updated ChatSession
        """
        session = self.get_or_create_session(session_id)
        session.add_ai_message(message)
        return session
    
    def get_context(self, session_id: str, last_n: Optional[int] = None) -> str:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Unique identifier
            last_n: Optional - only get last N messages
            
        Returns:
            Formatted context string
        """
        session = self.get_session(session_id)
        if session is None:
            return ""
        
        if last_n:
            messages = session.get_last_n_messages(last_n)
            context_lines = []
            for msg in messages:
                role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                context_lines.append(f"{role}: {msg.content}")
            return "\n".join(context_lines)
        
        return session.get_context_string()
    
    def get_extracted_info(self, session_id: str) -> Dict[str, Any]:
        """
        Extract structured information from session.
        
        Args:
            session_id: Unique identifier
            
        Returns:
            Dictionary of extracted information
        """
        session = self.get_session(session_id)
        if session is None:
            return {}
        
        return session.get_extracted_info()
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Unique identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def clear_all_sessions(self) -> None:
        """Clear all sessions (use with caution)."""
        self._sessions.clear()
    
    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions based on TTL."""
        now = datetime.now()
        expired = []
        
        for session_id, session in self._sessions.items():
            age_minutes = (now - session.last_accessed).total_seconds() / 60
            if age_minutes > self._session_ttl_minutes:
                expired.append(session_id)
        
        for session_id in expired:
            del self._sessions[session_id]
    
    def get_session_count(self) -> int:
        """Get the number of active sessions."""
        self._cleanup_expired_sessions()
        return len(self._sessions)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a session.
        
        Args:
            session_id: Unique identifier
            
        Returns:
            Dictionary with session info or None
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "message_count": len(session.messages),
            "metadata": session.metadata,
        }


# Global instance
_memory_manager: Optional[SessionMemoryManager] = None


def get_memory_manager(session_ttl_minutes: int = 120) -> SessionMemoryManager:
    """
    Get or create the global session memory manager.
    
    Args:
        session_ttl_minutes: Time-to-live for sessions (minutes)
        
    Returns:
        SessionMemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = SessionMemoryManager(session_ttl_minutes=session_ttl_minutes)
    return _memory_manager
