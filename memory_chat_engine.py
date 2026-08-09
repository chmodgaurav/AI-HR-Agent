"""
Memory-Integrated Chat Engine

Integrates conversation memory with LangChain for multi-turn conversations.
Provides context-aware responses and tool integration.
"""

from typing import Any, Dict, List, Optional, Union
from conversation_memory import SessionMemoryManager, get_memory_manager
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_core.tools import Tool
from langchain.tools import tool
import json


class MemoryChatEngine:
    """
    Chat engine that integrates conversational memory with LLM.
    
    Features:
    - Multi-turn conversations with context
    - Tool integration with context from memory
    - Session isolation
    - Automatic context inclusion in prompts
    """
    
    def __init__(
        self,
        llm: ChatOllama,
        memory_manager: Optional[SessionMemoryManager] = None,
        vector_db: Optional[Chroma] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Initialize the chat engine.
        
        Args:
            llm: LangChain ChatOllama instance
            memory_manager: SessionMemoryManager instance
            vector_db: Chroma vector database for RAG
            tools: List of tools available to the LLM
        """
        self.llm = llm
        self.memory_manager = memory_manager or get_memory_manager()
        self.vector_db = vector_db
        self.tools = tools or []
        
        # Bind tools to LLM if provided
        if self.tools:
            self.llm_with_tools = llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = llm
    
    def chat(
        self,
        session_id: str,
        user_message: str,
        include_context: bool = True,
        retrieve_from_kb: bool = True,
        context_window: int = 10,  # Last N messages
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            session_id: Unique session identifier (Employee ID)
            user_message: User's input message
            include_context: Include previous conversation history
            retrieve_from_kb: Retrieve relevant documents from knowledge base
            context_window: Number of previous messages to include
            
        Returns:
            Dictionary with response, context, and metadata
        """
        # Add user message to memory
        self.memory_manager.add_user_message(session_id, user_message)
        
        # Build the prompt
        prompt_parts = []
        
        # System prompt
        system_prompt = """You are an HR Assistant. You help employees with HR-related questions and processes.

Your responsibilities:
1. Answer questions about HR policies using available documents
2. Help employees complete HR workflows (like leave applications)
3. Remember previous messages in the conversation and use that context
4. Collect required information gradually in multi-turn conversations
5. Use available tools to process requests when needed

Be helpful, professional, and concise."""
        
        prompt_parts.append(system_prompt)
        
        # Add conversation context
        if include_context:
            context = self.memory_manager.get_context(session_id, last_n=context_window)
            if context:
                prompt_parts.append(f"\n\nPrevious conversation:\n{context}")
        
        # Add knowledge base context if enabled
        kb_context = ""
        if retrieve_from_kb and self.vector_db:
            docs = self.vector_db.similarity_search(user_message, k=4)
            if docs:
                kb_context = "\n\n".join(d.page_content for d in docs)
                prompt_parts.append(f"\n\nRelevant HR Information:\n{kb_context}")
        
        # Add current user message
        prompt_parts.append(f"\n\nUser: {user_message}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Invoke LLM
        try:
            response = self.llm_with_tools.invoke(full_prompt)
            
            # Extract response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Check if tools were called
            tool_calls = []
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_calls = response.tool_calls
            
            # Add AI response to memory
            self.memory_manager.add_ai_message(session_id, response_content)
            
            return {
                "status": "success",
                "response": response_content,
                "session_id": session_id,
                "tool_calls": tool_calls,
                "context": {
                    "previous_messages": self.memory_manager.get_context(session_id, last_n=5),
                    "extracted_info": self.memory_manager.get_extracted_info(session_id),
                    "kb_context": kb_context,
                },
                "message_count": len(self.memory_manager.get_session(session_id).get_messages()),
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
            }
    
    def chat_with_qa(
        self,
        session_id: str,
        user_message: str,
        include_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a Q&A message with knowledge base retrieval.
        
        This is optimized for document-based questions.
        
        Args:
            session_id: Session identifier
            user_message: User's question
            include_context: Include conversation history
            
        Returns:
            Response dictionary
        """
        if not self.vector_db:
            return {
                "status": "error",
                "error": "Knowledge base not configured",
            }
        
        # Add message to memory
        self.memory_manager.add_user_message(session_id, user_message)
        
        # Retrieve relevant documents
        docs = self.vector_db.similarity_search(user_message, k=4)
        kb_context = "\n\n".join(d.page_content for d in docs)
        
        # Build prompt
        prompt = f"""You are an HR Assistant.

Answer ONLY from the provided context.

If the answer isn't present, say: "I couldn't find that information."

Knowledge Base Context:
{kb_context}

Question: {user_message}

Provide a clear, concise answer."""
        
        try:
            response = self.llm.invoke(prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Add to memory
            self.memory_manager.add_ai_message(session_id, response_content)
            
            return {
                "status": "success",
                "response": response_content,
                "session_id": session_id,
                "context": self.memory_manager.get_context(session_id, last_n=5),
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id,
            }
    
    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get the current session context without generating a response.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dictionary with session context
        """
        return {
            "session_info": self.memory_manager.get_session_info(session_id),
            "context": self.memory_manager.get_context(session_id),
            "extracted_info": self.memory_manager.get_extracted_info(session_id),
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear a session's history.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful
        """
        return self.memory_manager.delete_session(session_id)
    
    def extract_info_for_tool(
        self,
        session_id: str,
        required_fields: List[str],
    ) -> Dict[str, Any]:
        """
        Extract required information from conversation for tool invocation.
        
        Args:
            session_id: Session identifier
            required_fields: List of field names required by the tool
            
        Returns:
            Dictionary with extracted information
        """
        session = self.memory_manager.get_session(session_id)
        if not session:
            return {}
        
        # Get all messages as text
        context = session.get_context_string()
        extracted = session.get_extracted_info()
        
        # For now, return basic extraction
        # In production, you'd use more sophisticated NER/information extraction
        return extracted
    
    def get_missing_info(
        self,
        session_id: str,
        required_fields: List[str],
    ) -> Dict[str, str]:
        """
        Identify missing required information.
        
        Args:
            session_id: Session identifier
            required_fields: List of required field names
            
        Returns:
            Dictionary mapping missing fields to human-readable prompts
        """
        extracted = self.extract_info_for_tool(session_id, required_fields)
        
        missing = {}
        field_prompts = {
            "employee_id": "What is your employee ID?",
            "leave_type": "What type of leave are you applying for? (Casual, Sick, Earned, Maternity, Paternity, Bereavement)",
            "start_date": "What is the start date of your leave? (YYYY-MM-DD)",
            "end_date": "What is the end date of your leave? (YYYY-MM-DD)",
            "reason": "What is the reason for your leave?",
        }
        
        for field in required_fields:
            if field not in extracted or not extracted[field]:
                missing[field] = field_prompts.get(field, f"Please provide {field}")
        
        return missing


class MultiTurnLeaveHandler:
    """
    Specialized handler for multi-turn leave application workflow.
    Collects information gradually across multiple turns.
    """
    
    REQUIRED_FIELDS = ["employee_id", "leave_type", "start_date", "end_date", "reason"]
    
    def __init__(
        self,
        chat_engine: MemoryChatEngine,
        apply_leave_tool: Any,  # The apply_leave_fn
    ):
        """
        Initialize the leave handler.
        
        Args:
            chat_engine: MemoryChatEngine instance
            apply_leave_tool: Function to apply leave
        """
        self.chat_engine = chat_engine
        self.apply_leave_tool = apply_leave_tool
    
    def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Process a user message for leave application.
        
        If all required information is collected, applies for leave.
        Otherwise, asks for missing information.
        
        Args:
            session_id: Session identifier
            user_message: User's message
            
        Returns:
            Response dictionary
        """
        # Add to memory
        self.chat_engine.memory_manager.add_user_message(session_id, user_message)
        
        # Extract available information
        extracted = self.chat_engine.memory_manager.get_extracted_info(session_id)
        
        # Check for missing fields
        missing = self.chat_engine.get_missing_info(session_id, self.REQUIRED_FIELDS)
        
        if missing:
            # Ask for next missing field
            next_field = list(missing.keys())[0]
            prompt = f"I need some more information to process your leave request.\n\n{missing[next_field]}"
            
            self.chat_engine.memory_manager.add_ai_message(session_id, prompt)
            
            return {
                "status": "collecting",
                "response": prompt,
                "session_id": session_id,
                "missing_fields": missing,
                "collected_fields": extracted,
            }
        
        else:
            # All information collected, apply for leave
            try:
                result = self.apply_leave_tool(
                    employee_id=extracted.get("employee_id", ""),
                    leave_type=extracted.get("leave_type", ""),
                    start_date=extracted.get("start_date", ""),
                    end_date=extracted.get("end_date", ""),
                    reason=extracted.get("reason", ""),
                )
                
                response = f"Your leave request has been submitted. Status: {result.get('status', 'Unknown')}"
                self.chat_engine.memory_manager.add_ai_message(session_id, response)
                
                return {
                    "status": "success",
                    "response": response,
                    "session_id": session_id,
                    "tool_result": result,
                }
            
            except Exception as e:
                error_msg = f"Error processing leave request: {str(e)}"
                self.chat_engine.memory_manager.add_ai_message(session_id, error_msg)
                
                return {
                    "status": "error",
                    "response": error_msg,
                    "session_id": session_id,
                    "error": str(e),
                }
