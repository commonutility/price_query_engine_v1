"""
Conversation memory management for the Price Query Engine.
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, ClassVar

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import Field
from langchain.schema import messages_from_dict, messages_to_dict

class PersistentConversationMemory(ConversationBufferMemory):
    """A conversation memory that persists to a file."""
    
    # Define class fields
    memory_file: str = Field(default="")
    auto_save: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(
        self,
        memory_file: Optional[str] = None,
        auto_save: bool = True,
        **kwargs
    ):
        """Initialize the memory."""
        # Set default memory file
        if memory_file is None:
            memory_file = os.path.join(
                os.path.expanduser("~"),
                ".price_query_engine_memory.json"
            )
            
        # Initialize with fields
        super().__init__(memory_file=memory_file, auto_save=auto_save, **kwargs)
        
        # Load existing memory if it exists
        self._load_memory()
    
    def _load_memory(self) -> None:
        """Load memory from file if it exists."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    
                message_dicts = data.get("messages", [])
                messages = messages_from_dict(message_dicts)
                
                # Create a new history with these messages
                self.chat_memory = ChatMessageHistory(messages=messages)
                
                # Also store metadata
                self.metadata = data.get("metadata", {})
            except Exception as e:
                print(f"Error loading memory: {e}")
                # Start with a fresh memory if loading fails
                self.chat_memory = ChatMessageHistory()
                self.metadata = {
                    "created_at": datetime.now().isoformat()
                }
        else:
            # Initialize new memory
            self.chat_memory = ChatMessageHistory()
            self.metadata = {
                "created_at": datetime.now().isoformat()
            }
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        
        # Update the last updated timestamp
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Auto-save if enabled
        if self.auto_save:
            self.save_memory()
    
    def save_memory(self) -> None:
        """Save memory to file."""
        # Convert messages to a serializable format
        message_dicts = messages_to_dict(self.chat_memory.messages)
        
        # Prepare data structure
        data = {
            "metadata": self.metadata,
            "messages": message_dicts
        }
        
        # Save to file
        try:
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            with open(self.memory_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        # Save the cleared state if auto-save is enabled
        if self.auto_save:
            self.save_memory()

class QueryHistoryMemory:
    """Stores the history of queries and their results."""
    
    def __init__(self, history_file: str = None):
        """Initialize the query history memory."""
        self.history_file = history_file or os.path.join(
            os.path.expanduser("~"),
            ".price_query_engine_history.json"
        )
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file if it exists."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history: {e}")
                return []
        return []
    
    def add_query(self, query: str, parameters: Dict[str, Any], result: str) -> None:
        """Add a query to the history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "parameters": parameters,
            "result_summary": result[:200] + "..." if len(result) > 200 else result
        }
        self.history.append(entry)
        self._save_history()
    
    def get_recent_queries(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent queries."""
        return self.history[-n:] if self.history else []
    
    def search_queries(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for queries containing the search term."""
        return [
            entry for entry in self.history
            if search_term.lower() in entry["query"].lower()
        ]
    
    def _save_history(self) -> None:
        """Save history to file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def clear(self) -> None:
        """Clear history."""
        self.history = []
        self._save_history()
