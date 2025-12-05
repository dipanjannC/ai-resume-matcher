from typing import Any, Dict, List, Optional
import os
from mem0 import Memory
from app.services.memory.base import MemoryStore
from app.core.logging import get_logger

logger = get_logger(__name__)

class Mem0Store(MemoryStore):
    """Mem0 implementation of MemoryStore."""
    
    def __init__(self):
        try:
            self.client = Memory()
            logger.info("Mem0 initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Mem0: {e}")
            self.client = None

    def add(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Add messages to memory."""
        if not self.client:
            return None
            
        try:
            user_id = kwargs.get("user_id", "default_user")
            # Mem0 expects messages or text. 
            # If messages is a list of dicts, we might need to format or pass directly.
            # Based on docs, .add(messages, user_id=...)
            return self.client.add(messages, user_id=user_id)
        except Exception as e:
            logger.error(f"Error adding to Mem0: {e}")
            return None

    def search(self, query: str, **kwargs) -> List[Any]:
        """Search memory."""
        if not self.client:
            return []
            
        try:
            user_id = kwargs.get("user_id", "default_user")
            return self.client.search(query, user_id=user_id)
        except Exception as e:
            logger.error(f"Error searching Mem0: {e}")
            return []

    def get_all(self, **kwargs) -> List[Any]:
        """Get all memories."""
        if not self.client:
            return []
            
        try:
            user_id = kwargs.get("user_id", "default_user")
            return self.client.get_all(user_id=user_id)
        except Exception as e:
            logger.error(f"Error getting all from Mem0: {e}")
            return []
