from typing import Any, Dict, List, Optional
import os
from app.services.memory.base import MemoryStore
from app.services.memory.mem0_store import Mem0Store
from app.services.memory.graphiti_store import GraphitiStore
from app.core.logging import get_logger

logger = get_logger(__name__)

class MemoryService:
    """Service for managing AI memory using configured provider."""
    
    def __init__(self):
        self.provider_name = os.getenv("MEMORY_PROVIDER", "mem0").lower()
        self.store: Optional[MemoryStore] = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize the memory store based on configuration."""
        if self.provider_name == "graphiti":
            self.store = GraphitiStore()
        else:
            self.store = Mem0Store()
            
        if not self.store:
            logger.warning(f"Failed to initialize memory provider: {self.provider_name}")

    def add_context(self, user_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Add context to memory."""
        if not self.store:
            return
            
        messages = [{"role": "user", "content": text}]
        if metadata:
            # Add metadata to message if supported by store
            pass
            
        self.store.add(messages, user_id=user_id)

    def get_context(self, user_id: str, query: str) -> List[Any]:
        """Retrieve relevant context."""
        if not self.store:
            return []
            
        return self.store.search(query, user_id=user_id)

# Global instance
memory_service = MemoryService()
