from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class MemoryStore(ABC):
    """Abstract base class for memory stores."""
    
    @abstractmethod
    def add(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Add messages to memory."""
        pass
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Any]:
        """Search memory for relevant context."""
        pass
    
    @abstractmethod
    def get_all(self, **kwargs) -> List[Any]:
        """Get all memory items."""
        pass
