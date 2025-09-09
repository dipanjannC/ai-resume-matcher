from typing import Optional
from app.core.config import settings
from app.core.logging import get_logger

from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)


class LLMService:
    """Service for managing different LLM providers"""
    
    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
    
    def get_groq(self, model_name: Optional[str] = None):
        """Get Groq LLM instance - default provider"""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
        
            model = model_name or "gemma2-9b-it"
            
            # Set environment variable for langchain-groq to pick up
            os.environ["GROQ_API_KEY"] = groq_api_key
            
            llm = ChatGroq(
                model=model,
                temperature=0.3,
            )

            logger.info("Created Groq LLM", model=model)
            return llm

        except Exception as e:
            logger.error("Failed to create Groq LLM", error=str(e))
            raise


# Create global instance
llm_service = LLMService()
