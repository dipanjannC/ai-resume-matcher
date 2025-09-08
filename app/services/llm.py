from typing import Optional
from app.core.config import settings
from app.core.logging import get_logger

from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
import os

from dotenv import load_dotenv
load_dotenv()

logger = get_logger(__name__)


class LLMService:
    """Service for managing different LLM providers"""
    
    def __init__(self):
        self.openai_model = "gpt-3.5-turbo"
    
    def get_openai(self, model_name: Optional[str] = None):
        """Get OpenAI LLM instance"""
        try:
            
            
            model = model_name or self.openai_model
            
            llm = ChatOpenAI(
                name=model,
                temperature=0.7,
                api_key=settings.OPENAI_API_KEY
            )
            
            logger.info("Created OpenAI LLM", model=model)
            return llm
            
        except Exception as e:
            logger.error("Failed to create OpenAI LLM", error=str(e))
            raise
    
    def get_groq(self, model_name: Optional[str] = None):
        """Get Groq LLM instance - optional provider"""
        try:
        
            model = model_name or "mixtral-8x7b-32768"
            llm = ChatGroq(
                model=model,
                temperature=0.3,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=os.getenv("GROQ_API_KEY"),
            )

            return llm

        except Exception as e:
            logger.error("Failed to create Groq LLM", error=str(e))
            return None
    
    def get_mistral(self, model_name: Optional[str] = None):
        """Get Mistral LLM instance - optional provider"""
        try:
            # This is optional since we removed langchain_mistralai dependency
            llm = ChatMistralAI(
            model=model_name,
            temperature=0.3,
            max_retries=2,
            max_tokens=256,
            api_key=os.getenv("MISTRAL_API_KEY"),
            )
            return llm
        
            
        except Exception as e:
            logger.error("Failed to create Mistral LLM", error=str(e))
            return None


# Create global instance
llm_service = LLMService()
