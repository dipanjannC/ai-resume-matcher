#!/usr/bin/env python3
"""
Quick test script to diagnose LangChain agents initialization issues
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_imports():
    """Test imports step by step"""
    print("1. Testing basic imports...")
    
    try:
        from app.core.config import settings
        print("✅ Config import successful")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from app.services.llm import LLMService
        print("✅ LLM service import successful")
    except Exception as e:
        print(f"❌ LLM service import failed: {e}")
        return False
    
    print("2. Testing API key...")
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print(f"✅ GROQ_API_KEY found (length: {len(groq_key)})")
    else:
        print("❌ GROQ_API_KEY not found")
        return False
    
    print("3. Testing LLM service initialization...")
    try:
        llm_service = LLMService()
        print("✅ LLM service created")
    except Exception as e:
        print(f"❌ LLM service creation failed: {e}")
        return False
    
    print("4. Testing Groq LLM creation...")
    try:
        llm = llm_service.get_groq()
        print("✅ Groq LLM created successfully")
    except Exception as e:
        print(f"❌ Groq LLM creation failed: {e}")
        return False
    
    print("5. Testing LangChain agents...")
    try:
        from app.services.langchain_agents import LangChainAgents
        agents = LangChainAgents()
        print("✅ LangChain agents created successfully")
    except Exception as e:
        print(f"❌ LangChain agents creation failed: {e}")
        return False
    
    print("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
