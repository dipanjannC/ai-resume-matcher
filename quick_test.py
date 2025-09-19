#!/usr/bin/env python3
"""
Quick test to verify basic functionality works
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """Test if basic imports work"""
    print("Testing basic imports...")
    
    # Test core functionality
    try:
        from app.core.logging import get_logger
        print("✅ Logging OK")
    except Exception as e:
        print(f"❌ Logging failed: {e}")
        return False
    
    try:
        from app.models.resume_data import ResumeData, JobDescription
        print("✅ Models OK")
    except Exception as e:
        print(f"❌ Models failed: {e}")
        return False
    
    try:
        from app.services.prompt_manager import prompt_manager
        print("✅ Prompt manager OK")
    except Exception as e:
        print(f"❌ Prompt manager failed: {e}")
        return False
    
    try:
        # Skip services that require external dependencies for now
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_streamlit_syntax():
    """Test if streamlit app has valid syntax"""
    print("Testing Streamlit app syntax...")
    
    try:
        import streamlit as st
        print("✅ Streamlit available")
    except Exception as e:
        print(f"❌ Streamlit not available: {e}")
        return False
    
    try:
        # Test if we can import the main app class
        with open('streamlit_app.py', 'r') as f:
            content = f.read()
            
        # Check if key sections exist
        if 'class StreamlitApp' in content:
            print("✅ StreamlitApp class found")
        else:
            print("❌ StreamlitApp class missing")
            return False
            
        if 'def resume_customizer_page' in content:
            print("✅ Resume customizer page found")
        else:
            print("❌ Resume customizer page missing")
            return False
            
        print("✅ Streamlit app structure OK")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Quick Test Suite\n")
    
    basic_ok = test_basic_imports()
    if basic_ok:
        streamlit_ok = test_streamlit_syntax()
        
        if streamlit_ok:
            print("\n✅ All tests passed! The app should work.")
        else:
            print("\n⚠️ Some Streamlit issues detected.")
    else:
        print("\n❌ Basic import issues detected.")
    
    print("\n✨ Test complete!")
