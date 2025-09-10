#!/usr/bin/env python3
"""
Launch script for AI Resume Matcher Streamlit application
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit application"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print("❌ streamlit_app.py not found!")
        sys.exit(1)
    
    print("🚀 Launching AI Resume Matcher Streamlit App...")
    print("📝 Make sure you have set your LLM API key ...")
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(streamlit_app),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "light"
        ], check=True)
    
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        sys.exit(1)
    
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
