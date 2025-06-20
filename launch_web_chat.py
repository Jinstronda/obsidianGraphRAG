#!/usr/bin/env python3
"""
Quick Launch Script for Obsidian AI Librarian Web Chat
=====================================================

Simple script to launch the enhanced web chat interface with one command.
Automatically opens in browser with all enhanced features enabled.

Usage:
    python launch_web_chat.py
    
Features:
- ✅ Enhanced browser opening (multiple strategies)
- ⚙️ AI model and prompt customization 
- 🔥 CUDA acceleration (GPU-powered)
- 🧠 Advanced Graph RAG features
- 🌐 Beautiful modern web interface

"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Launch the enhanced web chat interface."""
    try:
        # Load environment variables FIRST before any imports
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✅ Environment variables loaded from .env")
        except ImportError:
            print("⚠️  dotenv not installed, trying without .env file")
        
        print("🚀 Launching Obsidian AI Librarian...")
        print("🔥 Enhanced features: CUDA, Hybrid Search, Community Summaries")
        print("⚙️  AI Model & Prompt Customization Available")
        print("🌐 Beautiful Web Interface")
        print()
        
        # Import and run web chat (after dotenv is loaded)
        from web_chat import main as web_main
        web_main()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct directory and dependencies are installed")
        print("Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 