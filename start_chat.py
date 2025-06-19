#!/usr/bin/env python3
"""
Quick Start Script for Obsidian Graph RAG Chat
==============================================

This script provides a quick way to start the chat interface with
your OpenAI API key without setting environment variables.

Usage:
    python start_chat.py

Features:
- Prompts for OpenAI API key
- Loads existing cached data if available
- Starts the chat interface directly
- Handles common startup issues

Author: Assistant
"""

import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from graphrag import GraphRAGConfig, ObsidianGraphRAG


def print_header():
    """Print startup header."""
    print("\n" + "🧠" + " " * 20 + "OBSIDIAN AI LIBRARIAN" + " " * 20 + "🧠")
    print("=" * 68)
    print("Welcome to your personal AI knowledge assistant!")
    print("This system will help you explore and understand your Obsidian notes.")
    print("=" * 68)


def get_api_key():
    """Get OpenAI API key from environment or user."""
    print("\n🔑 OpenAI API Key")
    print("-" * 20)
    
    # Check if already set in environment (.env file)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and api_key.startswith('sk-'):
        print("✓ Found API key in .env file")
        return api_key
    
    print("To use the AI features, you need an OpenAI API key.")
    print("You can get one from: https://platform.openai.com/api-keys")
    print()
    print("💡 Tip: Add it to your .env file as OPENAI_API_KEY=your-key-here")
    print()
    
    # Prompt user for API key if not in .env
    while True:
        api_key = input("Please enter your OpenAI API key (or 'skip' to continue without AI): ").strip()
        
        if api_key.lower() == 'skip':
            print("⚠️  Continuing without AI features - you can only browse the graph structure.")
            return None
        
        if api_key and api_key.startswith('sk-'):
            return api_key
        
        print("❌ Invalid API key format. API keys should start with 'sk-'")
        print("Please try again or type 'skip' to continue without AI features.")


def get_interface_choice():
    """Get user's choice for interface type."""
    print("\n🎨 Choose Your Interface")
    print("-" * 30)
    print("Select how you'd like to interact with your AI librarian:")
    print()
    print("  1. 🌐 Web Interface (Recommended)")
    print("     • Beautiful, modern design")
    print("     • Works in your browser")
    print("     • Dark/light themes")
    print("     • Source citations")
    print("     • Easy to use")
    print()
    print("  2. 💻 Command Line Interface")
    print("     • Text-based chat")
    print("     • Works in terminal")
    print("     • Simple and fast")
    print()
    
    while True:
        choice = input("Select interface (1 for Web, 2 for CLI): ").strip()
        
        if choice in ['1', 'web', 'w']:
            return 'web'
        elif choice in ['2', 'cli', 'c']:
            return 'cli'
        else:
            print("Please enter 1 (Web) or 2 (CLI)")


def start_web_interface(config: GraphRAGConfig) -> None:
    """Start the web interface."""
    try:
        # Import web chat (done here to avoid import issues if Flask not available)
        from web_chat import WebChatServer
        
        print("\n🌐 Starting Web Interface...")
        print("-" * 30)
        
        # Initialize Graph RAG system
        graph_rag = ObsidianGraphRAG(config)
        graph_rag.initialize_system()
        
        print(f"\n📊 System Ready!")
        print(f"   📝 Documents: {len(graph_rag.documents):,}")
        print(f"   🕸️  Graph edges: {graph_rag.knowledge_graph.number_of_edges():,}")
        
        # Create and start web server with config settings
        web_server = WebChatServer(config, graph_rag)
        web_server.run(
            host=config.web_host,
            port=config.web_port,
            auto_open=config.auto_open_browser
        )
        
    except ImportError as e:
        print(f"\n❌ Web interface not available: {e}")
        print("💡 Install Flask: pip install flask")
        print("🔄 Falling back to CLI interface...")
        start_cli_interface(config)
    except Exception as e:
        print(f"\n❌ Web interface error: {e}")
        print("🔄 Falling back to CLI interface...")
        start_cli_interface(config)


def start_cli_interface(config: GraphRAGConfig) -> None:
    """Start the CLI interface."""
    print("\n💻 Starting CLI Interface...")
    print("-" * 30)
    
    # Initialize Graph RAG system
    graph_rag = ObsidianGraphRAG(config)
    graph_rag.initialize_system()
    
    print(f"\n📊 System Ready!")
    print(f"   📝 Documents: {len(graph_rag.documents):,}")
    print(f"   🕸️  Graph edges: {graph_rag.knowledge_graph.number_of_edges():,}")
    
    # Start chat session
    print("\n🤖 Starting AI Chat Interface...")
    print("You can now ask questions about your notes!")
    graph_rag.start_chat_session()


def main():
    """Main startup function."""
    try:
        print_header()
        
        # Initialize configuration (loads from .env file)
        config = GraphRAGConfig()
        
        # Get API key (from .env or user input)
        api_key = get_api_key()
        
        # Override config if user provided a different key
        if api_key:
            config.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key  # Set in environment for other components
        
        # Configure system
        print("\n🔧 System Configuration")
        print("-" * 25)
        print(f"📁 Vault: {Path(config.vault_path).name}")
        print(f"💾 Cache: {config.cache_dir}")
        print(f"🌐 Web: {config.web_host}:{config.web_port}")
        
        # Start appropriate interface based on API key availability
        if config.openai_api_key:
            # Get interface choice
            interface_choice = get_interface_choice()
            
            if interface_choice == 'web':
                start_web_interface(config)
            else:
                start_cli_interface(config)
        else:
            print("\n📖 Available Options:")
            print("   1. Run 'python check_system.py' to diagnose any issues")
            print("   2. Run 'python graph3d_launcher.py' for 3D graph visualization")
            print("   3. Add your OpenAI API key to the .env file and restart")
            print("\n💡 To enable AI features, get an API key from: https://platform.openai.com/api-keys")
            print("   Then add it to your .env file: OPENAI_API_KEY=your-key-here")
        
    except KeyboardInterrupt:
        print("\n\n👋 Startup cancelled. Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Startup Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Run 'python check_system.py' to diagnose issues")
        print("   2. Check that your vault path is correct in .env file")
        print("   3. Ensure you have the required dependencies installed")
        sys.exit(1)


if __name__ == "__main__":
    main() 