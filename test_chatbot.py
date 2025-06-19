#!/usr/bin/env python3
"""
Simple test script for the Obsidian Graph RAG chatbot.

This script creates a minimal test setup to verify the chat functionality
works without processing the entire vault.
"""

import os
import asyncio
import sys
sys.path.append('.')
from graphrag import ObsidianGraphRAG, GraphRAGConfig

async def main():
    """Simple test of the chat interface."""
    
    print("🧪 Simple Graph RAG Chat Test")
    print("=" * 40)
    
    # Create test config
    config = GraphRAGConfig()
    # Vault path should be set via environment variable OBSIDIAN_VAULT_PATH
    if not config.vault_path:
        print("❌ Please set OBSIDIAN_VAULT_PATH environment variable or modify the vault_path in the config.")
        print("Example: set OBSIDIAN_VAULT_PATH=C:\\path\\to\\your\\vault")
        return
    
    # Set API key if available
    api_key = os.getenv("OPENAI_API_KEY") or input("Enter OpenAI API key: ").strip()
    if not api_key:
        print("❌ API key required for chat functionality")
        return
    
    config.openai_api_key = api_key
    
    print("🔧 Initializing minimal system...")
    
    # Initialize system
    graph_rag = ObsidianGraphRAG(config)
    
    # Check if we have cached data
    try:
        graph_rag.initialize_system()
        
        print(f"📊 System loaded:")
        print(f"   Documents: {len(graph_rag.documents)}")
        print(f"   Graph edges: {graph_rag.knowledge_graph.number_of_edges()}")
        
        if len(graph_rag.documents) > 0:
            print("\n🤖 Starting chat session...")
            graph_rag.start_chat_session()
        else:
            print("❌ No documents loaded")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 