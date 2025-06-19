#!/usr/bin/env python3
"""
Load cached data and start chat interface directly.
"""

import os
import pickle
import json
from pathlib import Path
import sys
sys.path.append('.')

from graphrag import ObsidianGraphRAG, GraphRAGConfig, DataPersistenceManager

def main():
    """Load cached data and start chat."""
    
    print("🚀 Quick Chat Loader")
    print("=" * 30)
    
    # Setup config
    config = GraphRAGConfig()
    # Vault path should be set via environment variable OBSIDIAN_VAULT_PATH
    if not config.vault_path:
        print("❌ Please set OBSIDIAN_VAULT_PATH environment variable or modify the vault_path in the config.")
        print("Example: set OBSIDIAN_VAULT_PATH=C:\\path\\to\\your\\vault")
        return
    
    # Set API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ API key required")
        return
    
    config.openai_api_key = api_key
    
    # Initialize system
    graph_rag = ObsidianGraphRAG(config)
    
    # Check for cached data
    cache_dir = Path(config.cache_dir) / "processed_data"
    docs_file = cache_dir / "documents.pkl"
    graph_file = cache_dir / "knowledge_graph.gpickle"
    metadata_file = cache_dir / "metadata.json"
    
    if docs_file.exists() and graph_file.exists():
        print("📂 Found cached data, loading...")
        
        try:
            # Load documents
            with open(docs_file, 'rb') as f:
                graph_rag.documents = pickle.load(f)
            
            # Load graph  
            with open(graph_file, 'rb') as f:
                graph_rag.knowledge_graph = pickle.load(f)
            
            print(f"✅ Loaded {len(graph_rag.documents)} documents")
            print(f"✅ Loaded graph with {graph_rag.knowledge_graph.number_of_edges()} edges")
            
            # Start chat
            print("\n🤖 Starting chat...")
            graph_rag.start_chat_session()
            
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            
    else:
        print("❌ No cached data found. Run graphrag.py first to process your vault.")

if __name__ == "__main__":
    main() 