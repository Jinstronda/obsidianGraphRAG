#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        import os
        import asyncio
        from typing import List, Dict, Any
        from dotenv import load_dotenv
        print("✅ Basic imports successful")
        
        # Test sentence transformers
        from sentence_transformers import SentenceTransformer
        print("✅ SentenceTransformers import successful")
        
        # Test RAG-Anything
        from raganything import RAGAnything, RAGAnythingConfig
        print("✅ RAG-Anything import successful")
        
        # Test our custom modules
        from obsidian_chunker import ObsidianChunker
        print("✅ ObsidianChunker import successful")
        
        from simple_raganything import SimpleRAGAnything
        print("✅ SimpleRAGAnything import successful")
        
        print("\n🎉 All imports successful!")
        print("✅ Ready to run RAG-Anything with EmbeddingGemma 308M")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   bash setup_conda_env.sh")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
