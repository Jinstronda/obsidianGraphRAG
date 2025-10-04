#!/usr/bin/env python3
"""
Test if RAG initialization works properly
"""
import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_raganything import SimpleRAGAnything

async def test_initialization():
    """Test RAG initialization"""
    print("Testing RAG initialization...")

    vault_path = r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro"
    working_dir = "./rag_storage"

    try:
        # Initialize
        rag = SimpleRAGAnything(vault_path, working_dir)
        await rag.initialize()

        # Check if rag_anything was created
        if rag.rag_anything is None:
            print("\n[FAILED] self.rag_anything is still None!")
            return False
        else:
            print(f"\n[SUCCESS] self.rag_anything created: {type(rag.rag_anything)}")
            print(f"Using existing DB: {rag.using_existing_db}")
            return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_initialization())
    sys.exit(0 if result else 1)
