#!/usr/bin/env python3
"""
Test RAG system with small vault
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_small_vault():
    """Test with small 3-note vault"""
    print("="*70)
    print("TESTING RAG SYSTEM WITH SMALL VAULT")
    print("="*70)

    from src.simple_raganything import SimpleRAGAnything

    # Use small test vault
    vault_path = "./test_vault"
    working_dir = "./test_rag_storage"

    print(f"\nVault: {vault_path}")
    print(f"Storage: {working_dir}")

    # Initialize
    rag = SimpleRAGAnything(vault_path, working_dir)

    # Delete existing database
    if rag._detect_existing_database():
        print("\nDeleting existing test database...")
        rag._delete_existing_database()

    # Initialize RAG
    print("\nInitializing RAG system...")
    await rag.initialize()

    # Process vault
    print("\nProcessing vault...")
    await rag.process_vault()

    # Test query
    print("\n" + "="*70)
    print("TESTING QUERY")
    print("="*70)

    question = "What are the main topics covered in these notes?"
    print(f"\nQuestion: {question}")

    result = await rag.query(question)
    print(f"\nAnswer:\n{result}")

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_small_vault())