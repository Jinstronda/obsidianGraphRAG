#!/usr/bin/env python3
"""
Test full RAG-Anything workflow with small document using Gemini
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from simple_raganything import SimpleRAGAnything

async def test_small_document():
    """Test with a small synthetic document"""

    print("="*70)
    print("SMALL DOCUMENT EXTRACTION TEST WITH GEMINI")
    print("="*70)

    # Create a small test vault directory
    test_vault_path = "./test_vault"
    test_working_dir = "./test_rag_storage"

    # Create test vault
    os.makedirs(test_vault_path, exist_ok=True)

    # Create a small test document
    test_doc = """# Machine Learning Basics

## Introduction
Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed.

## Key Concepts
Neural networks are computational models inspired by biological neural systems. Deep learning uses multiple layers of neural networks to process complex patterns.

## Applications
Python is the primary programming language for machine learning. TensorFlow and PyTorch are popular frameworks built with Python.

## Conclusion
The field continues to evolve with new techniques and applications emerging regularly.
"""

    # Write test document
    test_file = os.path.join(test_vault_path, "ml_basics.md")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_doc)

    print(f"\n[1/5] Created test vault: {test_vault_path}")
    print(f"       Test document: ml_basics.md ({len(test_doc)} characters)")

    # Initialize RAG-Anything
    print(f"\n[2/5] Initializing RAG-Anything with Gemini...")
    rag = SimpleRAGAnything(
        vault_path=test_vault_path,
        working_dir=test_working_dir
    )

    await rag.initialize()

    # Process the vault
    print(f"\n[3/5] Processing test document...")
    stats = await rag.process_vault()

    print(f"\n[4/5] Extraction Results:")
    print("-" * 70)
    print(f"Files processed: {stats.get('files_processed', 0)}")
    print(f"Total chunks: {stats.get('total_chunks', 0)}")
    print(f"Processing time: {stats.get('total_time_seconds', 0):.1f}s")

    # Test query
    print(f"\n[5/5] Testing query...")
    test_query = "What is the relationship between Python and machine learning?"

    result = await rag.query(test_query, mode="hybrid")

    print("\nQuery Result:")
    print("=" * 70)
    print(result)
    print("=" * 70)

    # Cleanup instructions
    print("\n" + "="*70)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nTest files created:")
    print(f"  - Vault: {test_vault_path}")
    print(f"  - Storage: {test_working_dir}")
    print(f"\nTo cleanup: rm -rf {test_vault_path} {test_working_dir}")

    return True


async def main():
    try:
        success = await test_small_document()

        if success:
            print("\n" + "="*70)
            print("READY FOR FULL VAULT EXTRACTION!")
            print("="*70)
            print("\nGemini integration is working correctly.")
            print("You can now run: python run_obsidian_raganything.py")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
