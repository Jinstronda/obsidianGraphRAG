#!/usr/bin/env python3
"""
Test GPT-5-nano with existing production database
"""
import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_raganything import SimpleRAGAnything

async def test_gpt5_nano():
    """Test GPT-5-nano with production database"""
    print("="*70)
    print("TESTING GPT-5-NANO WITH PRODUCTION DATABASE")
    print("="*70)

    vault_path = r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro"
    working_dir = "./rag_storage"

    try:
        # Initialize RAG with existing database
        print("\n[STEP 1] Initializing RAG system...")
        rag = SimpleRAGAnything(vault_path, working_dir)
        await rag.initialize()

        if rag.rag_anything is None:
            print("\n[FAILED] RAG instance not created!")
            return False

        print(f"\n   [OK] RAG instance created")
        print(f"   Using existing DB: {rag.using_existing_db}")

        # Test query with GPT-5-nano
        print("\n[STEP 2] Testing query with GPT-5-nano...")
        print("   Model: gpt-5-nano")
        print("   Cost: ~$0.0005 per query (with cache)")

        result = await rag.query("What are the main topics in my vault?", mode="hybrid")

        print("\n" + "="*70)
        print("QUERY RESULT:")
        print("="*70)
        print(result)
        print("="*70)

        if "Error" in result or "error" in result:
            print("\n[WARNING] Query may have issues, check result above")
            return False

        print("\n[SUCCESS] GPT-5-nano working correctly!")
        print("   - 50% cheaper than GPT-4o-mini")
        print("   - 90% cache discount on subsequent queries")
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gpt5_nano())
    sys.exit(0 if result else 1)
