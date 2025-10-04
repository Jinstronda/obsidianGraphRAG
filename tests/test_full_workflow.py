#!/usr/bin/env python3
"""
Test complete RAG workflow with test vault
"""
import os
import sys
import asyncio
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_raganything import SimpleRAGAnything

async def test_full_workflow():
    """Test complete RAG workflow"""
    print("="*70)
    print("TESTING COMPLETE RAG WORKFLOW")
    print("="*70)

    # Use test vault
    vault_path = r"C:\Users\joaop\Documents\Hobbies\Obsidian Rag\test_vault"
    working_dir = "./test_rag_storage"

    try:
        # Step 1: Clean up any existing test database
        print("\n[STEP 1] Cleaning up old test database...")
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
            print(f"   Deleted: {working_dir}")
        os.makedirs(working_dir, exist_ok=True)
        print(f"   Created: {working_dir}")

        # Step 2: Initialize RAG
        print("\n[STEP 2] Initializing RAG system...")
        rag = SimpleRAGAnything(vault_path, working_dir)
        await rag.initialize()

        if rag.rag_anything is None:
            print("\n[FAILED] RAG instance not created!")
            return False

        print(f"\n   [OK] RAG instance created")
        print(f"   Using existing DB: {rag.using_existing_db}")

        # Step 3: Process vault
        print("\n[STEP 3] Processing test vault...")
        await rag.process_vault()

        # Step 4: Test query
        print("\n[STEP 4] Testing query...")
        result = await rag.query("What are the main topics?", mode="hybrid")

        print("\n" + "="*70)
        print("QUERY RESULT:")
        print("="*70)
        print(result)
        print("="*70)

        if "Error" in result:
            print("\n[FAILED] Query returned an error!")
            return False

        print("\n[SUCCESS] Complete workflow passed!")
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_full_workflow())
    sys.exit(0 if result else 1)
