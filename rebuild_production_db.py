#!/usr/bin/env python3
"""
Rebuild production database from scratch
"""
import os
import sys
import asyncio
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.simple_raganything import SimpleRAGAnything

async def rebuild_production():
    """Rebuild production database"""
    print("="*70)
    print("REBUILDING PRODUCTION DATABASE")
    print("="*70)

    # Production vault
    vault_path = r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro"
    working_dir = "./rag_storage"

    try:
        # Step 1: Delete incomplete database
        print("\n[STEP 1] Deleting incomplete database...")
        if os.path.exists(working_dir):
            shutil.rmtree(working_dir)
            print(f"   Deleted: {working_dir}")
        os.makedirs(working_dir, exist_ok=True)
        print(f"   Created fresh: {working_dir}")

        # Step 2: Initialize RAG
        print("\n[STEP 2] Initializing RAG system...")
        rag = SimpleRAGAnything(vault_path, working_dir)
        await rag.initialize()

        print(f"\n   [OK] RAG instance created")
        print(f"   Using existing DB: {rag.using_existing_db}")

        # Step 3: Process entire vault
        print("\n[STEP 3] Processing full Obsidian vault...")
        print("   This will take several minutes for large vaults...")
        await rag.process_vault()

        # Step 4: Test query
        print("\n[STEP 4] Testing query on production data...")
        result = await rag.query("What are the main topics in my vault?", mode="hybrid")

        print("\n" + "="*70)
        print("QUERY RESULT:")
        print("="*70)
        print(result)
        print("="*70)

        print("\n[SUCCESS] Production database rebuilt and tested!")
        return True

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(rebuild_production())
    sys.exit(0 if result else 1)
