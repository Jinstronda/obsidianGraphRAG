#!/usr/bin/env python3
"""
Debug Query Prompt - See what's being sent to Gemini
This helps diagnose why the LLM returns "I do not have enough information"
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


async def debug_query():
    """Debug a query to see the prompt being sent"""
    
    # Path to existing RAG storage
    working_dir = "./rag_storage"
    
    if not os.path.exists(working_dir):
        print(f"ERROR: RAG storage not found at {working_dir}")
        print("Please run the main system first to create the database")
        return
    
    print("="*70)
    print("DEBUG QUERY PROMPT")
    print("="*70)
    
    # Initialize RAG-Anything to access existing database
    from src.simple_raganything import SimpleRAGAnything
    
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    
    rag = SimpleRAGAnything(vault_path, working_dir)
    await rag.initialize()
    
    # Test query with only_need_prompt=True to see what's being sent
    test_question = "What are the top 10 topics in my notes?"
    
    print(f"\n[1/2] Testing query: {test_question}")
    print("\nRetrieving prompt that would be sent to LLM...")
    
    try:
        # Query with only_need_prompt to see the full prompt
        result = await rag.rag_anything.aquery(
            test_question,
            param=QueryParam(
                mode="hybrid",
                only_need_prompt=True  # This returns the prompt instead of calling LLM
            )
        )
        
        print("\n" + "="*70)
        print("PROMPT BEING SENT TO GEMINI:")
        print("="*70)
        print(result)
        print("="*70)
        
        # Check if "Source Data" appears in the prompt
        if "Source Data" in result or "source data" in result.lower():
            print("\n✓ 'Source Data' found in prompt - context is being passed")
        else:
            print("\n✗ 'Source Data' NOT found in prompt - THIS IS THE PROBLEM!")
            print("   The retrieved entities/relations aren't being included in the prompt")
        
        # Count the prompt length
        print(f"\nPrompt length: {len(result)} characters")
        print(f"Prompt lines: {len(result.split(chr(10)))} lines")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to get prompt: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Now try with only_need_context to see what context is retrieved
    print(f"\n[2/2] Checking retrieved context...")
    
    try:
        context_result = await rag.rag_anything.aquery(
            test_question,
            param=QueryParam(
                mode="hybrid",
                only_need_context=True  # This returns the retrieved context
            )
        )
        
        print("\n" + "="*70)
        print("RETRIEVED CONTEXT:")
        print("="*70)
        print(context_result)
        print("="*70)
        
        print(f"\nContext length: {len(context_result)} characters")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to get context: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function"""
    try:
        await debug_query()
        
        print("\n" + "="*70)
        print("DEBUG COMPLETE")
        print("="*70)
        print("\nNext steps:")
        print("1. Check if 'Source Data' appears in the prompt")
        print("2. Verify the context is being passed to the LLM function")
        print("3. If missing, the issue is in the prompt template or LLM function")
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
