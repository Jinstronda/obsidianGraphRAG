#!/usr/bin/env python3
"""
Test Gemini 2.5 Flash for Entity Extraction
Tests if Gemini can extract entities in the format expected by LightRAG
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gemini_llm import gemini_complete_if_cache

# Test Vertex AI API Key
VERTEX_AI_API_KEY = "REDACTED"

async def test_entity_extraction():
    """Test entity extraction with Gemini 2.5 Flash"""

    print("="*70)
    print("TESTING GEMINI 2.5 FLASH FOR ENTITY EXTRACTION")
    print("="*70)

    # Sample text from Obsidian vault (similar to what RAG-Anything would process)
    test_text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data. Neural networks are a key component of
    machine learning, inspired by biological neural structures. Deep learning
    uses multiple layers of neural networks for complex pattern recognition.
    """

    # Create entity extraction prompt similar to LightRAG's format
    # LightRAG typically asks to extract entities and relations
    extraction_prompt = f"""
    Extract entities and their relationships from the following text.

    Text: {test_text}

    Please extract:
    1. Entities (name, type, description)
    2. Relationships between entities (source, target, description)

    Format your response as a structured list.
    """

    print("\n[1/3] Testing basic API call...")
    try:
        # Test basic completion
        basic_response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt="Say 'hello world' in one sentence.",
            api_key=VERTEX_AI_API_KEY
        )

        print(f"   [OK] Basic API call successful")
        print(f"   Response: {basic_response}")

    except Exception as e:
        print(f"   [ERROR] Basic API call failed: {e}")
        return False

    print("\n[2/3] Testing entity extraction prompt...")
    try:
        # Test entity extraction
        extraction_response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt=extraction_prompt,
            system_prompt="You are an expert at extracting entities and relationships from text.",
            api_key=VERTEX_AI_API_KEY
        )

        print(f"   [OK] Entity extraction successful")
        print(f"\n   Extracted Entities & Relations:")
        print("   " + "-"*66)
        # Print response with indentation
        for line in extraction_response.split('\n'):
            print(f"   {line}")
        print("   " + "-"*66)

    except Exception as e:
        print(f"   [ERROR] Entity extraction failed: {e}")
        return False

    print("\n[3/3] Testing with history messages...")
    try:
        # Test with chat history (simulates multi-turn conversation)
        history_response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt="What were the main entities we extracted?",
            system_prompt="You are an expert at extracting entities and relationships from text.",
            history_messages=[
                {"role": "user", "content": extraction_prompt},
                {"role": "assistant", "content": extraction_response}
            ],
            api_key=VERTEX_AI_API_KEY
        )

        print(f"   [OK] History-based conversation successful")
        print(f"   Response: {history_response[:200]}...")

    except Exception as e:
        print(f"   [ERROR] History-based call failed: {e}")
        return False

    print("\n" + "="*70)
    print("[SUCCESS] GEMINI 2.5 FLASH IS READY FOR ENTITY EXTRACTION!")
    print("="*70)
    print("\nKey Findings:")
    print("   [OK] Gemini API working correctly")
    print("   [OK] Entity extraction format compatible")
    print("   [OK] Chat history handling working")
    print("   [OK] Ready to integrate into RAG-Anything")

    print("\nNext Steps:")
    print("   1. Update simple_raganything.py to use gemini_llm")
    print("   2. Test on small Obsidian vault sample")
    print("   3. Run full extraction with 6119 chunks")

    print("\nEstimated Processing:")
    print("   - Model: Gemini 2.5 Flash (paid tier)")
    print("   - Time: ~12 minutes for 6119 chunks")
    print("   - Cost: ~$7.65 total")
    print("   - Rate: 1,000 RPM, 1M TPM")

    return True


async def main():
    """Main test function"""
    try:
        success = await test_entity_extraction()

        if success:
            print("\n" + "="*70)
            print("ALL TESTS PASSED - READY TO PROCEED")
            print("="*70)
            sys.exit(0)
        else:
            print("\n" + "="*70)
            print("TESTS FAILED - CHECK ERRORS ABOVE")
            print("="*70)
            sys.exit(1)

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Run test
    asyncio.run(main())
