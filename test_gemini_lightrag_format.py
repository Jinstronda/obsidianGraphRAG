#!/usr/bin/env python3
"""
Test if Gemini can produce LightRAG's exact entity extraction format
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gemini_llm import gemini_complete_if_cache

API_KEY = "REDACTED"

async def test_lightrag_format():
    """Test LightRAG entity extraction format with Gemini"""

    print("="*70)
    print("TEST: GEMINI WITH LIGHTRAG ENTITY EXTRACTION FORMAT")
    print("="*70)

    # Sample text for extraction
    test_text = """
    Python is a high-level programming language. It is widely used for
    machine learning and data science. TensorFlow is a machine learning
    framework built with Python.
    """

    # LightRAG's EXACT extraction prompt format
    lightrag_prompt = f"""
-Target activity-
You are an intelligent assistant that helps extract entities and relationships from text.

-Goal-
Extract all entities and relationships from the provided text using the exact format specified below.

-Format Requirements-
1. Use <|#|> as the tuple delimiter to separate fields
2. Use <|COMPLETE|> to mark completion

3. Entity format:
entity<|#|>entity_name<|#|>entity_type<|#|>entity_description

4. Relationship format:
relation<|#|>source_entity<|#|>target_entity<|#|>relationship_keywords<|#|>relationship_description

5. Multiple keywords in relationship_keywords must be separated by commas (,)

6. End with <|COMPLETE|>

-Example Output-
entity<|#|>Python<|#|>language<|#|>High-level programming language
entity<|#|>Machine Learning<|#|>field<|#|>AI subdomain focusing on learning from data
relation<|#|>Python<|#|>Machine Learning<|#|>used-for,supports<|#|>Python is widely used for machine learning
<|COMPLETE|>

-Text to process-
{test_text}

-Output-
"""

    print("\n[1/3] Sending LightRAG format prompt to Gemini...")

    try:
        response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt=lightrag_prompt,
            api_key=API_KEY
        )

        print("\n[2/3] Gemini Response:")
        print("=" * 70)
        print(response)
        print("=" * 70)

        # Validate format
        print("\n[3/3] Format Validation:")
        print("-" * 70)

        has_entities = "entity<|#|>" in response
        has_relations = "relation<|#|>" in response
        has_complete = "<|COMPLETE|>" in response
        uses_tuple_delim = "<|#|>" in response

        print(f"Contains entity format: {has_entities}")
        print(f"Contains relation format: {has_relations}")
        print(f"Contains <|COMPLETE|>: {has_complete}")
        print(f"Uses <|#|> delimiter: {uses_tuple_delim}")

        if has_entities and has_complete and uses_tuple_delim:
            print("\n[SUCCESS] Gemini produced correct LightRAG format!")

            # Count extractions
            entity_count = response.count("entity<|#|>")
            relation_count = response.count("relation<|#|>")
            print(f"\nExtracted: {entity_count} entities, {relation_count} relations")

            return True
        else:
            print("\n[FAIL] Format does not match LightRAG requirements")
            return False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_lightrag_format()

    if success:
        print("\n" + "="*70)
        print("READY FOR PRODUCTION EXTRACTION!")
        print("="*70)
        print("\nGemini can now be used for entity extraction with LightRAG")
    else:
        print("\n" + "="*70)
        print("FORMAT MISMATCH - NEEDS ADJUSTMENT")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
