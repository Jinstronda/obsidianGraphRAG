#!/usr/bin/env python3
"""
Test what format Gemini returns for entity extraction
Compare with what LightRAG expects
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gemini_llm import gemini_complete_if_cache

VERTEX_AI_API_KEY = "REDACTED"

async def test_lightrag_format():
    """Test exact format LightRAG expects"""

    # Sample text chunk
    test_text = """
    Machine learning is a subset of artificial intelligence. Neural networks
    are key components of machine learning systems. Deep learning uses multiple
    layers of neural networks.
    """

    # This is the EXACT prompt format LightRAG uses for entity extraction
    # Copied from LightRAG source code
    lightrag_prompt = f"""
-Target activity-
You are an intelligent assistant that helps a human analyst to analyze claims against certain entities presented in a text document.

-Goal-
Given a text document that is potentially relevant to this activity, an entity specification, and a claim description, extract all entities that match the entity specification and all claims against those entities.

-Steps-
1. Extract all named entities that match the predefined entity specification. Entity specification can either be a list of entity names or a list of entity types.
2. For each entity identified in step 1, extract all claims associated with the entity. Claims need to match the specified claim description, and the entity should be the subject of the claim.
Format each claim as (<subject_entity>|<claim_property>|<claim_object>|<claim_description>|<claim_source_id>)

3. Return output in English as a single list of all the claims identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output {{#COMPLETE#}}

5. If you don't find any entities, return {{#NONE#}}

-Real Data-
######################
entity_spec: organization,person,geo,event
claim_description: relationships between entities
text: {test_text}
######################
output:
"""

    print("="*70)
    print("TESTING LIGHTRAG ENTITY EXTRACTION FORMAT")
    print("="*70)

    print("\n[1/2] Sending LightRAG-format prompt to Gemini...")
    print(f"\nPrompt preview:")
    print("-" * 70)
    print(lightrag_prompt[:500] + "...")
    print("-" * 70)

    try:
        response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt=lightrag_prompt,
            api_key=VERTEX_AI_API_KEY
        )

        print("\n[2/2] Gemini Response:")
        print("=" * 70)
        print(response)
        print("=" * 70)

        # Check for expected delimiters
        print("\n" + "="*70)
        print("FORMAT ANALYSIS")
        print("="*70)

        has_delimiter = "##" in response
        has_complete = "#COMPLETE#" in response or "COMPLETE" in response
        has_none = "#NONE#" in response or "NONE" in response
        has_pipes = "|" in response

        print(f"Contains ## delimiter: {has_delimiter}")
        print(f"Contains #COMPLETE#: {has_complete}")
        print(f"Contains #NONE#: {has_none}")
        print(f"Contains pipe format (|): {has_pipes}")

        if not has_delimiter and not has_complete and not has_none:
            print("\n[ERROR] Response missing LightRAG format markers!")
            print("This is why extraction shows 0 Ent + 0 Rel")
            print("\nLightRAG expects one of:")
            print("  1. Claims in format: (entity|property|object|desc|source)##")
            print("  2. {#COMPLETE#} at the end")
            print("  3. {#NONE#} if no entities found")

        return response

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    result = await test_lightrag_format()

    if result:
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("We need to either:")
        print("  1. Modify Gemini prompt to return correct format")
        print("  2. Add a parser to convert Gemini format to LightRAG format")
        print("  3. Use a different approach for Gemini integration")


if __name__ == "__main__":
    asyncio.run(main())
