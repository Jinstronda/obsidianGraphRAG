#!/usr/bin/env python3
"""
Test fixed Gemini API wrapper
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gemini_llm import gemini_complete_if_cache

API_KEY = "REDACTED"

async def test_fixed_api():
    print("="*70)
    print("TEST: FIXED GEMINI API")
    print("="*70)

    print("\n[1/2] Testing simple prompt...")
    result1 = await gemini_complete_if_cache(
        model="gemini-2.5-flash",
        prompt="Say hello in one word",
        api_key=API_KEY
    )
    print(f"Result: '{result1}'")

    if not result1:
        print("[FAIL] Empty response!")
        return False

    print("\n[2/2] Testing entity extraction prompt...")
    test_text = "Python is a programming language. Machine learning uses Python."

    extraction_prompt = f"""Extract entities from this text: {test_text}

Return format: (entity|type|description)
Example: (Python|language|programming language)"""

    result2 = await gemini_complete_if_cache(
        model="gemini-2.5-flash",
        prompt=extraction_prompt,
        api_key=API_KEY
    )

    print(f"Extraction result:")
    print("-" * 70)
    print(result2)
    print("-" * 70)

    if not result2:
        print("[FAIL] Empty extraction!")
        return False

    print("\n[SUCCESS] Gemini API working correctly!")
    return True


async def main():
    success = await test_fixed_api()

    if success:
        print("\n" + "="*70)
        print("READY TO RUN FULL EXTRACTION")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("STILL BROKEN - CHECK ERRORS")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
