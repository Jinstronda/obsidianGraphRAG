#!/usr/bin/env python3
"""
Minimal Gemini API test - absolute simplest case
"""

import asyncio
import aiohttp
import json

API_KEY = "REDACTED"

async def test_minimal():
    """Test 1: Absolute minimal Gemini call"""

    print("="*70)
    print("TEST 1: MINIMAL GEMINI API CALL")
    print("="*70)

    # Simplest possible request
    endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?key={API_KEY}"

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Say hi"}]
            }
        ]
    }

    print("\n[1/3] Sending request...")
    print(f"Endpoint: {endpoint[:80]}...")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:

                print(f"\n[2/3] Response status: {response.status}")

                if response.status != 200:
                    error = await response.text()
                    print(f"[ERROR] {error}")
                    return False

                # Get raw response
                raw_text = await response.text()
                print(f"\n[3/3] Raw response length: {len(raw_text)} bytes")
                print(f"\nRaw response:")
                print("-" * 70)
                print(raw_text)
                print("-" * 70)

                # Try to parse
                result = ""
                for line in raw_text.strip().split('\n'):
                    if line.strip():
                        data = json.loads(line)
                        print(f"\nParsed JSON keys: {list(data.keys())}")

                        if 'candidates' in data:
                            for candidate in data['candidates']:
                                print(f"Candidate keys: {list(candidate.keys())}")
                                if 'content' in candidate:
                                    print(f"Content keys: {list(candidate['content'].keys())}")
                                    if 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            print(f"Part keys: {list(part.keys())}")
                                            if 'text' in part:
                                                result += part['text']

                print(f"\n\nExtracted text: '{result}'")

                if result:
                    print("\n[SUCCESS] Got response from Gemini!")
                    return True
                else:
                    print("\n[FAIL] Empty response extracted")
                    return False

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    success = await test_minimal()

    if success:
        print("\n" + "="*70)
        print("GEMINI API IS WORKING!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("GEMINI API FAILED - CHECK ERROR ABOVE")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
