"""
Quick test to verify OpenAI API connection
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def test_openai():
    """Test if OpenAI API is working"""
    from lightrag.llm.openai import openai_complete_if_cache

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    print(f"Testing OpenAI connection...")
    print(f"Model: {model}")
    print(f"API Key: {api_key[:20]}...{api_key[-10:]}")

    try:
        # Simple test with 30s timeout
        result = await asyncio.wait_for(
            openai_complete_if_cache(
                model,
                "Say 'hello' in one word",
                system_prompt=None,
                history_messages=[],
                api_key=api_key,
                base_url=None
            ),
            timeout=30.0
        )
        print(f"\n[SUCCESS] Response: {result}")
        return True
    except asyncio.TimeoutError:
        print(f"\n[TIMEOUT] API did not respond within 30 seconds")
        return False
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openai())
    if success:
        print("\n[OK] OpenAI API is working correctly")
    else:
        print("\n[FAILED] OpenAI API connection failed - check API key, quota, or network")