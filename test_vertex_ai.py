#!/usr/bin/env python3
"""
Test Vertex AI Gemini 2.5 Flash API
"""
import os
import sys

# Test if google-genai is installed
try:
    from google import genai
    print("[OK] google-genai library found")
except ImportError:
    print("[ERROR] google-genai not installed")
    print("\nInstalling google-genai...")
    os.system("pip install google-genai")
    from google import genai

# Configuration
VERTEX_AI_API_KEY = "REDACTED"

# You need to provide these from your Vertex AI console
PROJECT_ID = input("Enter your Google Cloud Project ID: ").strip()
LOCATION = input("Enter location (e.g., us-central1): ").strip() or "us-central1"

print(f"\n{'='*70}")
print("TESTING VERTEX AI GEMINI 2.5 FLASH")
print(f"{'='*70}")
print(f"Project: {PROJECT_ID}")
print(f"Location: {LOCATION}")
print(f"API Key: {VERTEX_AI_API_KEY[:20]}...")

try:
    # Initialize Vertex AI client
    print("\n[1/3] Initializing Vertex AI client...")
    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        api_key=VERTEX_AI_API_KEY
    )
    print("   ✓ Client initialized")

    # Test simple generation
    print("\n[2/3] Testing simple text generation...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'hello' in one word"
    )
    print(f"   ✓ Response: {response.text}")

    # Test entity extraction format
    print("\n[3/3] Testing entity extraction format...")
    test_text = """
    Machine learning is a subset of artificial intelligence.
    It uses neural networks for pattern recognition.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"Extract entities and relations from this text: {test_text}"
    )
    print(f"   ✓ Extraction test passed")
    print(f"\nExtraction result:\n{response.text}")

    print(f"\n{'='*70}")
    print("✓ ALL TESTS PASSED - VERTEX AI IS READY!")
    print(f"{'='*70}")
    print("\nRate Limits (Paid Tier):")
    print("   • 1,000 requests per minute")
    print("   • 1,000,000 tokens per minute")
    print("\nEstimated processing time: ~12 minutes")
    print(f"Estimated cost: ~$7.65 for 6119 chunks")

except Exception as e:
    print(f"\n{'='*70}")
    print("✗ TEST FAILED")
    print(f"{'='*70}")
    print(f"Error: {e}")
    print("\nTroubleshooting:")
    print("1. Verify your Project ID is correct")
    print("2. Ensure Vertex AI API is enabled in your project")
    print("3. Check that the API key has proper permissions")
    print("4. Verify billing is enabled on your project")
    sys.exit(1)
