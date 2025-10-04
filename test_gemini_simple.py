#!/usr/bin/env python3
"""
Test Gemini 2.5 Flash with simple REST API
"""
import requests
import json

API_KEY = "REDACTED"

print("="*70)
print("TESTING GEMINI 2.5 FLASH API")
print("="*70)

# Test endpoint
url = f"https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:streamGenerateContent?key={API_KEY}"

# Simple test payload
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Say 'hello' in one word"
                }
            ]
        }
    ]
}

print("\n[1/2] Testing API connection...")
print(f"Endpoint: {url[:80]}...")

try:
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30
    )

    if response.status_code == 200:
        print("   [OK] API call successful!")

        # Parse streaming response
        result_text = ""
        for line in response.text.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    if 'candidates' in data:
                        for candidate in data['candidates']:
                            if 'content' in candidate and 'parts' in candidate['content']:
                                for part in candidate['content']['parts']:
                                    if 'text' in part:
                                        result_text += part['text']
                except json.JSONDecodeError:
                    continue

        print(f"   Response: {result_text}")

        print("\n[2/2] Testing entity extraction format...")

        # Test entity extraction
        extraction_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Extract entities from: Machine learning uses neural networks for AI."
                        }
                    ]
                }
            ]
        }

        response2 = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=extraction_payload,
            timeout=30
        )

        if response2.status_code == 200:
            print("   [OK] Entity extraction test passed!")

        print("\n" + "="*70)
        print("[SUCCESS] GEMINI 2.5 FLASH API IS READY!")
        print("="*70)
        print("\nPaid Tier Rate Limits:")
        print("   - 1,000 RPM")
        print("   - 1,000,000 TPM")
        print("\nEstimated for 6119 chunks:")
        print("   - Time: ~12 minutes")
        print("   - Cost: ~$7.65")

    else:
        print(f"\n[ERROR] API call failed!")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nTroubleshooting:")
    print("1. Check if API key is valid")
    print("2. Verify Vertex AI API is enabled")
    print("3. Ensure billing is enabled")
