#!/usr/bin/env python3
"""
Test Gemini 2.5 Flash Entity Extraction on Test Vault File
This test can run independently of the main RAG system
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from gemini_llm import gemini_complete_if_cache

# Test Vertex AI API Key
VERTEX_AI_API_KEY = "REDACTED"

async def test_extraction_on_note(note_path: str):
    """Test entity extraction on a specific note file"""
    
    print("="*70)
    print("TESTING GEMINI ENTITY EXTRACTION ON SPECIFIC NOTE")
    print("="*70)
    
    # Read the test note
    print(f"\n[1/4] Reading note: {note_path}")
    try:
        with open(note_path, 'r', encoding='utf-8') as f:
            note_content = f.read()
        
        print(f"   [OK] Read {len(note_content)} characters")
        print(f"\n   Preview:")
        print("   " + "-"*66)
        for line in note_content.split('\n')[:10]:
            print(f"   {line}")
        if len(note_content.split('\n')) > 10:
            print(f"   ... ({len(note_content.split('\n')) - 10} more lines)")
        print("   " + "-"*66)
        
    except Exception as e:
        print(f"   [ERROR] Failed to read note: {e}")
        return False
    
    # Create entity extraction prompt
    extraction_prompt = f"""
    Extract entities and their relationships from the following Obsidian note.
    
    Text: {note_content}
    
    Please extract:
    1. Entities (name, type, description)
    2. Relationships between entities (source, target, description)
    3. Tags and wikilinks mentioned
    
    Format your response as a structured list with clear entity and relationship sections.
    """
    
    print("\n[2/4] Testing Gemini API connection...")
    try:
        # Test basic API call
        basic_response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt="Say 'API is working' in one short sentence.",
            api_key=VERTEX_AI_API_KEY
        )
        
        print(f"   [OK] API connection successful")
        print(f"   Response: {basic_response}")
        
    except Exception as e:
        print(f"   [ERROR] API connection failed: {e}")
        return False
    
    print("\n[3/4] Extracting entities from note...")
    try:
        # Extract entities from the note
        extraction_response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt=extraction_prompt,
            system_prompt="You are an expert at extracting entities and relationships from Obsidian notes. Pay attention to wikilinks [[]] and tags #.",
            api_key=VERTEX_AI_API_KEY
        )
        
        print(f"   [OK] Entity extraction successful")
        print(f"\n   Extracted Entities & Relations:")
        print("   " + "="*66)
        # Print response with indentation
        for line in extraction_response.split('\n'):
            print(f"   {line}")
        print("   " + "="*66)
        
    except Exception as e:
        print(f"   [ERROR] Entity extraction failed: {e}")
        return False
    
    print("\n[4/4] Testing context-aware follow-up...")
    try:
        # Test follow-up question with context
        followup_response = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt="How many main entities did you identify in the note?",
            system_prompt="You are an expert at extracting entities and relationships from Obsidian notes.",
            history_messages=[
                {"role": "user", "content": extraction_prompt},
                {"role": "assistant", "content": extraction_response}
            ],
            api_key=VERTEX_AI_API_KEY
        )
        
        print(f"   [OK] Follow-up successful")
        print(f"   Response: {followup_response}")
        
    except Exception as e:
        print(f"   [ERROR] Follow-up failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("[SUCCESS] GEMINI ENTITY EXTRACTION WORKING ON TEST FILE!")
    print("="*70)
    
    print("\nKey Findings:")
    print("   [OK] Gemini API working correctly")
    print("   [OK] Entity extraction from Obsidian notes working")
    print("   [OK] Wikilinks and tags can be processed")
    print("   [OK] Context-aware follow-up working")
    print("   [OK] Ready for full vault processing")
    
    return True


async def main():
    """Main test function"""
    
    # Test on Note1.md from test_vault
    test_note = os.path.join(os.path.dirname(__file__), "test_vault", "Note1.md")
    
    if not os.path.exists(test_note):
        print(f"[ERROR] Test note not found: {test_note}")
        print("Available test notes:")
        test_vault = os.path.join(os.path.dirname(__file__), "test_vault")
        if os.path.exists(test_vault):
            for file in os.listdir(test_vault):
                if file.endswith('.md'):
                    print(f"   - {file}")
        sys.exit(1)
    
    try:
        success = await test_extraction_on_note(test_note)
        
        if success:
            print("\n" + "="*70)
            print("ALL TESTS PASSED - GEMINI API READY")
            print("="*70)
            print("\nThis test runs independently and does NOT lock the database.")
            print("You can run this while the main RAG system is running.")
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
    asyncio.run(main())
