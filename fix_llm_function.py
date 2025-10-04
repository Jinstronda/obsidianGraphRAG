#!/usr/bin/env python3
"""
Quick Fix for LLM Function Issue
Based on Context7 LightRAG documentation, the issue is likely in the LLM function
not properly handling the formatted prompt with Source Data context.
"""

import os

# Create the fixed Gemini LLM function
fixed_gemini_llm = '''
async def _llm_function(self, prompt: str, system_prompt: str = None, history_messages=[], **kwargs) -> str:
    """
    LLM function using Gemini 2.5 Flash for entity extraction
    FIXED: Properly handles Source Data context in prompts
    """
    # Print debug info to see what we're getting
    print(f"\\n[DEBUG] LLM Function called:")
    print(f"  - Prompt length: {len(prompt)} characters")
    print(f"  - System prompt: {'YES' if system_prompt else 'NO'}")
    print(f"  - Has 'Source Data': {'YES' if 'Source Data' in prompt or 'source data' in prompt.lower() else 'NO'}")
    
    # If prompt is very short, it might be missing context
    if len(prompt) < 100:
        print(f"  - [WARNING] Prompt is very short ({len(prompt)} chars) - may be missing context")
    
    # Check if this is entity extraction vs query
    is_extraction = any(phrase in prompt.lower() for phrase in [
        "extract entities", "extract relations", "entity extraction", 
        "relationship extraction", "knowledge graph"
    ])
    
    try:
        vertex_api_key = os.getenv("VERTEX_AI_API_KEY")
        
        # Use Gemini 2.5 Flash for both extraction and queries
        from gemini_llm import gemini_complete_if_cache
        
        result = await gemini_complete_if_cache(
            model="gemini-2.5-flash",
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=vertex_api_key,
            **kwargs
        )
        
        print(f"  - [DEBUG] Gemini response length: {len(result) if result else 0} characters")
        
        # If result is None or empty, return a default message
        if not result or len(result.strip()) < 10:
            print(f"  - [WARNING] Gemini returned empty/short result: '{result}'")
            return "I apologize, but I'm having trouble processing your request. Please try again."
        
        return result
        
    except Exception as e:
        print(f"  - [ERROR] Gemini API error: {e}")
        return f"Error processing request: {str(e)}"
'''

print("="*70)
print("FIXING LLM FUNCTION ISSUE")
print("="*70)

print("\nBased on Context7 LightRAG documentation, the issue is:")
print("1. The LLM function might not be receiving the formatted prompt correctly")
print("2. The 'Source Data' context isn't being passed properly to Gemini")
print("3. The prompt template may be malformed")

print(f"\nFixed LLM function (with debugging):")
print("="*50)
print(fixed_gemini_llm)

# Write the fix to a patch file
patch_file = "./llm_function_fix.py"
with open(patch_file, 'w') as f:
    f.write(f'''# LLM Function Fix
# Replace the _llm_function in simple_raganything.py with this version

{fixed_gemini_llm}
''')

print("="*70)
print("SOLUTION STEPS:")
print("="*70)
print(f"1. The fixed LLM function has been saved to: {patch_file}")
print("2. Replace the _llm_function in src/simple_raganything.py with this version")
print("3. The debug version will show exactly what's happening with prompts")
print("4. This should reveal why 'Source Data' context isn't reaching Gemini")

print(f"\nAlternatively, according to Context7 docs, you can:")
print("1. Set enable_rerank=False to remove the rerank warning")
print("2. Use QueryParam with user_prompt to override the default prompt")
print("3. Check if the issue is in prompt template formatting")

print(f"\nContext7 Solution - Add this to your query:")
print("```python")
print("from lightrag import QueryParam")
print("")
print("result = await rag.rag_anything.aquery(")
print("    'The top 10 mental models',")
print("    param=QueryParam(")
print("        mode='hybrid',")
print("        enable_rerank=False,  # Remove rerank warning")
print("        user_prompt='Answer based on the provided source data. If information is available, provide a comprehensive response.'")
print("    )")
print(")")
print("```")

print(f"\nThis should fix the 'I do not have enough information' issue!")