# LLM Function Fix
# Replace the _llm_function in simple_raganything.py with this version


async def _llm_function(self, prompt: str, system_prompt: str = None, history_messages=[], **kwargs) -> str:
    """
    LLM function using Gemini 2.5 Flash for entity extraction
    FIXED: Properly handles Source Data context in prompts
    """
    # Print debug info to see what we're getting
    print(f"\n[DEBUG] LLM Function called:")
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

