"""
Gemini 3 Flash LLM Integration for RAG-Anything
Uses Google GenAI SDK for all Gemini operations
"""

import os
import asyncio
from typing import List, Dict, Any, Optional

import google.genai as genai
from google.genai import types

# Gemini 3 Flash model ID
MODEL_GEMINI_3_FLASH = "gemini-3-flash-preview"


def _get_gemini_client() -> genai.Client:
    """Get Gemini client using API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Gemini 3 Flash completion function compatible with LightRAG's interface

    Args:
        model: Model name (ignored - always uses Gemini 3 Flash)
        prompt: User prompt text
        system_prompt: System prompt (optional)
        history_messages: Chat history (optional)
        api_key: API key (optional, uses GEMINI_API_KEY env var)
        **kwargs: Additional arguments (ignored)

    Returns:
        Generated text response from Gemini
    """
    if history_messages is None:
        history_messages = []

    client = _get_gemini_client()

    # Build conversation content
    contents = []

    # Add system prompt as first user message
    if system_prompt:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"System instructions: {system_prompt}")]
            )
        )

    # Add history messages
    for msg in history_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Map roles
        gemini_role = "model" if role == "assistant" else "user"
        contents.append(
            types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=content)]
            )
        )

    # Add current prompt
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    )

    # Retry logic with exponential backoff for transient errors
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            # Call Gemini 3 Flash
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=MODEL_GEMINI_3_FLASH,
                    contents=contents,
                )
            )

            # Extract text from response
            if response.candidates and response.candidates[0].content.parts:
                result_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        result_text += part.text
                return result_text.strip()

            return ""

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if error is retryable
            is_retryable = any(x in error_str for x in [
                "429", "503", "500", "timeout", "connection", "network", "unavailable"
            ])

            if not is_retryable or attempt == max_retries - 1:
                print(f"[ERROR] Gemini 3 Flash API call failed: {e}")
                raise

            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            print(f"[WARN] Gemini API error (attempt {attempt+1}/{max_retries}): {e}")
            print(f"[RETRY] Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)

    # Should never reach here, but safety fallback
    raise last_error


async def gemini_vision_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = None,
    image_data: Optional[str] = None,
    messages: Optional[List[Dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Gemini 3 Flash vision completion for multimodal content

    Args:
        prompt: Text prompt
        system_prompt: System prompt (optional)
        history_messages: Chat history (optional)
        image_data: Base64 encoded image data (optional)
        messages: Pre-formatted messages (optional)
        api_key: API key (optional)
        **kwargs: Additional arguments

    Returns:
        Generated response for multimodal content
    """
    if history_messages is None:
        history_messages = []

    client = _get_gemini_client()

    # Build content structure
    contents = []

    # Add system prompt
    if system_prompt:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"System instructions: {system_prompt}")]
            )
        )

    # Handle pre-formatted messages (multimodal VLM format)
    if messages:
        for msg in messages:
            if msg.get("role") == "system":
                continue  # Already handled

            role = "model" if msg.get("role") == "assistant" else "user"
            msg_content = msg.get("content", "")

            # Handle multimodal content
            if isinstance(msg_content, list):
                parts = []
                for item in msg_content:
                    if item.get("type") == "text":
                        parts.append(types.Part.from_text(text=item.get("text", "")))
                    elif item.get("type") == "image_url":
                        # Extract base64 image data
                        image_url = item.get("image_url", {}).get("url", "")
                        if "base64," in image_url:
                            b64_data = image_url.split("base64,")[1]
                            parts.append(
                                types.Part.from_bytes(
                                    data=__import__("base64").b64decode(b64_data),
                                    mime_type="image/jpeg"
                                )
                            )

                if parts:
                    contents.append(types.Content(role=role, parts=parts))
            else:
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=str(msg_content))]
                    )
                )

    # Handle traditional single image format
    elif image_data:
        import base64
        parts = [types.Part.from_text(text=prompt)]
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(image_data),
                mime_type="image/jpeg"
            )
        )
        contents.append(types.Content(role="user", parts=parts))

    # Pure text format
    else:
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)]
            )
        )

    # Retry logic with exponential backoff for transient errors
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            # Call Gemini 3 Flash
            response = await asyncio.to_thread(
                lambda: client.models.generate_content(
                    model=MODEL_GEMINI_3_FLASH,
                    contents=contents,
                )
            )

            # Extract text from response
            if response.candidates and response.candidates[0].content.parts:
                result_text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        result_text += part.text
                return result_text.strip()

            return ""

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Check if error is retryable
            is_retryable = any(x in error_str for x in [
                "429", "503", "500", "timeout", "connection", "network", "unavailable"
            ])

            if not is_retryable or attempt == max_retries - 1:
                print(f"[ERROR] Gemini 3 Flash Vision API call failed: {e}")
                raise

            # Exponential backoff: 1s, 2s, 4s
            wait_time = 2 ** attempt
            print(f"[WARN] Gemini Vision API error (attempt {attempt+1}/{max_retries}): {e}")
            print(f"[RETRY] Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)

    # Should never reach here, but safety fallback
    raise last_error
