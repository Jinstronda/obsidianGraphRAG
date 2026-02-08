"""
Context manager for handling conversation compaction when approaching token limits.
Uses GPT-5-Mini for intelligent summarization to enable infinite conversations.
"""
from __future__ import annotations
import asyncio
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# 80% of Gemini 3 Flash's 1M token limit
MAX_TOKENS_BEFORE_COMPACT = 800_000


class ContextManager:
    """
    Manages conversation context and handles compaction when approaching token limits.
    Uses GPT-5-Mini to intelligently summarize conversation history while preserving
    key facts, decisions, and context needed for continuity.
    """

    def __init__(self):
        self.client = OpenAI()
        self.total_tokens = 0
        self.compaction_count = 0

    def update_token_count(self, usage_metadata: dict) -> None:
        """Update token count from Gemini response's usage_metadata."""
        if usage_metadata:
            self.total_tokens = usage_metadata.get("total_token_count", 0)
            logger.info(f"[Context] Token count: {self.total_tokens:,} / {MAX_TOKENS_BEFORE_COMPACT:,}")

    def needs_compaction(self) -> bool:
        """Check if conversation history needs to be compacted."""
        return self.total_tokens > MAX_TOKENS_BEFORE_COMPACT

    def get_token_usage_percent(self) -> float:
        """Get current token usage as percentage of limit."""
        return (self.total_tokens / MAX_TOKENS_BEFORE_COMPACT) * 100

    async def compact_history(self, history: list[dict]) -> list[dict]:
        """
        Use GPT-5-Mini to summarize conversation history with retry logic.
        Returns a compacted history with a summary message replacing older exchanges.

        Args:
            history: List of conversation entries with 'role' and 'content'

        Returns:
            Compacted history list with summary as first entry
        """
        if not history:
            return history

        logger.info(f"[Context] Compacting {len(history)} messages (tokens: {self.total_tokens:,})")
        self.compaction_count += 1

        # Format history for summarization
        conversation_text = self._format_history_for_summary(history)

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-5-mini",
                    max_completion_tokens=5000,
                    messages=[{
                        "role": "user",
                        "content": f"""Summarize this conversation for context continuity. Preserve:
- Key facts, information, and insights discovered from searches
- User's questions and what topics they were interested in
- Important search results and findings
- Any decisions, preferences, or conclusions reached
- Enough context to continue the conversation naturally

Format as a detailed but concise summary. Include specific information, not vague references.

Conversation:
{conversation_text}"""
                    }]
                )

                summary = response.choices[0].message.content
                logger.info(f"[Context] Compacted to summary ({len(summary)} chars)")

                # Return new history with summary as context
                return [{
                    "role": "system",
                    "content": f"[Previous conversation summary - compaction #{self.compaction_count}]:\n{summary}",
                    "is_summary": True
                }]

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if error is retryable
                is_retryable = any(x in error_str for x in [
                    "429", "503", "500", "timeout", "connection", "rate"
                ])

                if not is_retryable or attempt == max_retries - 1:
                    logger.error(f"[Context] Compaction failed after {attempt+1} attempts: {e}")
                    # On failure, keep more history (20 instead of 5) for better continuity
                    return history[-20:] if len(history) > 20 else history

                # Exponential backoff: 1s, 2s, 4s
                wait_time = 2 ** attempt
                logger.warning(f"[Context] Compaction error (attempt {attempt+1}/{max_retries}): {e}")
                logger.info(f"[Context] Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        # Should never reach here, but safety fallback
        logger.error(f"[Context] All retries exhausted: {last_error}")
        return history[-20:] if len(history) > 20 else history

    def _format_history_for_summary(self, history: list[dict]) -> str:
        """Format conversation history as readable text for summarization."""
        lines = []
        for entry in history:
            role = entry.get("role", "unknown")
            content = entry.get("content", "")

            # Handle different entry formats
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant" or role == "model":
                # Truncate very long responses
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                lines.append(f"Assistant: {content}")
            elif role == "system":
                if entry.get("is_summary"):
                    lines.append(f"[Previous Summary]: {content}")
                else:
                    lines.append(f"System: {content}")

        return "\n\n".join(lines)

    def reset(self) -> None:
        """Reset context manager state (e.g., for new conversation)."""
        self.total_tokens = 0
        self.compaction_count = 0
        logger.info("[Context] Reset context manager")
