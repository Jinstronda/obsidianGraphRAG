"""
Agentic CRAG (Corrective RAG) implementation using Gemini 3 Flash.
Iteratively searches the knowledge base until sufficient information is gathered.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import AsyncGenerator, Any

import google.genai as genai
from google.genai import types

from src.prompts import (
    CRAG_SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    TOOL_TO_MODE,
    MAX_ITERATIONS,
    FORCED_ANSWER_PROMPT,
)
from src.simple_raganything import SimpleRAGAnything
from src.context_manager import ContextManager

logger = logging.getLogger(__name__)

# Gemini 3 Flash model ID
MODEL_GEMINI_3_FLASH = "gemini-3-flash-preview"


@dataclass
class AgentStep:
    """Represents a single step in the agent's reasoning process."""
    step_number: int
    tool_name: str
    tool_args: dict
    result: str
    result_preview: str  # Truncated for UI display


@dataclass
class AgentResponse:
    """Final response from the agent."""
    answer: str
    confidence: str
    sources_used: int
    steps: list[AgentStep]
    total_iterations: int


def _get_gemini_client() -> genai.Client:
    """Get Gemini client using API key from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")
    return genai.Client(api_key=api_key)


def _build_tool_config() -> types.Tool:
    """Build Gemini tool configuration from TOOL_DEFINITIONS."""
    function_declarations = []
    for tool_def in TOOL_DEFINITIONS:
        function_declarations.append(
            types.FunctionDeclaration(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
            )
        )
    return types.Tool(function_declarations=function_declarations)


class AgenticRAG:
    """
    Agentic CRAG implementation that wraps SimpleRAGAnything.
    Uses Gemini 3 Flash for intelligent tool selection and answer synthesis.
    """

    def __init__(self, rag_system: SimpleRAGAnything):
        """
        Initialize AgenticRAG with an existing SimpleRAGAnything instance.

        Args:
            rag_system: Initialized SimpleRAGAnything instance
        """
        self.rag = rag_system
        self.client = _get_gemini_client()
        self.tool_config = _build_tool_config()
        self.context_manager = ContextManager()

    async def _execute_search_tool(self, tool_name: str, query: str) -> str:
        """
        Execute a search tool by calling SimpleRAGAnything.query().
        Includes quality validation to filter out low-quality results.
        """
        mode = TOOL_TO_MODE.get(tool_name, "hybrid")
        logger.info(f"[CRAG] Executing {tool_name}: {query[:100]}...")

        try:
            result = await self.rag.query(question=query, mode=mode)

            # Detect low-quality responses that indicate no real content found
            LOW_QUALITY = [
                "i do not have enough information",
                "no relevant",
                "cannot find",
                "unable to locate",
                "not found in",
            ]
            if len(result) < 50 or any(p in result.lower() for p in LOW_QUALITY):
                logger.info(f"[CRAG] {tool_name} returned low-quality result")
                return f"[No relevant vault content for: {query}]"

            logger.info(f"[CRAG] {tool_name} returned {len(result)} chars")
            return f"[Vault results for '{query}']: {result}"
        except Exception as e:
            logger.error(f"[CRAG] {tool_name} failed: {e}")
            return f"[Search failed: {str(e)}]"

    async def _call_gemini(
        self,
        conversation: list[types.Content],
    ) -> types.GenerateContentResponse:
        """
        Call Gemini 3 Flash with tools.
        Uses asyncio.to_thread for sync API call.
        """
        config = types.GenerateContentConfig(
            tools=[self.tool_config],
        )

        response = await asyncio.to_thread(
            lambda: self.client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=conversation,
                config=config,
            )
        )
        return response

    async def query(self, question: str, history: list[dict] = None) -> AgentResponse:
        """
        Query the knowledge base using CRAG pattern.
        Returns the final synthesized answer.

        Args:
            question: User's question
            history: Optional conversation history for context

        Returns:
            AgentResponse with answer, confidence, and reasoning steps
        """
        steps = []
        async for event in self.query_streaming(question, history=history):
            if event["type"] == "step":
                steps.append(event["content"])
            elif event["type"] == "answer":
                return AgentResponse(
                    answer=event["content"]["answer"],
                    confidence=event["content"]["confidence"],
                    sources_used=event["content"].get("sources_used", 0),
                    steps=steps,
                    total_iterations=len(steps),
                )

        # Fallback if no answer event (shouldn't happen)
        return AgentResponse(
            answer="Unable to generate answer",
            confidence="low",
            sources_used=0,
            steps=steps,
            total_iterations=len(steps),
        )

    async def query_streaming(
        self, question: str, history: list[dict] = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Query with streaming events for real-time UI updates.

        Args:
            question: User's question
            history: Optional conversation history for context continuity

        Yields events:
        - {"type": "step", "content": AgentStep}
        - {"type": "answer", "content": {"answer": str, "confidence": str, "sources_used": int}}
        - {"type": "token_update", "content": {"tokens": int, "percent": float}}
        - {"type": "done"}
        """
        logger.info(f"[CRAG] Starting agentic query: {question[:100]}...")

        # Check if history needs compaction before starting
        if history and self.context_manager.needs_compaction():
            logger.info("[CRAG] Compacting conversation history...")
            history = await self.context_manager.compact_history(history)
            yield {
                "type": "context_compacted",
                "content": {"compaction_count": self.context_manager.compaction_count}
            }

        # Build conversation with system prompt and history
        conversation = []

        # Add system prompt first
        conversation.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=CRAG_SYSTEM_PROMPT)],
            )
        )

        # Add conversation history if provided
        if history:
            for entry in history:
                role = entry.get("role", "user")
                content = entry.get("content", "")

                if entry.get("is_summary"):
                    # This is a compacted summary from previous compaction
                    conversation.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=f"[Context]: {content}")],
                        )
                    )
                elif role == "user":
                    conversation.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=content)],
                        )
                    )
                elif role in ("assistant", "model"):
                    conversation.append(
                        types.Content(
                            role="model",
                            parts=[types.Part.from_text(text=content)],
                        )
                    )

            logger.info(f"[CRAG] Added {len(history)} history entries to conversation")

        # Add current question
        conversation.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"User Question: {question}")],
            )
        )

        gathered_context = []
        iteration = 0

        while iteration < MAX_ITERATIONS:
            iteration += 1
            logger.info(f"[CRAG] Iteration {iteration}/{MAX_ITERATIONS}")

            # Call Gemini with tools
            response = await self._call_gemini(conversation)

            # Track token usage from response
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                self.context_manager.update_token_count({
                    "total_token_count": getattr(usage, "total_token_count", 0),
                    "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                })
                yield {
                    "type": "token_update",
                    "content": {
                        "tokens": self.context_manager.total_tokens,
                        "percent": self.context_manager.get_token_usage_percent(),
                    }
                }

            # Check response candidates
            if not response.candidates:
                logger.error("[CRAG] No candidates in response")
                break

            candidate = response.candidates[0]
            content = candidate.content

            # Collect all function calls and text parts from response
            function_calls = []
            text_parts = []
            for part in content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    function_calls.append(part.function_call)
                elif hasattr(part, "text") and part.text:
                    text_parts.append(part.text.strip())

            # If no function calls, treat text as final answer
            if not function_calls and text_parts:
                combined_text = " ".join(text_parts)
                if combined_text:
                    yield {
                        "type": "answer",
                        "content": {
                            "answer": combined_text,
                            "confidence": "medium",
                            "sources_used": len(gathered_context),
                        },
                    }
                    yield {"type": "done"}
                    return

            # Check for finish_answer first
            for fc in function_calls:
                if fc.name == "finish_answer":
                    tool_args = dict(fc.args) if fc.args else {}
                    answer = tool_args.get("answer", "")
                    confidence = tool_args.get("confidence", "medium")
                    sources_used = tool_args.get("sources_used", len(gathered_context))
                    yield {
                        "type": "answer",
                        "content": {
                            "answer": answer,
                            "confidence": confidence,
                            "sources_used": sources_used,
                        },
                    }
                    yield {"type": "done"}
                    return

            # Filter to search tools only (not finish_answer)
            search_calls = [fc for fc in function_calls if fc.name != "finish_answer"]

            if not search_calls:
                continue

            # Execute up to 2 tools in parallel
            tools_to_execute = search_calls[:2]
            logger.info(f"[CRAG] Executing {len(tools_to_execute)} tool(s) in parallel")

            # Build async tasks for parallel execution
            async def execute_tool(fc):
                tool_args = dict(fc.args) if fc.args else {}
                query = tool_args.get("query", question)
                result = await self._execute_search_tool(fc.name, query)
                return (fc.name, tool_args, result)

            # Execute tools in parallel using asyncio.gather
            results = await asyncio.gather(*[execute_tool(fc) for fc in tools_to_execute])

            # Process results and yield steps
            function_response_parts = []
            step_counter = 0
            for tool_name, tool_args, result in results:
                step_counter += 1
                gathered_context.append(result)
                logger.info(f"[CRAG] Tool {tool_name} returned {len(result)} chars")

                # Create step record
                step = AgentStep(
                    step_number=iteration,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    result=result,
                    result_preview=result[:200] + "..." if len(result) > 200 else result,
                )

                yield {
                    "type": "step",
                    "content": {
                        "step": step.step_number,
                        "tool": step.tool_name,
                        "args": step.tool_args,
                        "result_preview": step.result_preview,
                        "parallel": len(tools_to_execute) > 1,
                    },
                }

                # Collect function response parts
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )

            # Add model's response to conversation (preserves thoughtSignature)
            conversation.append(content)

            # Add all function responses in a single message
            conversation.append(
                types.Content(
                    role="user",
                    parts=function_response_parts,
                )
            )

        # Max iterations reached - force an answer
        logger.warning(f"[CRAG] Max iterations ({MAX_ITERATIONS}) reached, forcing answer")

        context_str = "\n\n---\n\n".join(gathered_context) if gathered_context else "No relevant information found."
        forced_prompt = FORCED_ANSWER_PROMPT.format(
            max_iter=MAX_ITERATIONS,
            question=question,
            context=context_str,  # No truncation - Gemini 3 Flash has 1M token context
        )

        # Final call without tools to get answer
        final_conversation = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=forced_prompt)],
            )
        ]

        final_config = types.GenerateContentConfig()

        final_response = await asyncio.to_thread(
            lambda: self.client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=final_conversation,
                config=final_config,
            )
        )

        if final_response.candidates and final_response.candidates[0].content.parts:
            answer = final_response.candidates[0].content.parts[0].text
        else:
            answer = "I was unable to find sufficient information to answer your question."

        yield {
            "type": "answer",
            "content": {
                "answer": answer,
                "confidence": "low",
                "sources_used": len(gathered_context),
            },
        }
        yield {"type": "done"}
