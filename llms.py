"""Unified LLM client for Gemini with multimodal support.

Supports:
- Direct PDF/image processing (multimodal)
- Parallel document extraction
- JSON structured output
- Token usage tracking for cost calculation

Usage:
    # Extract from single document
    result = await extract_document(file_bytes, "pdf", "individual")

    # Extract from multiple documents in parallel
    results = await extract_documents_parallel(files, "individual")

    # Aggregate partial extractions
    final = await aggregate_extractions(partials, "individual")
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Generator

from google.genai import types
from openai import OpenAI
import httpx

from src.db.database import save_llm_debug_log

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from src.llms.config import (
    get_client,
    get_generation_config,
    calculate_cost,
    get_model_for_entity,
    DEFAULT_MODEL,
    MODEL_GEMINI_3_PRO,
    MODEL_GEMINI_3_FLASH,
    MODEL_GPT_5_2,
    MODEL_GPT_5_MINI,
)
from src.utils.mime_types import get_mime_type
from src.llms.prompts import (
    get_extraction_prompt,
    get_aggregation_prompt,
    get_schema_for_entity,
    get_classification_prompt,
    get_ocr_aggregation_prompt,
    get_clean_data_prompt,
    get_profile_extraction_prompt,
    get_dynamic_extraction_prompt,
    get_dynamic_aggregation_prompt,
    get_dynamic_ocr_aggregation_prompt,
    get_dynamic_schema_for_aggregation,
    get_dynamic_schema_for_aggregation_v2,
    get_dynamic_aggregation_prompt_v2,
    get_dynamic_ocr_aggregation_prompt_v2,
    expand_array_to_pdf_fields,
    detect_repeating_sections,
    _get_fields_from_schema,
)
from src.utils.pdf_extractor import extract_filled_pdf_fields
from src.config import EXTRACTION_MODE

# OpenAI client singleton with extended timeout for Railway
_openai_client = None

def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client with extended timeout."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[OPENAI] OPENAI_API_KEY not set in environment!")
        else:
            logger.info(f"[OPENAI] API key found: {api_key[:8]}...{api_key[-4:]}")
        _openai_client = OpenAI(
            api_key=api_key,
            timeout=httpx.Timeout(120.0, connect=30.0),
            max_retries=3,
        )
    return _openai_client


@dataclass
class LLMResult:
    """Result from an LLM call with usage metadata."""
    content: dict | str
    input_tokens: int
    output_tokens: int
    cost_usd: Decimal
    processing_time_ms: int
    error: str | None = None
    document_type: str | None = None


# ===== PUBLIC API =====

async def extract_document(
    file_bytes: bytes,
    file_type: str,
    entity_type: str,
    filename: str = "document",
    target_name: str | None = None,
) -> LLMResult:
    """Extract fields from a single document using Gemini multimodal.

    Args:
        target_name: Optional name to filter extraction for specific person/entity
    """
    start_time = time.perf_counter()
    logger.info(f"[EXTRACT] Starting extraction for '{filename}' ({file_type}, {len(file_bytes)} bytes)")
    if target_name:
        logger.info(f"[EXTRACT] Filtering for target: {target_name}")

    try:
        client = get_client()
        mime_type = get_mime_type(file_type)
        prompt = get_extraction_prompt(entity_type, target_name)
        model = get_model_for_entity(entity_type)
        logger.info(f"[EXTRACT] '{filename}' - MIME type: {mime_type}, entity: {entity_type}, model: {model}")

        # Build multimodal content: file + prompt
        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents = [file_part, prompt]
        logger.info(f"[EXTRACT] '{filename}' - Sending to Gemini ({model})...")

        # Generate JSON without schema - free extraction
        config = get_generation_config(response_json=True)

        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Extract usage metadata
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        logger.info(f"[EXTRACT] '{filename}' - Gemini response: {input_tokens} in / {output_tokens} out tokens")

        # Parse JSON response
        content = _parse_json_response(response.text)
        processing_time = int((time.perf_counter() - start_time) * 1000)

        # Log extracted fields
        fields_found = _count_non_null_fields(content)
        logger.info(f"[EXTRACT] '{filename}' - SUCCESS: {fields_found} fields extracted in {processing_time}ms")

        return LLMResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            processing_time_ms=processing_time,
        )

    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[EXTRACT] '{filename}' - FAILED: {str(e)}")
        return LLMResult(
            content={},
            input_tokens=0,
            output_tokens=0,
            cost_usd=Decimal("0"),
            processing_time_ms=processing_time,
            error=str(e),
        )


async def extract_documents_parallel(
    files: list[tuple[bytes, str, str]],  # (bytes, file_type, filename)
    entity_type: str,
    target_name: str | None = None,
) -> list[LLMResult]:
    """Extract from multiple documents in parallel.

    Args:
        target_name: Optional name to filter extraction for specific person/entity
    """
    logger.info(f"[PARALLEL] Starting parallel extraction of {len(files)} documents (mode: {EXTRACTION_MODE})")
    if target_name:
        logger.info(f"[PARALLEL] Filtering for target: {target_name}")

    # Use Mistral OCR if configured
    if EXTRACTION_MODE == "ocr_mistral":
        return await _extract_with_mistral_ocr(files, entity_type)

    # Default: Gemini multimodal extraction
    tasks = [
        extract_document(file_bytes, file_type, entity_type, filename, target_name)
        for file_bytes, file_type, filename in files
    ]
    results = await asyncio.gather(*tasks)
    success_count = sum(1 for r in results if not r.error)
    logger.info(f"[PARALLEL] Completed: {success_count}/{len(files)} documents extracted successfully")
    return results


async def _extract_with_mistral_ocr(
    files: list[tuple[bytes, str, str]],
    entity_type: str,
) -> list[LLMResult]:
    """Extract documents using smart extraction: direct PDF fields first, OCR fallback."""
    from src.ocr.mistral_ocr import extract_documents_mistral_parallel
    from src.ocr.document_ai import ocr_results_to_prompt

    start_time = time.perf_counter()
    logger.info(f"[SMART-EXTRACT-IRS] Processing {len(files)} documents for entity type '{entity_type}'")

    # PHASE 1: Try direct PDF extraction for fillable PDFs
    direct_extractions: dict[str, dict] = {}
    files_needing_ocr: list[tuple[bytes, str, str]] = []

    for file_bytes, file_type, filename in files:
        if file_type.lower() == "pdf":
            direct_fields = extract_filled_pdf_fields(file_bytes)
            if direct_fields:
                logger.info(f"[SMART-EXTRACT-IRS] Direct: {len(direct_fields)} fields from '{filename}' (FREE)")
                direct_extractions[filename] = direct_fields
            else:
                files_needing_ocr.append((file_bytes, file_type, filename))
        else:
            files_needing_ocr.append((file_bytes, file_type, filename))

    # PHASE 2: OCR for remaining files
    ocr_text = ""
    ocr_cost = Decimal("0")

    if files_needing_ocr:
        logger.info(f"[SMART-EXTRACT-IRS] Running Mistral OCR on {len(files_needing_ocr)} files...")
        ocr_results = await extract_documents_mistral_parallel(files_needing_ocr)
        filenames = [f[2] for f in files_needing_ocr]
        ocr_text = ocr_results_to_prompt(ocr_results, filenames)
        ocr_cost = sum(r.cost_usd for r in ocr_results)

    # PHASE 3: Combine direct extractions + OCR text
    combined_text_parts = []
    if direct_extractions:
        for filename, fields in direct_extractions.items():
            combined_text_parts.append(f"=== DIRECT EXTRACTION FROM: {filename} ===")
            for key, value in fields.items():
                combined_text_parts.append(f"{key}: {value}")
            combined_text_parts.append("")
    if ocr_text:
        combined_text_parts.append(ocr_text)

    combined_text = "\n".join(combined_text_parts)
    processing_time = int((time.perf_counter() - start_time) * 1000)

    direct_count = sum(len(f) for f in direct_extractions.values())
    logger.info(f"[SMART-EXTRACT-IRS] Complete: {direct_count} direct + {len(ocr_text)} OCR chars in {processing_time}ms")

    return [LLMResult(
        content={"_ocr_text": combined_text, "_entity_type": entity_type},
        input_tokens=0,
        output_tokens=0,
        cost_usd=ocr_cost,
        processing_time_ms=processing_time,
    )]


async def _extract_with_mistral_ocr_schema(
    files: list[tuple[bytes, str, str]],
    schema_data: dict,
    target_name: str | None = None,
) -> list[LLMResult]:
    """Extract documents using smart extraction: direct PDF fields first, OCR fallback.

    For filled PDFs (DocuSign, fillable forms), extracts field values directly using
    PyMuPDF - no OCR needed! Falls back to Mistral OCR for scanned PDFs and images.
    """
    from src.ocr.mistral_ocr import extract_documents_mistral_parallel
    from src.ocr.document_ai import ocr_results_to_prompt

    start_time = time.perf_counter()
    logger.info(f"[SMART-EXTRACT] Processing {len(files)} docs for schema '{schema_data.get('name')}'")

    # PHASE 1: Try direct PDF extraction for fillable PDFs
    direct_extractions: dict[str, dict] = {}
    files_needing_ocr: list[tuple[bytes, str, str]] = []

    for file_bytes, file_type, filename in files:
        if file_type.lower() == "pdf":
            direct_fields = extract_filled_pdf_fields(file_bytes)
            if direct_fields:
                logger.info(f"[SMART-EXTRACT] Direct: {len(direct_fields)} fields from '{filename}' (FREE)")
                direct_extractions[filename] = direct_fields
            else:
                logger.info(f"[SMART-EXTRACT] '{filename}' not fillable, queuing for OCR")
                files_needing_ocr.append((file_bytes, file_type, filename))
        else:
            # Images always need OCR
            logger.info(f"[SMART-EXTRACT] '{filename}' is image, queuing for OCR")
            files_needing_ocr.append((file_bytes, file_type, filename))

    # PHASE 2: OCR for remaining files
    ocr_text = ""
    ocr_cost = Decimal("0")

    if files_needing_ocr:
        logger.info(f"[SMART-EXTRACT] Running Mistral OCR on {len(files_needing_ocr)} files...")
        ocr_results = await extract_documents_mistral_parallel(files_needing_ocr)
        filenames = [f[2] for f in files_needing_ocr]
        ocr_text = ocr_results_to_prompt(ocr_results, filenames)
        ocr_cost = sum(r.cost_usd for r in ocr_results)
        logger.info(f"[SMART-EXTRACT] OCR complete: {len(ocr_text)} chars extracted")

    # PHASE 3: Combine direct extractions + OCR text for aggregation
    combined_text_parts = []

    # Add direct extraction results as structured text
    if direct_extractions:
        for filename, fields in direct_extractions.items():
            combined_text_parts.append(f"=== DIRECT EXTRACTION FROM: {filename} ===")
            for key, value in fields.items():
                combined_text_parts.append(f"{key}: {value}")
            combined_text_parts.append("")

    # Add OCR text
    if ocr_text:
        combined_text_parts.append(ocr_text)

    combined_text = "\n".join(combined_text_parts)
    processing_time = int((time.perf_counter() - start_time) * 1000)

    direct_count = sum(len(f) for f in direct_extractions.values())
    logger.info(f"[SMART-EXTRACT] Complete: {direct_count} direct fields + {len(ocr_text)} OCR chars in {processing_time}ms")

    # Return single result with combined text + schema marker
    return [LLMResult(
        content={
            "_ocr_text": combined_text,
            "_schema_data": schema_data,
            "_target_name": target_name,
            "_direct_extractions": direct_extractions,  # Pass through for potential direct use
        },
        input_tokens=0,
        output_tokens=0,
        cost_usd=ocr_cost,  # Only OCR costs money, direct extraction is FREE
        processing_time_ms=processing_time,
    )]


async def detect_entity_type(
    files: list[tuple[bytes, str, str]],  # (bytes, file_type, filename)
) -> tuple[str, str]:
    """Detect entity type by analyzing documents until confident.

    Tries each document until it gets a confident classification.
    Returns (entity_type, detected_from_filename) or raises if all unsure.
    """
    logger.info(f"[DETECT] Starting entity type detection with {len(files)} documents")

    client = get_client()
    prompt = get_classification_prompt()

    for file_bytes, file_type, filename in files:
        logger.info(f"[DETECT] Analyzing '{filename}'...")

        try:
            mime_type = get_mime_type(file_type)
            file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
            contents = [file_part, prompt]

            config = get_generation_config(response_json=True)

            response = await client.aio.models.generate_content(
                model=DEFAULT_MODEL,  # Use flash for speed
                contents=contents,
                config=config,
            )

            result = _parse_json_response(response.text)
            classification = result.get("classification", "not_sure")
            confidence = result.get("confidence", "low")
            reason = result.get("reason", "")

            logger.info(f"[DETECT] '{filename}' -> {classification} ({confidence}): {reason}")

            if classification in ("individual", "company"):
                logger.info(f"[DETECT] SUCCESS: Detected '{classification}' from '{filename}'")
                return classification, filename

            logger.info(f"[DETECT] '{filename}' inconclusive, trying next document...")

        except Exception as e:
            logger.warning(f"[DETECT] Error analyzing '{filename}': {e}")
            continue

    # All documents were inconclusive
    logger.error("[DETECT] FAILED: Could not determine entity type from any document")
    raise ValueError("Could not determine entity type. Please specify 'individual' or 'company' manually.")


async def aggregate_extractions(
    partial_results: list[dict],
    entity_type: str,
    client_name: str | None = None,
    user_instructions: str | None = None,
    existing_data: dict | None = None,
    available_document_types: list[str] | None = None,
) -> LLMResult:
    """Aggregate multiple partial extractions into final fields.

    Args:
        user_instructions: Natural language corrections that take HIGH PRIORITY over extracted data
        existing_data: Existing client profile data to include as context
        available_document_types: List of document types present in the submission (for checklist fields)
    """
    start_time = time.perf_counter()
    logger.info(f"[AGGREGATE] Starting aggregation of {len(partial_results)} partial extractions for client: {client_name}")
    if user_instructions:
        logger.info(f"[AGGREGATE] User instructions provided: {user_instructions[:100]}...")
    if existing_data:
        logger.info(f"[AGGREGATE] Including existing profile data with {len(existing_data)} keys")

    # Check if this is OCR mode (special _ocr_text key)
    if partial_results and "_ocr_text" in partial_results[0]:
        ocr_text = partial_results[0]["_ocr_text"]
        ocr_entity = partial_results[0].get("_entity_type", entity_type)
        logger.info(f"[AGGREGATE] Detected OCR mode, using aggregate_from_ocr...")
        return await aggregate_from_ocr(ocr_text, ocr_entity, client_name, user_instructions)

    # Use Gemini 3 Flash for all aggregations (both individual and company)
    return await _aggregate_with_gemini_flash(partial_results, entity_type, client_name, start_time, user_instructions, existing_data, available_document_types)


async def _aggregate_with_gemini_flash(
    partial_results: list[dict],
    entity_type: str,
    client_name: str | None,
    start_time: float,
    user_instructions: str | None = None,
    existing_data: dict | None = None,
    available_document_types: list[str] | None = None,
) -> LLMResult:
    """Aggregate using Gemini 3 Flash with medium thinking level."""
    try:
        client = get_client()
        schema = get_schema_for_entity(entity_type)
        prompt = get_aggregation_prompt(entity_type, client_name, user_instructions, existing_data, available_document_types)

        partials_json = json.dumps(partial_results, indent=2, default=str)
        schema_json = json.dumps(schema, indent=2)

        for i, p in enumerate(partial_results):
            non_null = _count_non_null_fields(p) if isinstance(p, dict) else 0
            logger.info(f"[AGGREGATE] Partial {i+1}: {non_null} non-null fields")

        full_prompt = f"""{prompt}

OUTPUT SCHEMA (follow this structure exactly):
{schema_json}

PARTIAL EXTRACTIONS:
{partials_json}"""

        logger.info(f"[AGGREGATE] Sending {len(partials_json)} chars to Gemini 3 Flash...")

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
        )

        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
        logger.info(f"[AGGREGATE] Gemini Flash response: {input_tokens} in / {output_tokens} out tokens")

        raw_content = response.text
        logger.info(f"[AGGREGATE] Raw response length: {len(raw_content)} chars")
        logger.info(f"[AGGREGATE] Raw response preview: {raw_content[:500]}...")

        content = json.loads(raw_content)
        content = _sanitize_field_values(content)
        processing_time = int((time.perf_counter() - start_time) * 1000)

        logger.info(f"[AGGREGATE] Parsed keys: {list(content.keys())}")
        fields_found = _count_non_null_fields(content)
        logger.info(f"[AGGREGATE] SUCCESS: {fields_found} final fields merged in {processing_time}ms")

        return LLMResult(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens), processing_time_ms=processing_time,
        )
    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[AGGREGATE] Gemini Flash FAILED: {str(e)}")
        return LLMResult(
            content={}, input_tokens=0, output_tokens=0, cost_usd=Decimal("0"),
            processing_time_ms=processing_time, error=str(e),
        )


def call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Simple sync LLM call for text generation."""
    client = get_client()
    config = get_generation_config(temperature=temperature)

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    return response.text


def call_llm_json(prompt: str, schema: dict | None = None, temperature: float = 0.3) -> dict:
    """Sync LLM call with JSON response."""
    client = get_client()
    config = get_generation_config(response_json=True, response_schema=schema, temperature=temperature)

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
    )
    return _parse_json_response(response.text)


def call_llm_stream(prompt: str, temperature: float = 0.3) -> Generator[dict, None, None]:
    """Stream LLM response chunks."""
    client = get_client()
    config = get_generation_config(temperature=temperature)

    for chunk in client.models.generate_content_stream(
        model=DEFAULT_MODEL,
        contents=prompt,
        config=config,
    ):
        if chunk.text:
            yield {"type": "chunk", "content": chunk.text}


# ===== INTERNAL FUNCTIONS =====

def _sanitize_field_values(data: dict | list) -> dict | list:
    """Replace field-name-like values with null (LLM hallucination fix)."""
    if isinstance(data, list):
        return [_sanitize_field_values(item) if isinstance(item, (dict, list)) else item for item in data]
    if not isinstance(data, dict):
        return data
    for key, value in data.items():
        if isinstance(value, dict):
            _sanitize_field_values(value)
        elif isinstance(value, list):
            data[key] = _sanitize_field_values(value)
        elif isinstance(value, str) and value.startswith("line_"):
            logger.warning(f"[SANITIZE] Replaced hallucinated field name: {key}={value}")
            data[key] = None
    return data


def _parse_json_response(text: str) -> dict:
    """Parse JSON from LLM response, handling edge cases."""
    if not text:
        return {}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return json.loads(text[start:end].strip())
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                return json.loads(text[start:end].strip())

        return {"error": "json_parse_failed", "raw_content": text}


def _count_non_null_fields(data: dict | list, depth: int = 0) -> int:
    """Count non-null fields recursively in a dict or list."""
    if isinstance(data, list):
        return sum(_count_non_null_fields(item, depth) if isinstance(item, (dict, list)) else (1 if item is not None else 0) for item in data)
    if not isinstance(data, dict):
        return 1 if data is not None else 0
    count = 0
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            count += _count_non_null_fields(value, depth + 1)
        elif isinstance(value, list):
            count += _count_non_null_fields(value, depth + 1)
        else:
            count += 1
    return count


# ===== OCR-BASED AGGREGATION =====

async def aggregate_from_ocr(
    ocr_text: str,
    entity_type: str,
    client_name: str | None = None,
    user_instructions: str | None = None,
) -> LLMResult:
    """Aggregate extracted OCR data into final schema fields.

    Takes structured OCR output (text, checkboxes, key-value pairs) and maps
    to the W-8BEN/W-8BEN-E schema using LLM for semantic understanding.

    Args:
        user_instructions: Natural language corrections that take HIGH PRIORITY over extracted data
    """
    start_time = time.perf_counter()
    logger.info(f"[OCR-AGGREGATE] Starting OCR aggregation for client: {client_name}")
    if user_instructions:
        logger.info(f"[OCR-AGGREGATE] User instructions provided: {user_instructions[:100]}...")

    # Use Gemini 3 Flash for all OCR aggregations (both individual and company)
    return await _aggregate_ocr_with_gemini_flash(ocr_text, entity_type, client_name, start_time, user_instructions)


async def _aggregate_ocr_with_gemini_flash(
    ocr_text: str,
    entity_type: str,
    client_name: str | None,
    start_time: float,
    user_instructions: str | None = None,
) -> LLMResult:
    """Aggregate OCR output using Gemini 3 Flash with medium thinking level."""
    try:
        client = get_client()
        schema = get_schema_for_entity(entity_type)
        prompt = get_ocr_aggregation_prompt(entity_type, client_name, user_instructions)

        full_prompt = f"{prompt}\n\nOCR OUTPUT:\n{ocr_text}"
        logger.info(f"[OCR-AGGREGATE] Sending {len(ocr_text)} chars to Gemini 3 Flash...")

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    response_mime_type="application/json",
                    response_json_schema=schema,
                ),
            )
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        logger.info(f"[OCR-AGGREGATE] Gemini Flash response: {input_tokens} in / {output_tokens} out tokens")

        content = _parse_json_response(response.text)
        content = _sanitize_field_values(content)
        processing_time = int((time.perf_counter() - start_time) * 1000)

        fields_found = _count_non_null_fields(content)
        logger.info(f"[OCR-AGGREGATE] SUCCESS: {fields_found} fields mapped in {processing_time}ms")

        return LLMResult(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens), processing_time_ms=processing_time,
        )
    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[OCR-AGGREGATE] Gemini Flash FAILED: {str(e)}")
        return LLMResult(
            content={}, input_tokens=0, output_tokens=0, cost_usd=Decimal("0"),
            processing_time_ms=processing_time, error=str(e),
        )


async def aggregate_from_ocr_schema(
    ocr_text: str,
    schema_data: dict,
    client_name: str | None = None,
    user_instructions: str | None = None,
    job_id: str | None = None,
    schema_id: str | None = None,
    existing_data: dict | None = None,
) -> LLMResult:
    """Aggregate OCR text using dynamic schema.

    Args:
        ocr_text: OCR-extracted text from Mistral
        schema_data: Dynamic schema definition
        client_name: Client/beneficial owner name
        user_instructions: Natural language corrections
        job_id: For debug logging
        schema_id: For debug logging
        existing_data: Existing client profile data to supplement extractions
    """
    start_time = time.perf_counter()
    logger.info(f"[OCR-AGGREGATE-SCHEMA] Starting OCR aggregation for schema '{schema_data.get('name')}'")

    # Diagnostic logging for schema structure debugging
    fields_direct = schema_data.get("fields", [])
    sections = schema_data.get("sections", [])
    logger.info(f"[OCR-AGGREGATE-SCHEMA] schema_data keys: {list(schema_data.keys())}")
    logger.info(f"[OCR-AGGREGATE-SCHEMA] Direct fields count: {len(fields_direct)}, Sections count: {len(sections)}")

    if user_instructions:
        logger.info(f"[OCR-AGGREGATE-SCHEMA] User instructions provided: {user_instructions[:100]}...")

    # Use v2 when schema has repeating sections (indexed fields like field[0], field[1])
    fields = _get_fields_from_schema(schema_data)
    repeating_sections = detect_repeating_sections(fields)
    use_v2 = len(repeating_sections) > 0
    if use_v2:
        logger.info(f"[OCR-AGGREGATE-SCHEMA] Using v2 array-based schema (repeating sections: {repeating_sections})")

    try:
        client = get_client()
        if use_v2:
            prompt, _, _ = get_dynamic_ocr_aggregation_prompt_v2(schema_data, client_name, user_instructions, existing_data)
        else:
            prompt = get_dynamic_ocr_aggregation_prompt(schema_data, client_name, user_instructions, existing_data)

        full_prompt = f"{prompt}\n\nOCR OUTPUT:\n{ocr_text}"
        logger.info(f"[OCR-AGGREGATE-SCHEMA] Sending {len(ocr_text)} chars to Gemini 3 Flash (medium thinking)...")

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    response_mime_type="application/json",
                ),
            )
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        logger.info(f"[OCR-AGGREGATE-SCHEMA] Gemini Flash response: {input_tokens} in / {output_tokens} out tokens")

        # Capture raw output BEFORE parsing for debug
        raw_output_text = response.text
        content = _parse_json_response(raw_output_text)
        content = _sanitize_field_values(content)

        # Save structured output BEFORE expansion for debugging
        raw_llm_response = content.copy() if isinstance(content, dict) else content

        # Expand arrays back to flat PDF fields if v2 was used
        if use_v2:
            expanded = expand_array_to_pdf_fields(content, schema_data)
            logger.info(f"[OCR-AGGREGATE-SCHEMA] Expanded {len(content)} array items to {len(expanded)} flat fields")
            content = expanded

        processing_time = int((time.perf_counter() - start_time) * 1000)

        fields_found = _count_non_null_fields(content)
        logger.info(f"[OCR-AGGREGATE-SCHEMA] SUCCESS: {fields_found} fields mapped in {processing_time}ms")

        if job_id:
            # Build complete I/O record for debugging
            llm_io_record = {
                "input": full_prompt,
                "output": raw_output_text,
                "structured_output": content,
            }
            await save_llm_debug_log(
                job_id=job_id,
                prompt_text=full_prompt,
                input_files=[{"schema_data": schema_data, "ocr_text_length": len(ocr_text), "use_v2": use_v2}],
                llm_response=raw_llm_response,
                llm_io=llm_io_record,
                schema_id=schema_id,
                model_name=MODEL_GEMINI_3_FLASH,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens),
            )

        return LLMResult(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens), processing_time_ms=processing_time,
        )
    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[OCR-AGGREGATE-SCHEMA] Gemini Flash FAILED: {str(e)}")
        return LLMResult(
            content={}, input_tokens=0, output_tokens=0, cost_usd=Decimal("0"),
            processing_time_ms=processing_time, error=str(e),
        )


async def clean_extracted_data(data: dict, entity_type: str, existing_fields: list[str] | None = None) -> dict:
    """Clean and format extracted data for better readability using gpt-5-mini.

    Args:
        data: Raw extracted data to clean
        entity_type: 'individual' or 'company'
        existing_fields: List of field keys already in client (e.g., ['email', 'phone', 'address', 'notes'])
    """
    client = _get_openai_client()
    data_json = json.dumps(data, indent=2, default=str)
    prompt = get_clean_data_prompt(entity_type, existing_fields)
    full_prompt = f"{prompt}\n\n<input_data>\n{data_json}\n</input_data>"

    logger.info(f"[CLEAN-DATA] Sending {len(data_json)} chars to gpt-5-mini...")

    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": full_prompt}],
            max_completion_tokens=4000,
            response_format={"type": "json_object"}
        )
    )

    raw_content = response.choices[0].message.content
    logger.info(f"[CLEAN-DATA] Response: {len(raw_content)} chars")

    cleaned = json.loads(raw_content)
    fields_count = len([k for k, v in cleaned.items() if v is not None])
    logger.info(f"[CLEAN-DATA] SUCCESS: {fields_count} clean fields")

    return cleaned


async def generate_field_hints_batch(
    fields: list[dict],
    schema_name: str,
    entity_type: str,
    template_ocr_text: str | None = None
) -> dict:
    """Generate AI instructions for a batch of schema fields using GPT-5-mini.

    Args:
        fields: List of field dicts with id, label, type, pdf_field_id, description
        schema_name: Name of the schema/form
        entity_type: 'individual' or 'company'
        template_ocr_text: OCR text from the PDF template for context

    Returns:
        dict with 'hints' (field_id -> instruction) and 'cost_usd'
    """
    client = _get_openai_client()

    # Build field list with slot context from descriptions
    field_lines = []
    slot_mappings = {}  # Track which indices map to which roles
    for f in fields:
        field_id = f.get('id', '')
        label = f.get('label', '')
        field_type = f.get('type', 'text')
        pdf_key = f.get('pdf_field_id', '')
        desc = f.get('description', '')

        # Extract slot role from description (e.g., "Primary Account Holder: First Name")
        role = ""
        if ":" in desc:
            role = desc.split(":")[0].strip()
            # Track unique roles per section index pattern
            if "[" in field_id and "]" in field_id:
                idx_str = field_id.split("[")[1].split("]")[0]
                if idx_str.isdigit():
                    section = field_id.split("[")[0]
                    key = f"{section}[{idx_str}]"
                    if key not in slot_mappings:
                        slot_mappings[key] = role

        field_lines.append(f"- {field_id}: label='{label}', type={field_type}, pdf_key='{pdf_key}'" + (f", role='{role}'" if role else ""))

    field_list = "\n".join(field_lines)

    # Build slot mapping context
    slot_context = ""
    if slot_mappings:
        slot_lines = [f"  - {k}: {v}" for k, v in sorted(slot_mappings.items())]
        slot_context = f"""
SLOT MAPPING (which array indices correspond to which roles):
{chr(10).join(slot_lines)}

When generating hints, reference these roles. For example, if Individual[0] is "Primary Account Holder",
a hint might say: "This field is for the Primary Account Holder (Individual[0]). Look for..."
"""

    # Build template context
    template_context = ""
    if template_ocr_text:
        # Truncate if too long
        max_template_chars = 15000
        if len(template_ocr_text) > max_template_chars:
            template_ocr_text = template_ocr_text[:max_template_chars] + "\n...[truncated]"
        template_context = f"""
<pdf_template_content>
The following is the OCR text from the PDF template. Use this to understand the document structure:
{template_ocr_text}
</pdf_template_content>
"""

    prompt = f"""<role>
You are analyzing a PDF form template to extract INSTRUCTIONAL NOTES for each field.
The AI filling this form cannot see the PDF - it only has client documents.
</role>

<template_content>
{template_context}
</template_content>

{slot_context}

<task>
Extract ONLY instructional text that appears NEAR each field in the PDF template.
Do NOT echo the field label - that's already visible to users.
</task>

<extract>
- Section headers/notes: "Complete this section only if..."
- Conditional instructions: "If U.S. citizen, skip to Part II"
- Skip/leave blank rules: "Leave blank if same as mailing address"
- Format requirements: "Enter date as MM-DD-YYYY"
- Validation notes: "Must match name on government ID"
- Checkbox explanations: "Check this box if you are NOT a U.S. person"
</extract>

<ignore>
- The field label itself (already visible to users)
- Generic advice you're making up
- Obvious purposes (don't say "First Name: enter your first name")
- Anything not LITERALLY written on the PDF template
</ignore>

<fields>
{field_list}
</fields>

<output_format>
Return JSON object. Use null for fields with NO instructional notes.
Example:
{{"TrustedContact.Name": "Section note: If no changes to existing Trusted Contact, skip this section", "Individual.FirstName": null, "MailingAddress.Street": "Note: Leave blank if same as permanent residence address", "DateOfBirth": "Format shown: MM-DD-YYYY"}}
</output_format>"""

    logger.info(f"[GENERATE-HINTS] Generating hints for {len(fields)} fields...")

    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
    )

    raw_content = response.choices[0].message.content
    hints_raw = json.loads(raw_content)
    # Filter out null values - only keep fields with actual notes
    hints = {k: v for k, v in hints_raw.items() if v is not None}

    # Calculate cost (GPT-5-mini pricing: $0.30/1M input, $1.20/1M output)
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = Decimal(str(input_tokens)) * Decimal("0.0000003") + Decimal(str(output_tokens)) * Decimal("0.0000012")

    logger.info(f"[GENERATE-HINTS] Generated {len(hints)} hints, cost: ${cost:.6f}")

    return {"hints": hints, "cost_usd": cost}


async def extract_profile(
    files: list[tuple[bytes, str, str]],
    client_name: str | None = None,
    entity_type: str = "individual",
) -> LLMResult:
    """Extract comprehensive profile data from documents for wealth management.

    Uses Mistral OCR first to extract text, then passes to LLM for structuring.
    Returns JSON with two sections:
    - client_info: Contact/profile data (human-readable keys)
    - form_fields: Tax form data (schema-structured keys for PDF filler)

    Args:
        files: List of (bytes, file_type, filename) tuples
        client_name: Optional client name to filter relevant information
        entity_type: 'individual' or 'company' - determines which schema to use
    """
    from src.ocr.mistral_ocr import extract_documents_mistral_parallel
    from src.ocr.document_ai import ocr_results_to_prompt

    start_time = time.perf_counter()
    logger.info(f"[PROFILE] Starting profile extraction from {len(files)} documents")
    if client_name:
        logger.info(f"[PROFILE] Filtering for client: {client_name}")
    logger.info(f"[PROFILE] Entity type: {entity_type}")

    try:
        # Step 1: Run Mistral OCR on all documents
        logger.info(f"[PROFILE] Step 1: Running Mistral OCR...")
        ocr_results = await extract_documents_mistral_parallel(files)
        filenames = [f[2] for f in files]
        ocr_text = ocr_results_to_prompt(ocr_results, filenames)
        ocr_cost = sum(r.cost_usd for r in ocr_results)
        logger.info(f"[PROFILE] OCR complete: {len(ocr_text)} chars extracted")

        # Step 2: Pass OCR text to Gemini Flash for profile extraction
        logger.info(f"[PROFILE] Step 2: Extracting profile from OCR text...")
        client = get_client()
        prompt = get_profile_extraction_prompt(client_name, entity_type)
        full_prompt = f"{prompt}\n\n<ocr_text>\n{ocr_text}\n</ocr_text>"

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    response_mime_type="application/json",
                ),
            )
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        logger.info(f"[PROFILE] Gemini Flash response: {input_tokens} in / {output_tokens} out tokens")

        raw_content = response.text
        logger.info(f"[PROFILE] Raw response length: {len(raw_content)} chars")
        logger.info(f"[PROFILE] Raw response preview: {raw_content[:1000]}...")
        content = json.loads(raw_content)
        logger.info(f"[PROFILE] Parsed content keys: {list(content.keys())[:30]}")
        processing_time = int((time.perf_counter() - start_time) * 1000)

        llm_cost = calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens)
        total_cost = ocr_cost + llm_cost

        fields_found = _count_non_null_fields(content)
        logger.info(f"[PROFILE] SUCCESS: {fields_found} profile fields extracted in {processing_time}ms")

        return LLMResult(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=total_cost,
            processing_time_ms=processing_time,
        )

    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[PROFILE] FAILED: {str(e)}")
        return LLMResult(
            content={},
            input_tokens=0,
            output_tokens=0,
            cost_usd=Decimal("0"),
            processing_time_ms=processing_time,
            error=str(e),
        )


# ===== DYNAMIC SCHEMA EXTRACTION =====

async def extract_document_with_schema(
    file_bytes: bytes,
    file_type: str,
    schema_data: dict,
    filename: str = "document",
    target_name: str | None = None,
) -> LLMResult:
    """Extract fields using dynamic schema definition."""
    start_time = time.perf_counter()
    logger.info(f"[EXTRACT-DYNAMIC] Starting extraction for '{filename}' using schema '{schema_data.get('name')}'")

    try:
        client = get_client()
        mime_type = get_mime_type(file_type)
        prompt = get_dynamic_extraction_prompt(schema_data, target_name)
        model = MODEL_GEMINI_3_FLASH

        file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime_type)
        contents = [file_part, prompt]

        config = get_generation_config(response_json=True)
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=config,
        )

        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0) or 0
        content = _parse_json_response(response.text)
        processing_time = int((time.perf_counter() - start_time) * 1000)

        fields_found = _count_non_null_fields(content)
        logger.info(f"[EXTRACT-DYNAMIC] SUCCESS: {fields_found} fields extracted in {processing_time}ms")

        return LLMResult(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[EXTRACT-DYNAMIC] FAILED: {str(e)}")
        return LLMResult(
            content={}, input_tokens=0, output_tokens=0, cost_usd=Decimal("0"),
            processing_time_ms=processing_time, error=str(e),
        )


async def extract_documents_with_schema(
    files: list[tuple[bytes, str, str]],
    schema_data: dict,
    target_name: str | None = None,
) -> list[LLMResult]:
    """Extract from multiple documents using dynamic schema."""
    logger.info(f"[PARALLEL-DYNAMIC] Extracting {len(files)} docs with schema '{schema_data.get('name')}' (mode: {EXTRACTION_MODE})")

    # Use Mistral OCR if configured (same pattern as extract_documents_parallel)
    if EXTRACTION_MODE == "ocr_mistral":
        return await _extract_with_mistral_ocr_schema(files, schema_data, target_name)

    # Fallback: Gemini multimodal extraction
    tasks = [
        extract_document_with_schema(file_bytes, file_type, schema_data, filename, target_name)
        for file_bytes, file_type, filename in files
    ]
    results = await asyncio.gather(*tasks)
    success_count = sum(1 for r in results if not r.error)
    logger.info(f"[PARALLEL-DYNAMIC] Completed: {success_count}/{len(files)} documents extracted")
    return results


async def aggregate_with_schema(
    partial_results: list[dict],
    schema_data: dict,
    client_name: str | None = None,
    user_instructions: str | None = None,
    job_id: str | None = None,
    schema_id: str | None = None,
    existing_data: dict | None = None,
    available_document_types: list[str] | None = None,
) -> LLMResult:
    """Aggregate extractions using dynamic schema.

    Args:
        partial_results: Partial extractions from documents
        schema_data: Dynamic schema definition
        client_name: Client/beneficial owner name
        user_instructions: Natural language corrections
        job_id: For debug logging
        schema_id: For debug logging
        existing_data: Existing client profile data to supplement extractions
        available_document_types: List of document types present in the submission (for checklist fields)
    """
    start_time = time.perf_counter()
    logger.info(f"[AGGREGATE-DYNAMIC] Aggregating {len(partial_results)} partials for schema '{schema_data.get('name')}'")

    # Check if this is OCR mode (special _ocr_text key from Mistral OCR extraction)
    if partial_results and "_ocr_text" in partial_results[0]:
        ocr_text = partial_results[0]["_ocr_text"]
        ocr_schema = partial_results[0].get("_schema_data", schema_data)
        logger.info(f"[AGGREGATE-DYNAMIC] Detected OCR mode, routing to aggregate_from_ocr_schema...")
        return await aggregate_from_ocr_schema(ocr_text, ocr_schema, client_name, user_instructions, job_id, schema_id, existing_data)

    # Use v2 when schema has repeating sections (indexed fields like field[0], field[1])
    fields = _get_fields_from_schema(schema_data)
    repeating_sections = detect_repeating_sections(fields)
    use_v2 = len(repeating_sections) > 0
    if use_v2:
        logger.info(f"[AGGREGATE-DYNAMIC] Using v2 array-based schema (repeating sections: {repeating_sections})")

    try:
        client = get_client()
        if use_v2:
            prompt, _, _ = get_dynamic_aggregation_prompt_v2(schema_data, client_name, user_instructions, existing_data, available_document_types)
        else:
            prompt = get_dynamic_aggregation_prompt(schema_data, client_name, user_instructions, existing_data, available_document_types)
        partials_json = json.dumps(partial_results, indent=2, default=str)

        full_prompt = f"""{prompt}

PARTIAL EXTRACTIONS:
{partials_json}"""

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="medium"),
                    response_mime_type="application/json",
                ),
            )
        )

        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
        content = json.loads(response.text)
        content = _sanitize_field_values(content)

        # Expand arrays back to flat PDF fields if v2 was used
        if use_v2:
            expanded = expand_array_to_pdf_fields(content, schema_data)
            logger.info(f"[AGGREGATE-DYNAMIC] Expanded {len(content)} array items to {len(expanded)} flat fields")
            content = expanded

        processing_time = int((time.perf_counter() - start_time) * 1000)

        fields_found = _count_non_null_fields(content)
        logger.info(f"[AGGREGATE-DYNAMIC] SUCCESS: {fields_found} fields merged in {processing_time}ms")

        return LLMResult(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[AGGREGATE-DYNAMIC] FAILED: {str(e)}")
        return LLMResult(
            content={}, input_tokens=0, output_tokens=0, cost_usd=Decimal("0"),
            processing_time_ms=processing_time, error=str(e),
        )


async def fix_extraction(
    previous_fields: dict,
    corrections: str,
    schema_data: dict | None = None,
) -> LLMResult:
    """Apply user corrections to previous extraction output.

    LLM receives the previous extraction + user correction text,
    returns fixed fields with corrections applied.
    """
    start_time = time.perf_counter()
    logger.info(f"[FIX-EXTRACTION] Applying corrections to {len(previous_fields)} fields")

    try:
        client = get_client()

        # Build schema context if available
        schema_context = ""
        if schema_data:
            field_names = [f.get("label", f.get("id", "")) for f in schema_data.get("fields", [])]
            schema_context = f"\nAvailable fields in schema: {', '.join(field_names[:50])}"

        prompt = f"""You are correcting a previous data extraction based on user feedback.

PREVIOUS EXTRACTION:
{json.dumps(previous_fields, indent=2, default=str)}

USER CORRECTIONS:
{corrections}
{schema_context}

Apply the user's corrections to the extraction. Return the corrected JSON with:
1. All corrections from the user applied
2. All unchanged fields preserved exactly as they were
3. Same JSON structure as the input

Return ONLY the corrected JSON object, no explanations."""

        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=MODEL_GEMINI_3_FLASH,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    response_mime_type="application/json",
                ),
            )
        )

        input_tokens = response.usage_metadata.prompt_token_count or 0
        output_tokens = response.usage_metadata.candidates_token_count or 0
        content = json.loads(response.text)
        content = _sanitize_field_values(content)

        processing_time = int((time.perf_counter() - start_time) * 1000)
        fields_changed = sum(1 for k in content if content.get(k) != previous_fields.get(k))
        logger.info(f"[FIX-EXTRACTION] SUCCESS: {fields_changed} fields changed in {processing_time}ms")

        return LLMResult(
            content=content, input_tokens=input_tokens, output_tokens=output_tokens,
            cost_usd=calculate_cost(MODEL_GEMINI_3_FLASH, input_tokens, output_tokens),
            processing_time_ms=processing_time,
        )
    except Exception as e:
        processing_time = int((time.perf_counter() - start_time) * 1000)
        logger.error(f"[FIX-EXTRACTION] FAILED: {str(e)}")
        return LLMResult(
            content={}, input_tokens=0, output_tokens=0, cost_usd=Decimal("0"),
            processing_time_ms=processing_time, error=str(e),
        )


