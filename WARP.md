# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project: Obsidian RAG-Anything with EmbeddingGemma 308M

Common commands

- Environment
  - Activate conda env (required):
    - pwsh/bash: conda activate turing0.1
  - Install dependencies:
    - Option A (script; requires Bash on Windows via Git Bash or WSL): bash setup_conda_env.sh
    - Option B (manual): python -m pip install -r requirements.txt

- Run
  - Interactive runner (prompts to reuse or rebuild DB):
    - python run_obsidian_raganything.py
  - Non-interactive fresh rebuild + sample query:
    - python run_rag_auto.py

- Tests (each test_*.py is an executable script)
  - Initialization smoke test: python test_initialization.py
  - Small vault end-to-end: python test_rag_small.py
  - Run a single test file (example): python test_gemini_simple.py

- GPU check
  - python check_gpu.py

- Lint/build
  - No linter or build configuration is defined in this repo as of now.

Environment configuration

- OBSIDIAN_VAULT_PATH: Path to your Obsidian vault
  - Default used in code: C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro
- WORKING_DIR: Directory where RAG storage/graph artifacts are written
  - Default: ./rag_storage
- VERTEX_AI_API_KEY: Required to use Gemini 2.5 Flash for LLM and vision calls
  - Set as an environment variable before running; value is not checked into the repo

High-level architecture and data flow

- Purpose
  - Process an Obsidian vault into semantically enriched chunks (2K-token windows) preserving wiki links, tags, and frontmatter; build a knowledge graph and vector index using RAG-Anything; answer text and multimodal queries.

- Core components
  - src/obsidian_chunker.py (ObsidianChunker)
    - Walks the vault (skips .obsidian), discovers .md files, and for each:
      - Extracts YAML frontmatter, wiki links ([[...]]), and tags (#tag)
      - Chunks text into ~2K-token windows with approximate token estimation
      - Attaches rich metadata and inter-chunk connections (previous/next, counts)
    - Output: a list of chunk dicts with Content and metadata, plus processing stats

  - src/simple_raganything.py (SimpleRAGAnything)
    - Orchestrates the end-to-end pipeline
    - Database handling: detects existing LightRAG/RAG-Anything artifacts in WORKING_DIR; offers reuse vs rebuild (interactive). Use run_rag_auto.py for non-interactive fresh rebuild.
    - Embeddings: loads SentenceTransformer google/embeddinggemma-300m (GPU if available) and wraps it in a LightRAG-compatible EmbeddingFunc (dim=768)
    - RAG-Anything config (RAGAnythingConfig): enables multimodal processing (images/tables/equations), parser="mineru", context_window/page mode, max_context_tokens, and header/caption inclusion
    - LLM functions: delegates text and vision to gemini_* wrappers (see below) using VERTEX_AI_API_KEY
    - Processing flow:
      1) Chunk vault with ObsidianChunker
      2) Convert chunks to RAG-Anything content_list
      3) Insert content_list into RAG-Anything to build knowledge graph and vector DB
      4) Provide query() and query_multimodal() over the built store

  - src/gemini_llm.py
    - Async wrappers gemini_complete_if_cache and gemini_vision_complete that call the Google Vertex AI streaming endpoint for Gemini 2.5 Flash
    - API key is sourced from VERTEX_AI_API_KEY; returns text outputs compatible with LightRAG-style interfaces

  - Runners
    - run_obsidian_raganything.py: CLI runner that checks conda env, sets up paths, initializes SimpleRAGAnything, and optionally offers an interactive query loop. When an existing DB is present, prompts reuse vs rebuild.
    - run_rag_auto.py: Non-interactive mode that deletes old DB (if any), rebuilds, and issues a sample query.

- Data and artifacts (WORKING_DIR)
  - Knowledge graph, vector DB, and status/cache JSONs produced by RAG-Anything/LightRAG (e.g., vdb_chunks.json, kv_store_text_chunks.json, graph_chunk_entity_relation.graphml). Presence of these files triggers "reuse vs rebuild" logic.

Notes from repository rules and docs

- From README.md
  - Always run inside conda environment turing0.1
  - Defaults: Obsidian vault at the path above; multimodal processing enabled; EmbeddingGemma 308M for local embeddings
  - Optional environment overrides: OBSIDIAN_VAULT_PATH and WORKING_DIR

- From .cursor rules (important excerpts)
  - Operate as an Autonomous Engineering Lead: sequential thinking; plan before edits; prefer simplicity and removal over addition when possible
  - Hard limits: keep files under 500 lines; split approaching 400; keep functions under ~30–40 lines; avoid god classes; modular design and single responsibility
  - Tooling preference: use code search, linters, formatters, test runners; verify claims with tests or static analysis
  - Safety: escalate only for irreversible changes or missing credentials; keep edits reversible
  - Bug registry protocol: if a BUG_REGISTRY.txt exists, check and update it for hard bugs

Operational guidance for Warp in this repo

- Use the non-interactive runner (python run_rag_auto.py) when avoiding interactive prompts is important
- Ensure VERTEX_AI_API_KEY is set in the environment for any path that exercises Gemini LLM or vision calls
- If you only need embeddings and chunking without LLM calls, you can still run the pipeline until LLM steps; Gemini calls will require the API key
