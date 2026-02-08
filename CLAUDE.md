# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Obsidian RAG - A system that turns Obsidian notes into a queryable knowledge base using RAG (Retrieval-Augmented Generation). It builds a knowledge graph from markdown notes, preserving wikilinks and metadata, then allows natural language queries against your notes.

## Commands

### Setup (First Time)
```bash
conda deactivate  # Exit any existing environment first
conda create -n turing0.1 python=3.10
conda activate turing0.1
pip install -r requirements.txt
cp .env.example .env  # Then edit .env with your API key and vault path
```

### Build Database (First Time)
```bash
conda activate turing0.1
python run_obsidian_raganything.py
python bootstrap_tracking.py  # Setup incremental tracking
```

### Run Web UI
```bash
conda activate turing0.1
python run_ui.py
# Opens at http://localhost:8000
```

### Incremental Sync (After Adding Notes)
```bash
python run_incremental_sync.py
# Or use Settings > Sync Vault in the web UI
```

## Architecture

### Core Pipeline
```
Obsidian Vault (.md files)
        |
        v
ObsidianChunker (src/obsidian_chunker.py)
  - 2000-token chunks
  - Preserves wikilinks, tags, frontmatter
        |
        v
SimpleRAGAnything (src/simple_raganything.py)
  - Uses RAGAnything/LightRAG framework
  - Extracts entities and relationships
  - Builds knowledge graph
        |
        v
Storage (./rag_storage/)
  - GraphML knowledge graph
  - Vector embeddings (JSON)
  - Chunk data
```

### Key Components

**src/simple_raganything.py** - Core RAG orchestration. `SimpleRAGAnything` class handles:
- Database initialization (detects existing vs new)
- Embedding via EmbeddingGemma 308M (local GPU)
- LLM calls to Gemini 2.5 Flash
- Query processing with 5 modes: hybrid, local, global, naive, mix

**src/obsidian_chunker.py** - `ObsidianChunker` class chunks markdown files:
- Extracts wikilinks with regex pattern `\[\[([^\]]+)\]\]`
- Preserves tags, frontmatter, and file relationships
- Generates `ChunkMetadata` dataclass for each chunk

**src/gemini_llm.py** - Gemini API integration:
- `gemini_complete_if_cache()` for text completion
- `gemini_vision_complete()` for multimodal content
- Uses Vertex AI endpoint with streaming

**src/vault_monitor.py** - `VaultMonitor` tracks file changes via MD5 hashes:
- `IncrementalVaultUpdater` processes only changed files
- Stores tracking state in `vault_tracking.json`

**ui/app.py** - FastAPI backend:
- WebSocket at `/ws/chat` for streaming responses
- REST endpoints: `/api/sync`, `/api/vault/status`, `/health`
- Initializes RAG system in `non_interactive=True` mode

### Data Storage

All RAG data lives in `./rag_storage/`:
- `graph_chunk_entity_relation.graphml` - Knowledge graph
- `kv_store_*.json` - Various key-value stores (docs, entities, relations, chunks)
- `vdb_*.json` - Vector databases for embeddings
- `vault_tracking.json` - File hash tracking for incremental sync

### External Dependencies

- **RAGAnything/LightRAG** - Core RAG framework (from `raganything[all]`)
- **EmbeddingGemma 308M** - Local embedding model via sentence-transformers
- **BGE Reranker v2-m3** - Local reranking model (src/bge_reranker.py)
- **Gemini 2.5 Flash** - LLM via Google's Vertex AI (free tier)

## Environment Variables

Required in `.env`:
- `VERTEX_AI_API_KEY` - Gemini API key from https://aistudio.google.com/apikey
- `OBSIDIAN_VAULT_PATH` - Absolute path to your Obsidian vault

Optional:
- `RAG_WORKING_DIR` - Storage directory (default: `./rag_storage`)
- `DEV_MODE` - Enable uvicorn auto-reload (default: false)

## Code Patterns

### Async/Await Pattern
The codebase is heavily async. RAG operations use `await`:
```python
await rag_system.initialize()
result = await rag_system.query(question, mode="hybrid")
```

### Import Pattern for src modules
Files use try/except for relative vs absolute imports:
```python
try:
    from .gemini_llm import gemini_complete_if_cache
except ImportError:
    from gemini_llm import gemini_complete_if_cache
```

### Conda Environment Check
Entry points verify the correct conda environment:
```python
def check_conda_environment():
    if os.environ.get('CONDA_DEFAULT_ENV') != 'turing0.1':
        sys.exit(1)
```
