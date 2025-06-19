#!/usr/bin/env python3
"""
Obsidian Graph RAG System
=========================

A comprehensive Retrieval-Augmented Generation system specifically designed for Obsidian vaults.
This system combines traditional vector-based RAG with knowledge graph capabilities to provide
enhanced context understanding and multi-hop reasoning capabilities.

Key Features:
- Graph-based knowledge representation of Obsidian notes
- Hybrid retrieval (vector similarity + graph traversal)
- Entity extraction and relationship detection
- Sequential reasoning for complex queries
- Local and global search capabilities
- Real-time vault monitoring and incremental updates

Architecture Overview:
1. Data Layer: Obsidian vault parsing and preprocessing
2. Processing Layer: Entity extraction and text chunking
3. Graph Layer: Knowledge graph construction and community detection
4. Storage Layer: Vector database and graph storage
5. Retrieval Layer: Hybrid search combining multiple strategies
6. Reasoning Layer: Sequential thinking for complex queries
7. Generation Layer: OpenAI integration for answer synthesis

Author: Assistant
License: MIT
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import re
import yaml

# Core libraries for text processing and ML
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from community import community_louvain  # For community detection

# OpenAI and vector storage
import openai
from openai import OpenAI
import tiktoken

# File monitoring and async operations
import aiofiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration management
from configparser import ConfigParser
from dataclasses import asdict

# Additional imports for data persistence
import pickle

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if it exists
except ImportError:
    pass  # dotenv not installed, continue without it


# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class GraphRAGConfig:
    """
    Configuration class for the Graph RAG system.
    
    This centralizes all configuration options with sensible defaults while
    allowing users to customize behavior through environment variables or
    configuration files.
    """
    
    # Vault and file system settings
    vault_path: str = field(default_factory=lambda: os.getenv("OBSIDIAN_VAULT_PATH", ""))
    file_patterns: List[str] = field(default_factory=lambda: ["*.md"])
    exclude_patterns: List[str] = field(default_factory=lambda: [".obsidian/*", "*.tmp"])
    
    # OpenAI API settings
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o"))  # Use GPT-4o which has 128K context
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "2000")))  # Reduced to leave room for context
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.1")))
    
    # Text processing settings
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    semantic_chunking: bool = True
    
    # Entity extraction settings
    entity_extraction_prompt: str = """
    Extract entities and their relationships from the following text.
    Focus on:
    - People, organizations, concepts, topics
    - Important relationships between entities
    - Key themes and ideas
    
    Return as JSON with structure:
    {
        "entities": [{"name": "entity", "type": "type", "description": "desc"}],
        "relationships": [{"source": "entity1", "target": "entity2", "type": "relation", "description": "desc"}]
    }
    """
    
    # Graph construction settings
    min_entity_mentions: int = 2
    similarity_threshold: float = 0.7
    max_graph_nodes: int = 10000
    community_resolution: float = 1.0
    
    # Retrieval settings
    top_k_vector: int = field(default_factory=lambda: int(os.getenv("TOP_K_VECTOR", "10")))
    top_k_graph: int = field(default_factory=lambda: int(os.getenv("TOP_K_GRAPH", "5")))
    max_graph_hops: int = 2
    rerank_results: bool = True
    
    # NEW: Enhanced retrieval settings
    enable_hybrid_search: bool = field(default_factory=lambda: os.getenv("ENABLE_HYBRID_SEARCH", "false").lower() == "true")
    enable_dynamic_retrieval: bool = field(default_factory=lambda: os.getenv("ENABLE_DYNAMIC_RETRIEVAL", "false").lower() == "true")
    enable_community_summarization: bool = field(default_factory=lambda: os.getenv("ENABLE_COMMUNITY_SUMMARIZATION", "false").lower() == "true")
    enable_tensor_reranking: bool = field(default_factory=lambda: os.getenv("ENABLE_TENSOR_RERANKING", "false").lower() == "true")
    enable_lazy_graphrag: bool = field(default_factory=lambda: os.getenv("ENABLE_LAZY_GRAPHRAG", "false").lower() == "true")
    
    # Hybrid search settings
    bm25_weight: float = 0.3  # Weight for BM25 vs vector search
    sparse_vector_weight: float = 0.2  # Weight for sparse vectors
    
    # Advanced clustering settings
    use_leiden_clustering: bool = field(default_factory=lambda: os.getenv("USE_LEIDEN_CLUSTERING", "false").lower() == "true")
    leiden_resolution: float = 1.0
    leiden_iterations: int = 10
    
    # Community summarization settings
    max_community_size: int = 20  # Max entities per community summary
    community_summary_model: str = "gpt-4o-mini"  # Cheaper model for summaries
    generate_hierarchical_summaries: bool = True
    
    # Tensor reranking settings
    reranking_model: str = "jinaai/jina-colbert-v2"  # ColBERT-style model
    enable_late_interaction: bool = True
    tensor_rerank_top_k: int = 50  # Rerank top K results
    
    # Multi-granular indexing settings
    enable_multi_granular: bool = field(default_factory=lambda: os.getenv("ENABLE_MULTI_GRANULAR", "false").lower() == "true")
    skeleton_graph_ratio: float = 0.1  # Percentage of docs in skeleton
    keyword_bipartite_features: int = 1000  # Max features for keyword graph
    
    # Sequential thinking settings
    enable_sequential_thinking: bool = True
    max_reasoning_steps: int = 5
    thinking_temperature: float = 0.0
    
    # Storage settings
    vector_db_path: str = "./data/vectors.db"
    graph_db_path: str = "./data/graph.gpickle"
    cache_dir: str = field(default_factory=lambda: os.getenv("CACHE_DIR", "./cache"))
    
    # Performance settings
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    enable_caching: bool = True
    cache_ttl: int = 3600
    
    # Logging settings
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = "./logs/graphrag.log"
    
    # Web interface settings
    web_host: str = field(default_factory=lambda: os.getenv("WEB_HOST", "127.0.0.1"))
    web_port: int = field(default_factory=lambda: int(os.getenv("WEB_PORT", "5000")))
    auto_open_browser: bool = field(default_factory=lambda: os.getenv("AUTO_OPEN_BROWSER", "true").lower() == "true")


@dataclass
class Document:
    """
    Represents a single Obsidian document with metadata.
    
    This structure captures all relevant information about an Obsidian note,
    including its content, metadata, and relationships to other documents.
    """
    id: str  # Unique identifier (usually file path hash)
    path: str  # Full file path
    title: str  # Document title (from filename or frontmatter)
    content: str  # Full text content
    frontmatter: Dict[str, Any] = field(default_factory=dict)  # YAML frontmatter
    tags: Set[str] = field(default_factory=set)  # Extracted tags
    wikilinks: Set[str] = field(default_factory=set)  # [[wikilink]] references
    backlinks: Set[str] = field(default_factory=set)  # Documents that link to this one
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    word_count: int = 0
    embedding: Optional[np.ndarray] = None  # Document-level embedding


@dataclass
class TextChunk:
    """
    Represents a chunk of text from a document.
    
    Chunks are the basic unit of retrieval in the RAG system. They contain
    enough context to be meaningful while being small enough for efficient
    processing and retrieval.
    """
    id: str  # Unique chunk identifier
    document_id: str  # Parent document ID
    content: str  # Chunk text content
    start_char: int  # Starting character position in document
    end_char: int  # Ending character position in document
    heading: Optional[str] = None  # Section heading if applicable
    chunk_index: int = 0  # Position within document
    embedding: Optional[np.ndarray] = None  # Chunk embedding
    entities: Set[str] = field(default_factory=set)  # Extracted entities
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class Entity:
    """
    Represents an extracted entity with metadata.
    
    Entities are the nodes in our knowledge graph, representing people,
    concepts, organizations, or any other important items mentioned in the vault.
    """
    id: str  # Unique entity identifier
    name: str  # Entity name/label
    type: str  # Entity type (person, concept, organization, etc.)
    description: str = ""  # Entity description
    mentions: List[str] = field(default_factory=list)  # Chunk IDs where mentioned
    aliases: Set[str] = field(default_factory=set)  # Alternative names
    properties: Dict[str, Any] = field(default_factory=dict)  # Additional properties
    centrality_score: float = 0.0  # Graph centrality measure
    embedding: Optional[np.ndarray] = None  # Entity embedding


@dataclass
class Relationship:
    """
    Represents a relationship between entities.
    
    Relationships are the edges in our knowledge graph, capturing how entities
    are connected and the nature of their connections.
    """
    id: str  # Unique relationship identifier
    source_entity: str  # Source entity ID
    target_entity: str  # Target entity ID
    relation_type: str  # Type of relationship
    description: str = ""  # Relationship description
    strength: float = 1.0  # Relationship strength/weight
    evidence: List[str] = field(default_factory=list)  # Chunk IDs providing evidence
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class Community:
    """
    Represents a community of related entities in the knowledge graph.
    
    Communities are groups of highly connected entities that often represent
    coherent topics or themes within the vault.
    """
    id: str  # Unique community identifier
    entities: Set[str] = field(default_factory=set)  # Entity IDs in community
    title: str = ""  # Community title/theme
    summary: str = ""  # Community summary
    level: int = 0  # Hierarchical level (for hierarchical community detection)
    size: int = 0  # Number of entities
    density: float = 0.0  # Internal connection density


@dataclass
class QueryResult:
    """
    Represents the result of a query to the Graph RAG system.
    
    This structure contains both the generated answer and all the supporting
    information used to create that answer, enabling transparency and debugging.
    """
    query: str  # Original query
    answer: str  # Generated answer
    retrieved_chunks: List[TextChunk] = field(default_factory=list)  # Retrieved text chunks
    relevant_entities: List[Entity] = field(default_factory=list)  # Relevant entities
    relevant_relationships: List[Relationship] = field(default_factory=list)  # Relevant relationships
    communities: List[Community] = field(default_factory=list)  # Relevant communities
    reasoning_steps: List[str] = field(default_factory=list)  # Sequential thinking steps
    citations: List[str] = field(default_factory=list)  # Source citations
    confidence_score: float = 0.0  # Answer confidence
    processing_time: float = 0.0  # Query processing time
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


# =============================================================================
# SEQUENTIAL THINKING INTEGRATION
# =============================================================================

class SequentialThinkingOrchestrator:
    """
    Orchestrates complex reasoning using the sequential thinking approach.
    
    This class manages multi-step reasoning processes, breaking down complex
    queries into manageable steps and coordinating the retrieval and reasoning
    needed for each step.
    
    The sequential thinking process involves:
    1. Query decomposition and planning
    2. Iterative information gathering
    3. Step-by-step reasoning
    4. Answer synthesis and validation
    """
    
    def __init__(self, config: GraphRAGConfig, client: OpenAI):
        """
        Initialize the sequential thinking orchestrator.
        
        Args:
            config: Graph RAG configuration
            client: OpenAI client for LLM operations
        """
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # Prompt templates for different thinking operations
        self.decomposition_prompt = """
        Given the following query, break it down into logical steps that need to be addressed
        to provide a comprehensive answer. Consider what information needs to be retrieved
        and what reasoning needs to be performed.
        
        Query: {query}
        
        Provide your decomposition as a numbered list of steps, each with:
        1. What information is needed
        2. What type of search to perform (local entity search, global theme search, etc.)
        3. What reasoning or analysis is required
        
        Steps:
        """
        
        self.reasoning_prompt = """
        Based on the retrieved information below, perform the reasoning step: {step}
        
        Retrieved Information:
        {context}
        
        Current reasoning chain:
        {previous_steps}
        
        Provide your reasoning for this step, building on previous steps if applicable:
        """
        
        self.synthesis_prompt = """
        Based on all the reasoning steps below, synthesize a comprehensive answer to the original query.
        Include proper citations and ensure the answer is well-structured and complete.
        
        Original Query: {query}
        
        Reasoning Steps:
        {reasoning_steps}
        
        All Retrieved Context:
        {all_context}
        
        Final Answer:
        """
    
    async def run_sequential_thinking(self, query: str, retrieval_function) -> Tuple[str, List[str]]:
        """
        Run the complete sequential thinking process for a given query.
        
        Args:
            query: The user's query
            retrieval_function: Function to retrieve relevant information
            
        Returns:
            Tuple of (final_answer, reasoning_steps)
        """
        try:
            # Step 1: Decompose the query into logical steps
            reasoning_steps = []
            decomposition = await self._decompose_query(query)
            reasoning_steps.append(f"Query Decomposition: {decomposition}")
            
            # Step 2: Execute each step iteratively
            all_context = []
            for i, step in enumerate(decomposition.split('\n')):
                if step.strip() and step[0].isdigit():
                    # Retrieve information for this step
                    step_context = await retrieval_function(step, previous_context=all_context)
                    all_context.extend(step_context)
                    
                    # Perform reasoning for this step
                    step_reasoning = await self._reason_step(query, step, step_context, reasoning_steps)
                    reasoning_steps.append(f"Step {i+1} Reasoning: {step_reasoning}")
                    
                    # Check if we need more steps based on the reasoning
                    if len(reasoning_steps) >= self.config.max_reasoning_steps:
                        break
            
            # Step 3: Synthesize final answer
            final_answer = await self._synthesize_answer(query, reasoning_steps, all_context)
            
            return final_answer, reasoning_steps
            
        except Exception as e:
            self.logger.error(f"Error in sequential thinking: {e}")
            return f"I encountered an error while processing your query: {e}", reasoning_steps
    
    async def _decompose_query(self, query: str) -> str:
        """Decompose a complex query into logical steps."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{
                    "role": "user", 
                    "content": self.decomposition_prompt.format(query=query)
                }],
                temperature=self.config.thinking_temperature,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error decomposing query: {e}")
            return f"1. Search for information related to: {query}\n2. Analyze and synthesize findings"
    
    async def _reason_step(self, query: str, step: str, context: List[str], previous_steps: List[str]) -> str:
        """Perform reasoning for a specific step."""
        try:
            context_str = "\n".join(context) if context else "No specific context retrieved for this step."
            previous_str = "\n".join(previous_steps) if previous_steps else "This is the first reasoning step."
            
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{
                    "role": "user",
                    "content": self.reasoning_prompt.format(
                        step=step,
                        context=context_str,
                        previous_steps=previous_str
                    )
                }],
                temperature=self.config.thinking_temperature,
                max_tokens=800
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in step reasoning: {e}")
            return f"Unable to complete reasoning for step: {step}"
    
    async def _synthesize_answer(self, query: str, reasoning_steps: List[str], all_context: List[str]) -> str:
        """Synthesize the final answer from all reasoning steps."""
        try:
            steps_str = "\n".join(reasoning_steps)
            context_str = "\n".join(all_context) if all_context else "No additional context available."
            
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{
                    "role": "user",
                    "content": self.synthesis_prompt.format(
                        query=query,
                        reasoning_steps=steps_str,
                        all_context=context_str
                    )
                }],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error synthesizing answer: {e}")
            return f"I was able to gather information but encountered an error while synthesizing the final answer: {e}"


# =============================================================================
# DATA PERSISTENCE MANAGER
# =============================================================================

class DataPersistenceManager:
    """
    Manages saving and loading of processed data to avoid reprocessing.
    
    This class handles the persistence of:
    - Processed documents
    - Knowledge graph
    - Document embeddings
    - Processing metadata and timestamps
    
    This allows users to skip the time-consuming processing steps when
    the vault hasn't changed significantly.
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the persistence manager.
        
        Args:
            config: Graph RAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data file paths
        self.data_dir = Path(config.cache_dir) / "processed_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.documents_file = self.data_dir / "documents.pkl"
        self.graph_file = self.data_dir / "knowledge_graph.gpickle"
        self.embeddings_file = self.data_dir / "embeddings.pkl"
        self.metadata_file = self.data_dir / "metadata.json"
    
    def detect_changes(self, vault_path: str) -> Tuple[List[Path], List[Path], List[str], bool]:
        """
        Detect which files have changed since last processing.
        
        Args:
            vault_path: Path to the vault
            
        Returns:
            Tuple of (new_files, modified_files, deleted_file_paths, should_rebuild_all)
        """
        if not self.metadata_file.exists():
            self.logger.info("No previous data found, will build from scratch")
            return [], [], [], True
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if vault path changed
            if metadata.get('vault_path') != vault_path:
                self.logger.info("Vault path changed, rebuilding all data")
                return [], [], [], True
            
            # Check if data files exist
            required_files = [self.documents_file, self.graph_file, self.embeddings_file]
            if not all(f.exists() for f in required_files):
                self.logger.info("Some data files missing, rebuilding all data")
                return [], [], [], True
            
            # Get file metadata from cache (if available)
            cached_file_metadata = metadata.get('file_metadata', {})
            
            # If no file metadata in cache, fall back to rebuild
            if not cached_file_metadata:
                self.logger.info("🔄 Upgrading cache format for incremental processing (one-time rebuild)")
                self.logger.info("💡 After this upgrade, only new/modified files will be processed")
                return [], [], [], True
            
            # Scan current vault files
            current_files = list(Path(vault_path).rglob("*.md"))
            current_file_metadata = {}
            
            for file_path in current_files:
                try:
                    stat = file_path.stat()
                    relative_path = str(file_path.relative_to(Path(vault_path)))
                    current_file_metadata[relative_path] = {
                        'mtime': stat.st_mtime,
                        'size': stat.st_size
                    }
                except (OSError, ValueError) as e:
                    self.logger.warning(f"Could not stat file {file_path}: {e}")
                    continue
            
            # Find changes
            new_files = []
            modified_files = []
            deleted_files = []
            
            # Check for new and modified files
            for relative_path, current_meta in current_file_metadata.items():
                if relative_path not in cached_file_metadata:
                    # New file
                    new_files.append(Path(vault_path) / relative_path)
                else:
                    # Check if modified
                    cached_meta = cached_file_metadata[relative_path]
                    if (current_meta['mtime'] != cached_meta.get('mtime') or 
                        current_meta['size'] != cached_meta.get('size')):
                        modified_files.append(Path(vault_path) / relative_path)
            
            # Check for deleted files
            for relative_path in cached_file_metadata:
                if relative_path not in current_file_metadata:
                    deleted_files.append(relative_path)
            
            total_changes = len(new_files) + len(modified_files) + len(deleted_files)
            
            if total_changes == 0:
                self.logger.info("No file changes detected")
                return [], [], [], False
            
            self.logger.info(f"Detected changes: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted files")
            return new_files, modified_files, deleted_files, False
            
        except Exception as e:
            self.logger.warning(f"Error detecting changes, will rebuild: {e}")
            return [], [], [], True
    
    def prompt_user_for_update(self, new_files: List[Path], modified_files: List[Path], deleted_files: List[str]) -> bool:
        """
        Prompt user whether they want to update the graph with detected changes.
        
        Args:
            new_files: List of new files detected
            modified_files: List of modified files detected
            deleted_files: List of deleted file paths
            
        Returns:
            True if user wants to update
        """
        total_changes = len(new_files) + len(modified_files) + len(deleted_files)
        
        print(f"\n📊 Graph Update Available")
        print("=" * 50)
        print(f"Changes detected in your vault:")
        
        if new_files:
            print(f"  📄 {len(new_files)} new files")
            if len(new_files) <= 5:
                for file in new_files:
                    print(f"    + {file.name}")
            else:
                for file in new_files[:3]:
                    print(f"    + {file.name}")
                print(f"    + ... and {len(new_files) - 3} more")
        
        if modified_files:
            print(f"  ✏️  {len(modified_files)} modified files")
            if len(modified_files) <= 5:
                for file in modified_files:
                    print(f"    ~ {file.name}")
            else:
                for file in modified_files[:3]:
                    print(f"    ~ {file.name}")
                print(f"    ~ ... and {len(modified_files) - 3} more")
        
        if deleted_files:
            print(f"  🗑️  {len(deleted_files)} deleted files")
            if len(deleted_files) <= 5:
                for file_path in deleted_files:
                    print(f"    - {Path(file_path).name}")
            else:
                for file_path in deleted_files[:3]:
                    print(f"    - {Path(file_path).name}")
                print(f"    - ... and {len(deleted_files) - 3} more")
        
        print("=" * 50)
        print("Options:")
        print("  1. ✅ Update graph incrementally (recommended)")
        print("  2. 🚫 Skip update (use existing graph)")
        print("  3. 🔄 Rebuild entire graph from scratch")
        
        while True:
            try:
                choice = input("\nSelect option (1-3): ").strip()
                
                if choice == "1":
                    return True
                elif choice == "2":
                    print("Skipping update, using existing graph data.")
                    return False
                elif choice == "3":
                    print("Will rebuild entire graph from scratch.")
                    return "rebuild"
                else:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue
                    
            except KeyboardInterrupt:
                print("\nUpdate cancelled, using existing graph data.")
                return False
    
    def save_data(self, documents: Dict[str, Document], knowledge_graph: nx.Graph, 
                  embeddings: Dict[str, np.ndarray], vault_path: str) -> None:
        """
        Save all processed data to disk.
        
        Args:
            documents: Processed documents
            knowledge_graph: Built knowledge graph
            embeddings: Document embeddings
            vault_path: Vault path for metadata
        """
        try:
            self.logger.info("Saving processed data to disk...")
            
            # Save documents
            with open(self.documents_file, 'wb') as f:
                pickle.dump(documents, f)
            
            # Save knowledge graph
            with open(self.graph_file, 'wb') as f:
                pickle.dump(knowledge_graph, f)
            
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Build file metadata for incremental processing
            file_metadata = {}
            vault_path_obj = Path(vault_path)
            
            for doc in documents.values():
                try:
                    file_path = Path(doc.path)
                    if file_path.exists():
                        stat = file_path.stat()
                        relative_path = str(file_path.relative_to(vault_path_obj))
                        file_metadata[relative_path] = {
                            'mtime': stat.st_mtime,
                            'size': stat.st_size,
                            'processed_at': datetime.now().isoformat()
                        }
                except (OSError, ValueError) as e:
                    self.logger.warning(f"Could not get metadata for {doc.path}: {e}")
                    continue
            
            # Save metadata
            metadata = {
                'vault_path': vault_path,
                'file_count': len(documents),
                'graph_nodes': knowledge_graph.number_of_nodes(),
                'graph_edges': knowledge_graph.number_of_edges(),
                'embeddings_count': len(embeddings),
                'file_metadata': file_metadata,
                'created_at': datetime.now().isoformat(),
                'version': '2.0'  # Bumped version for incremental processing
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Data saved successfully to {self.data_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
    
    def incremental_update(self, new_files: List[Path], modified_files: List[Path], 
                          deleted_files: List[str], vault_path: str, 
                          graph_rag_system) -> None:
        """
        Perform incremental update by processing only changed files.
        
        Args:
            new_files: List of new files to process
            modified_files: List of modified files to process
            deleted_files: List of deleted file paths to remove
            vault_path: Path to the vault
            graph_rag_system: Reference to main GraphRAG system
        """
        try:
            # Load existing data
            documents, knowledge_graph, embeddings = self.load_data()
            
            # Initialize components for processing
            parser = MarkdownParser(graph_rag_system.config)
            link_extractor = LinkExtractor(graph_rag_system.config)
            
            # Process deleted files
            for relative_path in deleted_files:
                # Find and remove documents
                docs_to_remove = []
                for doc_id, doc in documents.items():
                    if doc.path.endswith(relative_path.replace('/', os.sep)):
                        docs_to_remove.append(doc_id)
                
                for doc_id in docs_to_remove:
                    del documents[doc_id]
                    # Remove from graph if it exists
                    if knowledge_graph.has_node(doc_id):
                        knowledge_graph.remove_node(doc_id)
                    # Remove embeddings if they exist
                    if doc_id in embeddings:
                        del embeddings[doc_id]
                
                self.logger.info(f"Removed {len(docs_to_remove)} documents for deleted file: {relative_path}")
            
            # Process new and modified files
            files_to_process = new_files + modified_files
            processed_count = 0
            
            for file_path in files_to_process:
                try:
                    # Parse document
                    document = parser.parse_file(file_path)
                    
                    # Extract links
                    document = link_extractor.extract_all_links(document)
                    
                    # If this is a modified file, remove old version from graph
                    if file_path in modified_files and knowledge_graph.has_node(document.id):
                        knowledge_graph.remove_node(document.id)
                    
                    # Store document
                    documents[document.id] = document
                    
                    # Add to graph
                    knowledge_graph.add_node(
                        document.id,
                        type='document',
                        title=document.title,
                        path=document.path,
                        word_count=document.word_count,
                        tags=list(document.tags)
                    )
                    
                    processed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            # Rebuild graph connections for new/modified documents
            self._rebuild_graph_connections(documents, knowledge_graph, files_to_process)
            
            # Save updated data
            self.save_data(documents, knowledge_graph, embeddings, vault_path)
            
            # Update the system's data
            graph_rag_system.documents = documents
            graph_rag_system.knowledge_graph = knowledge_graph
            
            self.logger.info(f"Incremental update complete: processed {processed_count} files, "
                           f"removed {len(deleted_files)} files")
            
        except Exception as e:
            self.logger.error(f"Error during incremental update: {e}")
            raise
    
    def _rebuild_graph_connections(self, documents: Dict[str, Document], 
                                 knowledge_graph: nx.Graph, 
                                 processed_files: List[Path]) -> None:
        """
        Rebuild graph connections for processed files and their targets.
        
        Args:
            documents: All documents
            knowledge_graph: Knowledge graph to update
            processed_files: Files that were just processed
        """
        # Get document IDs for processed files
        processed_doc_ids = set()
        for file_path in processed_files:
            for doc_id, doc in documents.items():
                if Path(doc.path) == file_path:
                    processed_doc_ids.add(doc_id)
                    break
        
        # Rebuild connections for processed documents
        edges_added = 0
        for doc_id in processed_doc_ids:
            if doc_id not in documents:
                continue
                
            document = documents[doc_id]
            
            # Remove existing edges from this document
            if knowledge_graph.has_node(doc_id):
                edges_to_remove = list(knowledge_graph.edges(doc_id))
                knowledge_graph.remove_edges_from(edges_to_remove)
            
            # Add new edges based on wikilinks
            for wikilink in document.wikilinks:
                target_doc = self._find_document_by_title(wikilink, documents)
                if target_doc and target_doc.id != doc_id:
                    knowledge_graph.add_edge(
                        doc_id,
                        target_doc.id,
                        type='wikilink',
                        source_title=document.title,
                        target_title=target_doc.title
                    )
                    edges_added += 1
                    
                    # Update backlinks
                    target_doc.backlinks.add(document.title)
        
        self.logger.info(f"Rebuilt graph connections: {edges_added} edges added")
    
    def _find_document_by_title(self, title: str, documents: Dict[str, Document]) -> Optional[Document]:
        """
        Find a document by its title from the documents dictionary.
        
        Args:
            title: Document title to search for
            documents: Dictionary of documents
            
        Returns:
            Document object if found, None otherwise
        """
        # Direct title match
        for document in documents.values():
            if document.title.lower() == title.lower():
                return document
        
        # Try filename match (without extension)
        for document in documents.values():
            file_stem = Path(document.path).stem
            if file_stem.lower() == title.lower():
                return document
        
        return None
    
    def load_data(self) -> Tuple[Dict[str, Document], nx.Graph, Dict[str, np.ndarray]]:
        """
        Load all processed data from disk with improved error handling and diagnostics.
        
        Returns:
            Tuple of (documents, knowledge_graph, embeddings)
        """
        try:
            self.logger.info("Loading processed data from disk...")
            
            # Check if all required files exist
            missing_files = []
            required_files = [
                (self.documents_file, "documents.pkl"),
                (self.graph_file, "knowledge_graph.gpickle"),
                (self.embeddings_file, "embeddings.pkl"),
                (self.metadata_file, "metadata.json")
            ]
            
            for file_path, file_name in required_files:
                if not file_path.exists():
                    missing_files.append(f"{file_name} ({file_path})")
            
            if missing_files:
                error_msg = f"Missing cache files: {', '.join(missing_files)}"
                self.logger.error(error_msg)
                self.logger.info(f"Cache directory: {self.data_dir}")
                self.logger.info(f"Available files: {list(self.data_dir.glob('*'))}")
                raise FileNotFoundError(error_msg)
            
            # Load documents with error handling
            try:
                self.logger.info(f"Loading documents from {self.documents_file}")
                with open(self.documents_file, 'rb') as f:
                    documents = pickle.load(f)
                self.logger.info(f"✓ Loaded {len(documents)} documents")
            except Exception as e:
                self.logger.error(f"Failed to load documents: {e}")
                raise Exception(f"Cannot load documents.pkl: {e}")
            
            # Load knowledge graph with error handling
            try:
                self.logger.info(f"Loading knowledge graph from {self.graph_file}")
                with open(self.graph_file, 'rb') as f:
                    knowledge_graph = pickle.load(f)
                self.logger.info(f"✓ Loaded graph with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges")
            except Exception as e:
                self.logger.error(f"Failed to load knowledge graph: {e}")
                raise Exception(f"Cannot load knowledge_graph.gpickle: {e}")
            
            # Load embeddings with error handling
            try:
                self.logger.info(f"Loading embeddings from {self.embeddings_file}")
                with open(self.embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.logger.info(f"✓ Loaded {len(embeddings)} embeddings")
            except Exception as e:
                self.logger.error(f"Failed to load embeddings: {e}")
                raise Exception(f"Cannot load embeddings.pkl: {e}")
            
            # Validate data consistency
            self._validate_loaded_data(documents, knowledge_graph, embeddings)
            
            self.logger.info("✓ All data loaded successfully!")
            return documents, knowledge_graph, embeddings
            
        except Exception as e:
            self.logger.error(f"Critical error loading data: {e}")
            self.logger.info("💡 To fix this issue:")
            self.logger.info("   1. Run the diagnostic script: python check_system.py")
            self.logger.info("   2. Or delete the cache folder to rebuild: rm -rf ./cache")
            self.logger.info("   3. Or run the system with rebuild option")
            raise
    
    def _validate_loaded_data(self, documents: Dict[str, Document], 
                            knowledge_graph: nx.Graph, 
                            embeddings: Dict[str, np.ndarray]) -> None:
        """
        Validate that loaded data is consistent and complete.
        
        Args:
            documents: Loaded documents
            knowledge_graph: Loaded knowledge graph
            embeddings: Loaded embeddings
        """
        issues = []
        
        # Check document-graph consistency
        doc_ids = set(documents.keys())
        graph_nodes = set(knowledge_graph.nodes())
        
        if doc_ids != graph_nodes:
            missing_in_graph = doc_ids - graph_nodes
            extra_in_graph = graph_nodes - doc_ids
            
            if missing_in_graph:
                issues.append(f"Documents missing from graph: {len(missing_in_graph)} items")
            if extra_in_graph:
                issues.append(f"Extra nodes in graph: {len(extra_in_graph)} items")
        
        # Check embeddings consistency (embeddings might be partial)
        docs_with_embeddings = set(embeddings.keys())
        missing_embeddings = doc_ids - docs_with_embeddings
        
        if missing_embeddings:
            self.logger.warning(f"⚠️  {len(missing_embeddings)} documents missing embeddings (will generate as needed)")
        
        if issues:
            warning_msg = "Data consistency issues detected: " + "; ".join(issues)
            self.logger.warning(f"⚠️  {warning_msg}")
        else:
            self.logger.info("✓ Data consistency validated")
    
    def diagnose_system_status(self, vault_path: str) -> Dict[str, Any]:
        """
        Comprehensive system diagnosis for troubleshooting.
        
        Args:
            vault_path: Path to the vault
            
        Returns:
            Dictionary with diagnostic information
        """
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'vault_path': vault_path,
            'cache_dir': str(self.data_dir),
            'files': {},
            'vault_info': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check vault
            vault_path_obj = Path(vault_path)
            if vault_path_obj.exists():
                md_files = list(vault_path_obj.rglob("*.md"))
                diagnosis['vault_info'] = {
                    'exists': True,
                    'markdown_files': len(md_files),
                    'total_size_mb': round(sum(f.stat().st_size for f in md_files if f.exists()) / 1024 / 1024, 2)
                }
            else:
                diagnosis['vault_info'] = {'exists': False}
                diagnosis['issues'].append(f"Vault path does not exist: {vault_path}")
            
            # Check cache directory
            if not self.data_dir.exists():
                diagnosis['issues'].append(f"Cache directory does not exist: {self.data_dir}")
                diagnosis['recommendations'].append("Run initial vault processing to create cache")
                return diagnosis
            
            # Check each required file
            required_files = [
                ('documents.pkl', self.documents_file),
                ('knowledge_graph.gpickle', self.graph_file),
                ('embeddings.pkl', self.embeddings_file),
                ('metadata.json', self.metadata_file)
            ]
            
            for file_name, file_path in required_files:
                file_info = {
                    'exists': file_path.exists(),
                    'path': str(file_path)
                }
                
                if file_path.exists():
                    try:
                        stat = file_path.stat()
                        file_info.update({
                            'size_mb': round(stat.st_size / 1024 / 1024, 2),
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            'readable': True
                        })
                        
                        # Try to peek into the file
                        if file_name == 'metadata.json':
                            try:
                                with open(file_path, 'r') as f:
                                    metadata = json.load(f)
                                file_info['content_preview'] = {
                                    'file_count': metadata.get('file_count', 'unknown'),
                                    'graph_nodes': metadata.get('graph_nodes', 'unknown'),
                                    'created_at': metadata.get('created_at', 'unknown')
                                }
                            except Exception as e:
                                file_info['read_error'] = str(e)
                                diagnosis['issues'].append(f"Cannot read {file_name}: {e}")
                        
                    except Exception as e:
                        file_info['stat_error'] = str(e)
                        diagnosis['issues'].append(f"Cannot access {file_name}: {e}")
                else:
                    diagnosis['issues'].append(f"Missing file: {file_name}")
                
                diagnosis['files'][file_name] = file_info
            
            # Generate recommendations
            if diagnosis['issues']:
                if any('Missing file' in issue for issue in diagnosis['issues']):
                    diagnosis['recommendations'].append("Some cache files are missing. Consider rebuilding the cache.")
                if any('Cannot read' in issue for issue in diagnosis['issues']):
                    diagnosis['recommendations'].append("Some cache files are corrupted. Delete cache directory and rebuild.")
                if any('Cannot access' in issue for issue in diagnosis['issues']):
                    diagnosis['recommendations'].append("File permission issues detected. Check file permissions.")
            else:
                diagnosis['recommendations'].append("System appears healthy. Try running the chat interface.")
            
        except Exception as e:
            diagnosis['issues'].append(f"Diagnostic error: {e}")
        
        return diagnosis


# =============================================================================
# MAIN GRAPH RAG SYSTEM CLASS
# =============================================================================

class ObsidianGraphRAG:
    """
    Main Graph RAG system for Obsidian vaults.
    
    This is the primary interface for the Graph RAG system. It orchestrates
    all components including document processing, graph construction, retrieval,
    and generation to provide intelligent question-answering capabilities
    over Obsidian vaults.
    
    The system implements a hybrid approach combining:
    - Vector-based semantic search for content similarity
    - Graph-based traversal for relationship discovery
    - Sequential thinking for complex reasoning
    - Community detection for thematic understanding
    """
    
    def __init__(self, config: GraphRAGConfig = None):
        """
        Initialize the Graph RAG system.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or GraphRAGConfig()
        self._setup_logging()
        
        # Initialize OpenAI client (optional for Phase 2 testing)
        if self.config.openai_api_key:
            self.client = OpenAI(api_key=self.config.openai_api_key)
            self.sequential_thinking = SequentialThinkingOrchestrator(self.config, self.client)
        else:
            self.logger.warning("OpenAI API key not provided. Some features will be limited.")
            self.client = None
            self.sequential_thinking = None
        
        # Initialize core components
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, TextChunk] = {}
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.communities: Dict[str, Community] = {}
        
        # Initialize graph
        self.knowledge_graph = nx.Graph()
        
        # Vector storage (simplified - in production, use a proper vector database)
        self.chunk_embeddings: Dict[str, np.ndarray] = {}
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        
        # Create necessary directories
        self._create_directories()
        
        self.logger.info("ObsidianGraphRAG initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        # Create log directory if it doesn't exist
        log_path = Path(self.config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging with proper encoding for Windows
        import sys
        import io
        
        # Create a custom stream handler with UTF-8 encoding for console output
        console_handler = logging.StreamHandler(sys.stdout)
        if sys.platform == "win32":
            # For Windows, wrap stdout to handle Unicode properly
            console_handler.stream = io.TextIOWrapper(
                sys.stdout.buffer, encoding='utf-8', errors='replace'
            )
        console_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(self.config.log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, self.config.log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            handlers=[console_handler, file_handler],
            force=True  # Override any existing configuration
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _create_directories(self):
        """Create necessary directories for data storage."""
        directories = [
            Path(self.config.cache_dir),
            Path(self.config.vector_db_path).parent,
            Path(self.config.graph_db_path).parent,
            Path(self.config.log_file).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # PHASE 2: DOCUMENT PROCESSING & PARSING
    # =============================================================================

    def scan_and_parse_vault(self) -> None:
        """
        Scan the vault and parse all documents to build initial graph structure.
        
        This is the main entry point for Phase 2 processing. It orchestrates
        the vault scanning, document parsing, link extraction, and initial
        graph construction processes.
        """
        try:
            self.logger.info("Starting vault scan and parsing process")
            
            # Initialize components
            scanner = VaultScanner(self.config)
            parser = MarkdownParser(self.config)
            link_extractor = LinkExtractor(self.config)
            
            # Scan vault for files
            discovered_files = scanner.scan_vault()
            self.logger.info(f"Found {len(discovered_files)} files to process")
            
            # Process each file
            processed_count = 0
            for file_path in discovered_files:
                try:
                    # Parse document
                    document = parser.parse_file(file_path)
                    
                    # Extract links
                    document = link_extractor.extract_all_links(document)
                    
                    # Store document
                    self.documents[document.id] = document
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        self.logger.info(f"Processed {processed_count}/{len(discovered_files)} files")
                
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            # Build initial graph connections
            self._build_document_graph()
            
            self.logger.info(f"Vault processing complete. Processed {processed_count} documents")
            
        except Exception as e:
            self.logger.error(f"Error during vault scan and parsing: {e}")
            raise
    
    def _build_document_graph(self) -> None:
        """
        Build the initial knowledge graph from document connections.
        
        This method creates the graph structure based on the links found
        between documents. It creates nodes for each document and edges
        for each wikilink connection.
        """
        try:
            self.logger.info("Building document connection graph")
            
            # Add document nodes to graph
            for doc_id, document in self.documents.items():
                self.knowledge_graph.add_node(
                    doc_id,
                    type='document',
                    title=document.title,
                    path=document.path,
                    word_count=document.word_count,
                    tags=list(document.tags)
                )
            
            # Add edges for wikilink connections
            edges_added = 0
            for doc_id, document in self.documents.items():
                for wikilink in document.wikilinks:
                    # Find target document by title
                    target_doc = self._find_document_by_title(wikilink)
                    if target_doc:
                        self.knowledge_graph.add_edge(
                            doc_id,
                            target_doc.id,
                            type='wikilink',
                            source_title=document.title,
                            target_title=target_doc.title
                        )
                        edges_added += 1
                        
                        # Update backlinks
                        target_doc.backlinks.add(document.title)
            
            self.logger.info(f"Graph built: {len(self.documents)} nodes, {edges_added} edges")
            
        except Exception as e:
            self.logger.error(f"Error building document graph: {e}")
            raise
    
    def _find_document_by_title(self, title: str) -> Optional[Document]:
        """
        Find a document by its title, handling various matching strategies.
        
        Args:
            title: Document title to search for
            
        Returns:
            Document object if found, None otherwise
        """
        # Direct title match
        for document in self.documents.values():
            if document.title.lower() == title.lower():
                return document
        
        # Try filename match (without extension)
        for document in self.documents.values():
            file_stem = Path(document.path).stem
            if file_stem.lower() == title.lower():
                return document
        
        return None

    async def start_chat_session(self) -> None:
        """
        Start an interactive chat session with the AI librarian.
        
        This method initializes the chat bot and starts the conversation loop.
        It requires that documents have been processed and the knowledge graph
        has been built.
        """
        try:
            if not self.documents:
                raise ValueError("No documents loaded. Please run scan_and_parse_vault() first.")
            
            if not self.client:
                raise ValueError("OpenAI client not available. Please set your API key.")
            
            # Initialize chat bot with enhanced features
            chat_bot = ObsidianChatBot(self)
            chat_bot.initialize()
            
            # Setup enhanced features in retriever
            if hasattr(chat_bot.retriever, 'setup_enhanced_features'):
                await chat_bot.retriever.setup_enhanced_features(self.documents, self.knowledge_graph)
            
            # Start chat session
            await chat_bot.start_chat()
            
        except Exception as e:
            self.logger.error(f"Error starting chat session: {e}")
            raise



    def initialize_system(self) -> None:
        """
        Initialize the system with data persistence support.
        
        This method checks if processed data exists and loads it if valid,
        otherwise it processes the vault from scratch. Supports incremental
        processing for efficient updates.
        """
        try:
            # Initialize persistence manager
            persistence_manager = DataPersistenceManager(self.config)
            
            # Detect changes in the vault
            new_files, modified_files, deleted_files, should_rebuild_all = persistence_manager.detect_changes(
                self.config.vault_path
            )
            
            if should_rebuild_all:
                self.logger.info("Processing vault from scratch...")
                
                # Process vault completely
                self.scan_and_parse_vault()
                
                # Save processed data
                embeddings_dict = {}  # Will be populated later when needed
                persistence_manager.save_data(
                    self.documents, 
                    self.knowledge_graph, 
                    embeddings_dict, 
                    self.config.vault_path
                )
                
            elif new_files or modified_files or deleted_files:
                # Changes detected - prompt user
                user_choice = persistence_manager.prompt_user_for_update(
                    new_files, modified_files, deleted_files
                )
                
                if user_choice == "rebuild":
                    # User chose to rebuild everything
                    self.logger.info("Rebuilding entire vault as requested...")
                    self.scan_and_parse_vault()
                    
                    embeddings_dict = {}
                    persistence_manager.save_data(
                        self.documents, 
                        self.knowledge_graph, 
                        embeddings_dict, 
                        self.config.vault_path
                    )
                    
                elif user_choice:
                    # User chose incremental update
                    self.logger.info("Performing incremental update...")
                    persistence_manager.incremental_update(
                        new_files, modified_files, deleted_files,
                        self.config.vault_path, self
                    )
                    
                else:
                    # User chose to skip update - load existing data
                    try:
                        self.logger.info("Loading existing processed data...")
                        self.documents, self.knowledge_graph, embeddings_dict = persistence_manager.load_data()
                    except Exception as load_error:
                        self.logger.error(f"Failed to load existing data: {load_error}")
                        self._handle_loading_failure(persistence_manager, load_error)
                        return
                    
            else:
                # No changes detected - load existing data
                try:
                    self.logger.info("No changes detected, loading existing processed data...")
                    self.documents, self.knowledge_graph, embeddings_dict = persistence_manager.load_data()
                except Exception as load_error:
                    self.logger.error(f"Failed to load existing data: {load_error}")
                    self._handle_loading_failure(persistence_manager, load_error)
                    return
                
            self.logger.info("✓ System initialization completed!")
        
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise
    
    def _handle_loading_failure(self, persistence_manager: 'DataPersistenceManager', error: Exception) -> None:
        """
        Handle failures when loading cached data by offering recovery options.
        
        Args:
            persistence_manager: Persistence manager instance
            error: The loading error that occurred
        """
        self.logger.error("=" * 60)
        self.logger.error("🚨 CACHE LOADING FAILED")
        self.logger.error("=" * 60)
        self.logger.error(f"Error: {error}")
        
        # Run diagnostics
        try:
            diagnosis = persistence_manager.diagnose_system_status(self.config.vault_path)
            
            self.logger.info("\n📊 System Diagnosis:")
            self.logger.info(f"   Vault: {diagnosis['vault_info']}")
            self.logger.info(f"   Cache Directory: {diagnosis['cache_dir']}")
            
            if diagnosis['issues']:
                self.logger.error(f"\n❌ Issues Found:")
                for issue in diagnosis['issues']:
                    self.logger.error(f"   - {issue}")
            
            if diagnosis['recommendations']:
                self.logger.info(f"\n💡 Recommendations:")
                for rec in diagnosis['recommendations']:
                    self.logger.info(f"   - {rec}")
                    
        except Exception as diag_error:
            self.logger.error(f"Diagnostic failed: {diag_error}")
        
        # Offer recovery options
        self.logger.info("\n🔧 Recovery Options:")
        self.logger.info("   1. Rebuild cache from scratch (recommended)")
        self.logger.info("   2. Delete cache and restart")
        self.logger.info("   3. Exit and troubleshoot manually")
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                self.logger.info("🔄 Rebuilding cache from scratch...")
                try:
                    self.scan_and_parse_vault()
                    embeddings_dict = {}
                    persistence_manager.save_data(
                        self.documents, 
                        self.knowledge_graph, 
                        embeddings_dict, 
                        self.config.vault_path
                    )
                    self.logger.info("✓ Cache rebuilt successfully!")
                except Exception as rebuild_error:
                    self.logger.error(f"❌ Rebuild failed: {rebuild_error}")
                    raise
                    
            elif choice == "2":
                cache_dir = Path(self.config.cache_dir)
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    self.logger.info(f"🗑️  Deleted cache directory: {cache_dir}")
                    self.logger.info("Please restart the application to rebuild the cache.")
                else:
                    self.logger.info(f"Cache directory {cache_dir} does not exist.")
                
            else:
                self.logger.info("Please troubleshoot the issue manually and restart.")
                
        except KeyboardInterrupt:
            self.logger.info("\nOperation cancelled.")
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            raise

    def save_embeddings(self, embeddings: Dict[str, np.ndarray]) -> None:
        """
        Save embeddings separately (they take time to generate).
        
        Args:
            embeddings: Document embeddings to save
        """
        try:
            embeddings_file = Path(self.config.cache_dir) / "processed_data" / "embeddings.pkl"
            with open(embeddings_file, 'wb') as f:
                pickle.dump(embeddings, f)
            self.logger.info("Embeddings saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")

    def load_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load saved embeddings.
        
        Returns:
            Dictionary of document embeddings
        """
        try:
            embeddings_file = Path(self.config.cache_dir) / "processed_data" / "embeddings.pkl"
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.logger.info(f"Loaded {len(embeddings)} embeddings from cache")
                return embeddings
            return {}
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            return {}


class VaultScanner:
    """
    Handles scanning and monitoring of Obsidian vault files.
    
    This class is responsible for discovering all markdown files in the vault,
    monitoring for changes, and providing efficient file system operations.
    It handles the complexities of Windows file paths, internationalization,
    and various folder structures that might exist in Obsidian vaults.
    
    Key Features:
    - Recursive directory scanning with pattern matching
    - Real-time file monitoring using watchdog
    - Efficient incremental processing
    - Robust error handling for file system issues
    - Support for international characters in file names
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the vault scanner.
        
        Args:
            config: Graph RAG configuration containing vault path and patterns
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Normalize the vault path for cross-platform compatibility
        self.vault_path = Path(config.vault_path).resolve()
        
        # File tracking for incremental updates
        self.file_cache: Dict[str, float] = {}  # file_path -> modification_time
        self.observer = None  # Will hold watchdog observer
        
        # Statistics tracking
        self.stats = {
            'files_found': 0,
            'files_processed': 0,
            'files_skipped': 0,
            'errors': 0
        }
        
        self.logger.info(f"VaultScanner initialized for: {self.vault_path}")
    
    def scan_vault(self) -> List[Path]:
        """
        Perform initial scan of the vault to discover all markdown files.
        
        This method recursively walks through the vault directory, applying
        include/exclude patterns to find relevant markdown files. It handles
        various edge cases like broken symlinks, permission issues, and
        special characters in file names.
        
        Returns:
            List of Path objects for discovered markdown files
        """
        discovered_files = []
        
        try:
            self.logger.info(f"Starting vault scan: {self.vault_path}")
            
            # Ensure vault path exists
            if not self.vault_path.exists():
                raise FileNotFoundError(f"Vault path does not exist: {self.vault_path}")
            
            # Walk through all files recursively
            for file_path in self.vault_path.rglob("*"):
                try:
                    # Skip directories
                    if file_path.is_dir():
                        continue
                    
                    # Check if file matches include patterns
                    if self._matches_include_pattern(file_path):
                        # Check if file should be excluded
                        if not self._matches_exclude_pattern(file_path):
                            discovered_files.append(file_path)
                            self.stats['files_found'] += 1
                        else:
                            self.stats['files_skipped'] += 1
                            self.logger.debug(f"Skipped excluded file: {file_path}")
                    
                except (PermissionError, OSError) as e:
                    self.logger.warning(f"Cannot access file {file_path}: {e}")
                    self.stats['errors'] += 1
                    continue
            
            self.logger.info(f"Vault scan complete. Found {len(discovered_files)} files")
            return discovered_files
            
        except Exception as e:
            self.logger.error(f"Error during vault scan: {e}")
            raise
    
    def _matches_include_pattern(self, file_path: Path) -> bool:
        """
        Check if file matches any include pattern.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be included
        """
        for pattern in self.config.file_patterns:
            if file_path.match(pattern):
                return True
        return False
    
    def _matches_exclude_pattern(self, file_path: Path) -> bool:
        """
        Check if file matches any exclude pattern.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file should be excluded
        """
        for pattern in self.config.exclude_patterns:
            if file_path.match(pattern) or str(file_path).find(pattern.replace('*', '')) != -1:
                return True
        return False
    
    def get_file_modification_time(self, file_path: Path) -> float:
        """
        Get file modification timestamp for change detection.
        
        Args:
            file_path: Path to file
            
        Returns:
            Modification timestamp
        """
        try:
            return file_path.stat().st_mtime
        except OSError as e:
            self.logger.warning(f"Cannot get modification time for {file_path}: {e}")
            return 0.0
    
    def has_file_changed(self, file_path: Path) -> bool:
        """
        Check if file has been modified since last scan.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file has changed or is new
        """
        current_mtime = self.get_file_modification_time(file_path)
        cached_mtime = self.file_cache.get(str(file_path), 0.0)
        
        if current_mtime > cached_mtime:
            self.file_cache[str(file_path)] = current_mtime
            return True
        return False


class MarkdownParser:
    """
    Parses individual Obsidian markdown files to extract content and metadata.
    
    This class handles the complexities of Obsidian's markdown format, including
    YAML frontmatter, various link formats, tags, and special Obsidian syntax.
    It's designed to be robust against malformed files and provides detailed
    error reporting for debugging.
    
    Key Features:
    - YAML frontmatter parsing with error handling
    - Content extraction with structure preservation
    - Metadata extraction (word count, creation date, etc.)
    - Encoding detection and handling
    - Special character and emoji support
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the markdown parser.
        
        Args:
            config: Graph RAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Patterns for parsing different elements
        self.frontmatter_pattern = re.compile(r'^---\s*\n(.*?)\n---\s*\n', re.DOTALL)
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.tag_pattern = re.compile(r'#([a-zA-Z0-9/_-]+)')
    
    def parse_file(self, file_path: Path) -> Document:
        """
        Parse a single markdown file and create a Document object.
        
        This method handles the complete parsing process:
        1. Read file with appropriate encoding detection
        2. Extract and parse YAML frontmatter
        3. Parse content and extract metadata
        4. Calculate statistics and create Document object
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Document object with parsed content and metadata
            
        Raises:
            Exception: If file cannot be parsed
        """
        try:
            self.logger.debug(f"Parsing file: {file_path}")
            
            # Read file content with encoding detection
            content = self._read_file_with_encoding(file_path)
            
            # Extract frontmatter if present
            frontmatter, content_without_frontmatter = self._extract_frontmatter(content)
            
            # Extract basic metadata
            title = self._extract_title(file_path, frontmatter, content_without_frontmatter)
            tags = self._extract_tags(frontmatter, content_without_frontmatter)
            
            # Get file system metadata
            file_stats = file_path.stat()
            created_at = datetime.fromtimestamp(file_stats.st_ctime)
            modified_at = datetime.fromtimestamp(file_stats.st_mtime)
            
            # Calculate word count
            word_count = len(content_without_frontmatter.split())
            
            # Generate unique document ID
            doc_id = self._generate_document_id(file_path)
            
            # Create Document object
            document = Document(
                id=doc_id,
                path=str(file_path),
                title=title,
                content=content_without_frontmatter,
                frontmatter=frontmatter,
                tags=tags,
                created_at=created_at,
                modified_at=modified_at,
                word_count=word_count
            )
            
            self.logger.debug(f"Successfully parsed: {title} ({word_count} words)")
            return document
            
        except Exception as e:
            self.logger.error(f"Error parsing file {file_path}: {e}")
            raise
    
    def _read_file_with_encoding(self, file_path: Path) -> str:
        """
        Read file content with automatic encoding detection.
        
        This method tries multiple encodings to handle international characters
        and various file formats that might exist in Obsidian vaults.
        
        Args:
            file_path: Path to file
            
        Returns:
            File content as string
        """
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                self.logger.warning(f"File {file_path} read with character replacement")
                return content
        except Exception as e:
            raise Exception(f"Cannot read file {file_path}: {e}")
    
    def _extract_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract and parse YAML frontmatter from markdown content.
        
        Args:
            content: Full markdown content
            
        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)
        """
        frontmatter = {}
        
        match = self.frontmatter_pattern.match(content)
        if match:
            try:
                # Parse YAML frontmatter
                yaml_content = match.group(1)
                frontmatter = yaml.safe_load(yaml_content) or {}
                
                # Remove frontmatter from content
                content_without_frontmatter = content[match.end():]
                
            except yaml.YAMLError as e:
                self.logger.warning(f"Invalid YAML frontmatter: {e}")
                content_without_frontmatter = content
        else:
            content_without_frontmatter = content
        
        return frontmatter, content_without_frontmatter
    
    def _extract_title(self, file_path: Path, frontmatter: Dict[str, Any], content: str) -> str:
        """
        Extract document title from various sources.
        
        Priority order:
        1. Title from frontmatter
        2. First H1 heading in content
        3. Filename without extension
        
        Args:
            file_path: Path to file
            frontmatter: Parsed frontmatter
            content: Document content
            
        Returns:
            Document title
        """
        # Try frontmatter title first
        if 'title' in frontmatter:
            return str(frontmatter['title'])
        
        # Try first H1 heading
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        # Fall back to filename
        return file_path.stem
    
    def _extract_tags(self, frontmatter: Dict[str, Any], content: str) -> Set[str]:
        """
        Extract tags from frontmatter and content.
        
        Args:
            frontmatter: Parsed frontmatter
            content: Document content
            
        Returns:
            Set of unique tags
        """
        tags = set()
        
        # Extract from frontmatter
        if 'tags' in frontmatter:
            fm_tags = frontmatter['tags']
            if isinstance(fm_tags, list):
                tags.update(str(tag) for tag in fm_tags)
            elif isinstance(fm_tags, str):
                tags.add(fm_tags)
        
        # Extract from content using regex
        content_tags = self.tag_pattern.findall(content)
        tags.update(content_tags)
        
        return tags
    
    def _generate_document_id(self, file_path: Path) -> str:
        """
        Generate unique document ID based on file path.
        
        Args:
            file_path: Path to file
            
        Returns:
            Unique document ID
        """
        return hashlib.md5(str(file_path).encode('utf-8')).hexdigest()


class LinkExtractor:
    """
    Extracts all types of links and references from Obsidian markdown content.
    
    This class handles the various link formats used in Obsidian:
    - Wikilinks: [[Note Name]], [[Note Name|Alias]], [[Note Name#Heading]]
    - Block references: [[Note Name^block-id]]
    - Embedded content: ![[Note Name]]
    - External links: [Text](URL)
    - Tags: #tag, #nested/tag
    
    The extracted links are used to build the knowledge graph connections
    that represent the relationships between documents in the vault.
    """
    
    def __init__(self, config: GraphRAGConfig):
        """
        Initialize the link extractor.
        
        Args:
            config: Graph RAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Regex patterns for different link types
        self.wikilink_pattern = re.compile(r'\[\[([^\]]+)\]\]')
        self.embed_pattern = re.compile(r'!\[\[([^\]]+)\]\]')
        self.external_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.tag_pattern = re.compile(r'#([a-zA-Z0-9/_-]+)')
        self.block_ref_pattern = re.compile(r'\^([a-zA-Z0-9-]+)')
    
    def extract_all_links(self, document: Document) -> Document:
        """
        Extract all links and references from a document and update the document object.
        
        This method processes the document content to find all types of links
        and references, then updates the document's wikilinks set with the
        discovered connections.
        
        Args:
            document: Document object to process
            
        Returns:
            Updated document object with extracted links
        """
        try:
            content = document.content
            
            # Extract wikilinks (including embedded content)
            wikilinks = self._extract_wikilinks(content)
            document.wikilinks.update(wikilinks)
            
            # Extract embedded content
            embeds = self._extract_embeds(content)
            # Embedded content also creates connections
            document.wikilinks.update(embeds)
            
            # Tags are already extracted in MarkdownParser, but we can validate here
            content_tags = self._extract_content_tags(content)
            document.tags.update(content_tags)
            
            self.logger.debug(f"Extracted {len(document.wikilinks)} links from {document.title}")
            return document
            
        except Exception as e:
            self.logger.error(f"Error extracting links from {document.title}: {e}")
            return document
    
    def _extract_wikilinks(self, content: str) -> Set[str]:
        """
        Extract wikilinks from content.
        
        Handles various wikilink formats:
        - [[Note Name]]
        - [[Note Name|Display Text]]
        - [[Note Name#Heading]]
        - [[Note Name#Heading|Display Text]]
        
        Args:
            content: Document content
            
        Returns:
            Set of referenced note names
        """
        wikilinks = set()
        
        matches = self.wikilink_pattern.findall(content)
        for match in matches:
            # Handle different wikilink formats
            link_parts = match.split('|')
            target = link_parts[0].strip()
            
            # Remove heading reference if present
            if '#' in target:
                target = target.split('#')[0].strip()
            
            # Remove block reference if present
            if '^' in target:
                target = target.split('^')[0].strip()
            
            if target:
                wikilinks.add(target)
        
        return wikilinks
    
    def _extract_embeds(self, content: str) -> Set[str]:
        """
        Extract embedded content references.
        
        Args:
            content: Document content
            
        Returns:
            Set of embedded note names
        """
        embeds = set()
        
        matches = self.embed_pattern.findall(content)
        for match in matches:
            # Similar processing to wikilinks
            target = match.split('|')[0].strip()
            if '#' in target:
                target = target.split('#')[0].strip()
            if target:
                embeds.add(target)
        
        return embeds
    
    def _extract_content_tags(self, content: str) -> Set[str]:
        """
        Extract hashtags from content.
        
        Args:
            content: Document content
            
        Returns:
            Set of tags
        """
        tags = set()
        matches = self.tag_pattern.findall(content)
        tags.update(matches)
        return tags


# =============================================================================
# PHASE 3: VECTOR EMBEDDINGS & SEMANTIC SEARCH
# =============================================================================

class EmbeddingManager:
    """
    Manages vector embeddings for documents and provides semantic search capabilities.
    
    This class handles the generation, storage, and retrieval of document embeddings
    for semantic similarity search. It works alongside the graph-based retrieval
    to provide hybrid Graph RAG functionality.
    """
    
    def __init__(self, config: GraphRAGConfig, client: OpenAI = None, graph_rag_system=None):
        """
        Initialize the embedding manager.
        
        Args:
            config: Graph RAG configuration
            client: OpenAI client for embedding generation
            graph_rag_system: Reference to main system for persistence
        """
        self.config = config
        self.client = client
        self.graph_rag_system = graph_rag_system
        self.logger = logging.getLogger(__name__)
        
        # Embedding storage
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_dimension = 1536  # OpenAI embedding dimension
        
        # Try to load existing embeddings
        if self.graph_rag_system:
            self.embeddings = self.graph_rag_system.load_embeddings()
        
    def generate_embeddings(self, documents: Dict[str, Document]) -> None:
        """
        Generate embeddings for all documents.
        
        Args:
            documents: Dictionary of document_id -> Document
        """
        if not self.client:
            self.logger.warning("No OpenAI client available for embedding generation")
            return
        
        # Check if we already have embeddings for all documents
        if len(self.embeddings) == len(documents):
            self.logger.info(f"All {len(documents)} documents already have embeddings, skipping generation")
            return
            
        try:
            missing_docs = {doc_id: doc for doc_id, doc in documents.items() if doc_id not in self.embeddings}
            if missing_docs:
                self.logger.info(f"Generating embeddings for {len(missing_docs)} new documents (already have {len(self.embeddings)})")
            else:
                self.logger.info(f"Generating embeddings for {len(documents)} documents")
            
            # Process documents in smaller batches and handle token limits
            batch_size = 20  # Smaller batches to avoid token limits
            
            # Only process documents that don't have embeddings yet
            docs_to_process = missing_docs if missing_docs else documents
            doc_items = list(docs_to_process.items())
            
            for i in range(0, len(doc_items), batch_size):
                batch = doc_items[i:i + batch_size]
                
                # Prepare texts for embedding
                texts = []
                doc_ids = []
                
                for doc_id, document in batch:
                    # Combine title and content for embedding, but truncate if too long
                    text = f"{document.title}\n\n{document.content}"
                    
                    # Truncate text to roughly 6000 tokens (24000 chars) to stay under 8192 token limit
                    if len(text) > 6000:
                        text = text[:6000] + "..."
                    
                    texts.append(text)
                    doc_ids.append(doc_id)
                
                try:
                    # Generate embeddings
                    response = self.client.embeddings.create(
                        model=self.config.embedding_model,
                        input=texts
                    )
                except Exception as e:
                    # If batch still fails, process documents individually
                    self.logger.warning(f"Batch failed, processing individually: {e}")
                    for j, (doc_id, text) in enumerate(zip(doc_ids, texts)):
                        try:
                            # Truncate further if needed
                            if len(text) > 3000:
                                text = text[:3000] + "..."
                            
                            response = self.client.embeddings.create(
                                model=self.config.embedding_model,
                                input=[text]
                            )
                            
                            embedding = np.array(response.data[0].embedding)
                            self.embeddings[doc_id] = embedding
                            
                        except Exception as individual_error:
                            self.logger.error(f"Failed to embed document {doc_id}: {individual_error}")
                            continue
                    continue
                
                # Store embeddings
                for j, embedding_data in enumerate(response.data):
                    doc_id = doc_ids[j]
                    embedding = np.array(embedding_data.embedding)
                    self.embeddings[doc_id] = embedding
                
                self.logger.info(f"Generated embeddings for batch {i//batch_size + 1}/{(len(doc_items)-1)//batch_size + 1}")
            
            self.logger.info("Embedding generation completed")
            
            # Save embeddings to disk
            if self.graph_rag_system:
                self.graph_rag_system.save_embeddings(self.embeddings)
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find documents most similar to the query using semantic search.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (document_id, similarity_score) tuples
        """
        if not self.client or not self.embeddings:
            self.logger.warning("No embeddings available for semantic search")
            return []
        
        try:
            # Generate query embedding
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=[query]
            )
            query_embedding = np.array(response.data[0].embedding)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append((doc_id, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []


# =============================================================================
# PHASE 4: GRAPH RAG RETRIEVAL SYSTEM
# =============================================================================

class GraphRAGRetriever:
    """
    Enhanced Graph RAG retrieval combining multiple search strategies.
    
    This class orchestrates the hybrid retrieval approach:
    1. Semantic search to find initially relevant documents
    2. Graph traversal through wiki-links to expand context
    3. Ranking and filtering of combined results
    
    NEW ENHANCED FEATURES:
    4. Hybrid search (Vector + BM25 + Sparse)
    5. Dynamic retrieval strategy selection
    6. Tensor-based reranking
    7. Community-based global search
    8. Multi-granular indexing
    
    This approach leverages both the semantic meaning of content and the
    explicit relationships encoded in Obsidian's wiki-link structure.
    """
    
    def __init__(self, config: GraphRAGConfig, client=None):
        """
        Initialize the enhanced Graph RAG retriever.
        
        Args:
            config: Graph RAG configuration
            client: OpenAI client for enhanced features
        """
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # Enhanced components (initialized on first use)
        self.hybrid_search_manager = None
        self.clustering_manager = None
        self.community_manager = None
        self.tensor_reranker = None
        self.dynamic_retrieval_manager = None
        self.lazy_graphrag_manager = None
        self.multi_granular_manager = None
        
        # Community summaries cache
        self.community_summaries = {}
        
        # Initialize enhanced components if enabled
        self._initialize_enhanced_components()
    
    def _initialize_enhanced_components(self):
        """Initialize enhanced Graph RAG components if enabled."""
        try:
            # Import enhanced components
            from enhanced_graphrag import (
                HybridSearchManager, AdvancedClusteringManager, 
                CommunitySummarizationManager, TensorRerankingManager,
                DynamicRetrievalManager, LazyGraphRAGManager,
                MultiGranularIndexManager
            )
            
            # Initialize components based on configuration
            if self.config.enable_hybrid_search:
                self.hybrid_search_manager = HybridSearchManager(self.config)
                self.logger.info("Hybrid search manager initialized")
            
            if self.config.use_leiden_clustering:
                self.clustering_manager = AdvancedClusteringManager(self.config)
                self.logger.info("Advanced clustering manager initialized")
            
            if self.config.enable_community_summarization and self.client:
                self.community_manager = CommunitySummarizationManager(self.config, self.client)
                self.logger.info("Community summarization manager initialized")
            
            if self.config.enable_tensor_reranking:
                self.tensor_reranker = TensorRerankingManager(self.config)
                self.logger.info("Tensor reranking manager initialized")
            
            if self.config.enable_dynamic_retrieval and self.client:
                self.dynamic_retrieval_manager = DynamicRetrievalManager(self.config, self.client)
                self.logger.info("Dynamic retrieval manager initialized")
            
            if self.config.enable_lazy_graphrag and self.client:
                self.lazy_graphrag_manager = LazyGraphRAGManager(self.config, self.client)
                self.logger.info("LazyGraphRAG manager initialized")
            
            if self.config.enable_multi_granular:
                self.multi_granular_manager = MultiGranularIndexManager(self.config)
                self.logger.info("Multi-granular index manager initialized")
                
        except ImportError as e:
            self.logger.warning(f"Enhanced components not available: {e}")
        except Exception as e:
            self.logger.error(f"Error initializing enhanced components: {e}")
    
    async def setup_enhanced_features(self, documents: Dict[str, Document], 
                                    knowledge_graph: nx.Graph):
        """Set up enhanced features that require document data."""
        try:
            # Setup hybrid search with documents
            if self.hybrid_search_manager:
                self.hybrid_search_manager.documents = documents
                self.hybrid_search_manager._setup_hybrid_search()
            
            # Generate community summaries if enabled
            if self.community_manager and not self.community_summaries:
                # Get communities from clustering
                if self.clustering_manager:
                    communities = self.clustering_manager.leiden_clustering(knowledge_graph)
                else:
                    # Fall back to basic clustering
                    from community import community_louvain
                    partition = community_louvain.best_partition(knowledge_graph)
                    communities = {}
                    for node, community_id in partition.items():
                        if community_id not in communities:
                            communities[community_id] = []
                        communities[community_id].append(node)
                
                # Generate summaries
                self.community_summaries = await self.community_manager.generate_community_summaries(
                    communities, documents
                )
                self.logger.info(f"Generated {len(self.community_summaries)} community summaries")
            
            # Setup multi-granular indexing
            if self.multi_granular_manager:
                self.multi_granular_manager.full_graph = knowledge_graph
                self.multi_granular_manager.build_skeleton_graph(knowledge_graph, documents)
                self.multi_granular_manager.build_keyword_bipartite(documents)
                
        except Exception as e:
            self.logger.error(f"Error setting up enhanced features: {e}")
        
    async def retrieve_context(self, 
                              query: str,
                              documents: Dict[str, Document],
                              knowledge_graph: nx.Graph,
                              embedding_manager: EmbeddingManager) -> List[Document]:
        """
        Enhanced retrieve relevant documents using multiple Graph RAG strategies.
        
        This method implements the enhanced Graph RAG algorithm:
        1. Dynamic strategy selection (local vs global)
        2. Hybrid search (vector + BM25 + sparse)
        3. Graph traversal through wiki-links
        4. Tensor-based reranking
        5. Community-based global search
        
        Args:
            query: User's query
            documents: All available documents
            knowledge_graph: Graph of document relationships
            embedding_manager: For semantic search
            
        Returns:
            List of relevant documents for context
        """
        try:
            self.logger.debug(f"Enhanced retrieving context for query: {query}")
            
            # Step 1: Dynamic strategy selection
            search_strategy = "LOCAL"  # Default
            if self.dynamic_retrieval_manager:
                search_strategy, confidence, reasoning = await self.dynamic_retrieval_manager.classify_and_route_query(query)
                self.logger.debug(f"Query classified as {search_strategy} (confidence: {confidence:.2f})")
            
            # Step 2: Choose retrieval approach based on strategy
            if search_strategy == "GLOBAL" and self.community_summaries:
                return await self._global_search(query, documents, knowledge_graph, embedding_manager)
            else:
                return await self._local_search(query, documents, knowledge_graph, embedding_manager)
                
        except Exception as e:
            self.logger.error(f"Error in enhanced Graph RAG retrieval: {e}")
            # Fall back to original method
            return self._fallback_retrieve_context(query, documents, knowledge_graph, embedding_manager)
    
    async def _local_search(self, query: str, documents: Dict[str, Document],
                           knowledge_graph: nx.Graph, embedding_manager: EmbeddingManager) -> List[Document]:
        """Enhanced local search for entity-focused queries."""
        
        # Step 1: Get initial semantic results
        semantic_results = embedding_manager.search_similar_documents(
            query, 
            top_k=self.config.top_k_vector
        )
        
        if not semantic_results:
            self.logger.warning("No semantic search results found")
            return []
        
        # Step 2: Apply hybrid search if enabled
        from enhanced_graphrag import SearchResult
        
        if self.hybrid_search_manager:
            search_results = self.hybrid_search_manager.hybrid_search(
                query, semantic_results, top_k=self.config.top_k_vector * 2
            )
        else:
            # Convert to SearchResult format
            search_results = [SearchResult(
                document_id=doc_id,
                content=documents.get(doc_id, {}).get('content', ''),
                vector_score=score,
                combined_score=score
            ) for doc_id, score in semantic_results]
        
        # Step 3: Graph expansion
        expanded_doc_ids = set()
        
        # Add initial results
        for result in search_results:
            expanded_doc_ids.add(result.document_id)
        
        # Graph traversal
        for result in search_results[:self.config.top_k_vector]:  # Only expand top results
            doc_id = result.document_id
            if doc_id in knowledge_graph:
                neighbors = list(knowledge_graph.neighbors(doc_id))
                
                for neighbor in neighbors[:self.config.top_k_graph]:
                    expanded_doc_ids.add(neighbor)
                    
                    if self.config.max_graph_hops >= 2:
                        second_hop = list(knowledge_graph.neighbors(neighbor))
                        for second_neighbor in second_hop[:2]:
                            expanded_doc_ids.add(second_neighbor)
        
        # Step 4: Multi-granular search enhancement
        if self.multi_granular_manager:
            multi_granular_results = self.multi_granular_manager.multi_granular_search(query, "LOCAL")
            expanded_doc_ids.update(multi_granular_results)
        
        # Step 5: Apply tensor reranking if enabled
        if self.tensor_reranker and len(search_results) > 1:
            # Update search results with documents from graph expansion
            all_results = []
            for doc_id in expanded_doc_ids:
                if doc_id in documents:
                    # Find existing result or create new one
                    existing = next((r for r in search_results if r.document_id == doc_id), None)
                    if existing:
                        all_results.append(existing)
                    else:
                        all_results.append(SearchResult(
                            document_id=doc_id,
                            content=documents[doc_id].content,
                            combined_score=0.5  # Lower score for graph-only matches
                        ))
            
            # Apply tensor reranking
            all_results = self.tensor_reranker.rerank_results(query, all_results)
            
            # Update expanded_doc_ids based on reranked results
            expanded_doc_ids = set(result.document_id for result in all_results)
        
        # Step 6: Collect final documents
        context_documents = []
        for doc_id in expanded_doc_ids:
            if doc_id in documents:
                context_documents.append(documents[doc_id])
        
        # Sort by relevance (prioritize semantic matches)
        semantic_doc_ids = set(result.document_id for result in search_results[:self.config.top_k_vector])
        context_documents.sort(key=lambda doc: 0 if doc.id in semantic_doc_ids else 1)
        
        self.logger.debug(f"Local search retrieved {len(context_documents)} documents")
        return context_documents
    
    async def _global_search(self, query: str, documents: Dict[str, Document],
                            knowledge_graph: nx.Graph, embedding_manager: EmbeddingManager) -> List[Document]:
        """Enhanced global search using community summaries."""
        
        try:
            # Step 1: Find relevant communities based on query
            relevant_communities = []
            
            if self.community_summaries:
                for community_id, community_report in self.community_summaries.items():
                    # Simple relevance check (could be enhanced with embeddings)
                    summary_text = f"{community_report.title} {community_report.summary}"
                    query_words = set(query.lower().split())
                    summary_words = set(summary_text.lower().split())
                    
                    # Calculate word overlap (simple relevance metric)
                    overlap = len(query_words.intersection(summary_words))
                    if overlap > 0 or len(query_words) == 1:  # Include if any overlap or single word query
                        relevant_communities.append((community_id, community_report, overlap))
            
            # Sort by relevance
            relevant_communities.sort(key=lambda x: x[2], reverse=True)
            
            # Step 2: LazyGraphRAG on-demand summaries
            if self.lazy_graphrag_manager and relevant_communities:
                community_docs = []
                for community_id, community_report, _ in relevant_communities[:5]:  # Top 5 communities
                    # Get documents from this community
                    for entity in community_report.entities[:10]:  # Limit entities
                        # Find documents mentioning this entity (simplified)
                        for doc_id, doc in documents.items():
                            if entity.lower() in doc.content.lower():
                                community_docs.append(doc.content[:1000])  # Limit length
                                break
                
                # Generate focused summary
                if community_docs:
                    focused_summary = await self.lazy_graphrag_manager.generate_summary_on_demand(
                        community_docs, query
                    )
                    self.logger.debug(f"Generated focused summary: {focused_summary[:100]}...")
            
            # Step 3: Collect documents from relevant communities
            relevant_doc_ids = set()
            
            for community_id, community_report, _ in relevant_communities[:3]:  # Top 3 communities
                # Map entities back to documents (simplified approach)
                for entity in community_report.entities[:5]:  # Top entities
                    for doc_id, doc in documents.items():
                        if entity.lower() in doc.content.lower() or entity.lower() in doc.title.lower():
                            relevant_doc_ids.add(doc_id)
            
            # Step 4: Enhance with semantic search if few results
            if len(relevant_doc_ids) < self.config.top_k_vector:
                semantic_results = embedding_manager.search_similar_documents(
                    query, top_k=self.config.top_k_vector
                )
                for doc_id, _ in semantic_results:
                    relevant_doc_ids.add(doc_id)
            
            # Step 5: Apply multi-granular search for global queries
            if self.multi_granular_manager:
                multi_granular_results = self.multi_granular_manager.multi_granular_search(query, "GLOBAL")
                relevant_doc_ids.update(multi_granular_results)
            
            # Step 6: Collect final documents
            context_documents = []
            for doc_id in relevant_doc_ids:
                if doc_id in documents:
                    context_documents.append(documents[doc_id])
            
            # Limit results for global search
            context_documents = context_documents[:self.config.top_k_vector * 2]
            
            self.logger.debug(f"Global search retrieved {len(context_documents)} documents")
            return context_documents
            
        except Exception as e:
            self.logger.error(f"Error in global search: {e}")
            # Fall back to local search
            return await self._local_search(query, documents, knowledge_graph, embedding_manager)
    
    def _fallback_retrieve_context(self, query: str, documents: Dict[str, Document],
                                  knowledge_graph: nx.Graph, embedding_manager: EmbeddingManager) -> List[Document]:
        """Original retrieval method as fallback."""
        try:
            # Original implementation
            semantic_results = embedding_manager.search_similar_documents(query, top_k=self.config.top_k_vector)
            
            if not semantic_results:
                return []
            
            expanded_doc_ids = set()
            for doc_id, score in semantic_results:
                expanded_doc_ids.add(doc_id)
            
            for doc_id, score in semantic_results:
                if doc_id in knowledge_graph:
                    neighbors = list(knowledge_graph.neighbors(doc_id))
                    for neighbor in neighbors[:self.config.top_k_graph]:
                        expanded_doc_ids.add(neighbor)
                        if self.config.max_graph_hops >= 2:
                            second_hop = list(knowledge_graph.neighbors(neighbor))
                            for second_neighbor in second_hop[:2]:
                                expanded_doc_ids.add(second_neighbor)
            
            context_documents = []
            for doc_id in expanded_doc_ids:
                if doc_id in documents:
                    context_documents.append(documents[doc_id])
            
            semantic_doc_ids = set(doc_id for doc_id, _ in semantic_results)
            context_documents.sort(key=lambda doc: 0 if doc.id in semantic_doc_ids else 1)
            
            return context_documents
            
        except Exception as e:
            self.logger.error(f"Error in fallback retrieval: {e}")
            return []
    
    def format_context_for_llm(self, documents: List[Document], max_tokens: int = 10000) -> str:
        """
        Format retrieved documents into context string for LLM.
        
        Args:
            documents: Retrieved documents
            max_tokens: Maximum tokens for context (rough estimate)
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_tokens = 0
        
        for i, doc in enumerate(documents):
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            doc_text = f"## {doc.title}\n\n{doc.content}\n\n"
            estimated_tokens = len(doc_text) // 4
            
            if current_tokens + estimated_tokens > max_tokens:
                break
                
            context_parts.append(doc_text)
            current_tokens += estimated_tokens
        
        return "".join(context_parts)


# =============================================================================
# PHASE 5: CONVERSATIONAL INTERFACE
# =============================================================================

class ObsidianChatBot:
    """
    Interactive chat interface for conversing with Obsidian notes using Graph RAG.
    
    This class provides a simple command-line interface for users to ask questions
    about their notes. It uses Graph RAG to retrieve relevant context and OpenAI
    to generate natural language responses.
    
    Features:
    - Natural language question answering
    - Graph RAG context retrieval
    - Source citations in responses
    - Simple conversation loop
    """
    
    def __init__(self, graph_rag_system: 'ObsidianGraphRAG'):
        """
        Initialize the chat bot.
        
        Args:
            graph_rag_system: Main Graph RAG system instance
        """
        self.system = graph_rag_system
        self.config = graph_rag_system.config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components with enhanced support
        self.embedding_manager = EmbeddingManager(self.config, self.system.client, self.system)
        self.retriever = GraphRAGRetriever(self.config, self.system.client)  # Pass client for enhanced features
        
        # Conversation state
        self.conversation_history = []
        
        # System prompt for the LLM
        self.system_prompt = """You are an AI assistant that helps users explore and understand their personal knowledge base from Obsidian notes.

Your role:
- Answer questions based ONLY on the provided context from the user's notes
- Provide helpful, accurate responses that synthesize information from multiple notes when relevant
- Include citations to source notes in your responses
- If information isn't in the provided context, say so clearly
- Be conversational but concise

Format your responses with:
- Clear, direct answers
- Citations like "According to your note on [Topic]..."
- Suggestions for follow-up questions when appropriate

Remember: You're helping the user explore their own knowledge - be encouraging and insightful."""
    
    def initialize(self) -> None:
        """Initialize the chat bot by generating embeddings."""
        try:
            self.logger.info("Initializing chat bot...")
            
            if not self.system.documents:
                raise ValueError("No documents loaded. Please run vault scanning first.")
            
            # Generate embeddings for all documents
            self.embedding_manager.generate_embeddings(self.system.documents)
            
            self.logger.info("Chat bot initialization complete")
            
        except Exception as e:
            self.logger.error(f"Error initializing chat bot: {e}")
            raise
    
    async def ask_question(self, question: str) -> str:
        """
        Process a user question and generate a response.
        
        Args:
            question: User's question
            
        Returns:
            Generated response
        """
        try:
            if not self.system.client:
                return "Error: OpenAI client not available. Please set your API key."
            
            # Retrieve relevant context using enhanced Graph RAG
            context_documents = await self.retriever.retrieve_context(
                question,
                self.system.documents,
                self.system.knowledge_graph,
                self.embedding_manager
            )
            
            if not context_documents:
                return "I couldn't find any relevant information in your notes to answer that question."
            
            # Format context for LLM
            context = self.retriever.format_context_for_llm(context_documents)
            
            # Generate response
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context from my notes:\n\n{context}\n\nQuestion: {question}"}
            ]
            
            response = self.system.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.temperature
                # No max_tokens limit - let AI choose response length
            )
            
            answer = response.choices[0].message.content
            
            # Add to conversation history
            self.conversation_history.append({
                "question": question,
                "answer": answer,
                "context_documents": [doc.title for doc in context_documents]
            })
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return f"I encountered an error while processing your question: {e}"
    
    async def start_chat(self) -> None:
        """Start the interactive chat session."""
        try:
            print("\n" + "="*60)
            print("🤖 Obsidian AI Librarian")
            print("="*60)
            print("I can help you explore and understand your notes!")
            print("Ask me questions about your knowledge base.")
            print("Type 'quit', 'exit', or 'bye' to end the conversation.")
            print("="*60 + "\n")
            
            while True:
                try:
                    # Get user input
                    question = input("📝 You: ").strip()
                    
                    # Check for exit commands
                    if question.lower() in ['quit', 'exit', 'bye', 'q']:
                        print("\n👋 Thanks for exploring your knowledge base! Goodbye!")
                        break
                    
                    if not question:
                        continue
                    
                    # Process question and generate response
                    print("\n🤔 Thinking...")
                    response = await self.ask_question(question)
                    
                    print(f"\n🤖 AI Librarian: {response}\n")
                    print("-" * 60 + "\n")
                    
                except KeyboardInterrupt:
                    print("\n\n👋 Chat interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}\n")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in chat session: {e}")
            print(f"❌ Chat session error: {e}")


async def main():
    """
    Main async entry point for the Graph RAG system.
    
    This demonstrates the complete workflow including all phases:
    - Phase 2: Document Processing & Parsing
    - Phase 3: Vector Embeddings
    - Phase 4: Graph RAG Retrieval
    - Phase 5: Conversational Interface
    """
    
    print("🧠 Obsidian Graph RAG AI Librarian")
    print("=" * 60)
    
    try:
        # Configure for the user's specific vault
        config = GraphRAGConfig()
        # Vault path should be set via environment variable OBSIDIAN_VAULT_PATH
        if not config.vault_path:
            print("❌ Please set OBSIDIAN_VAULT_PATH environment variable.")
            print("Example: set OBSIDIAN_VAULT_PATH=C:\\path\\to\\your\\vault")
            print("Or set it programmatically: config.vault_path = 'your/vault/path'")
            return
        
        # Prompt for OpenAI API key if not in environment
        if not config.openai_api_key:
            print("⚠️  OpenAI API key not found in environment.")
            api_key = input("Please enter your OpenAI API key (or press Enter to skip AI features): ").strip()
            if api_key:
                config.openai_api_key = api_key
            else:
                print("⚠️  Continuing without OpenAI API key. AI features will be limited.")
        
        print(f"\n🔧 Initializing system for vault: {config.vault_path}")
        graph_rag = ObsidianGraphRAG(config)
        print("✓ System initialized successfully")
        
        # Initialize system with persistence support
        print("\n📚 Initializing Data (with caching support)")
        print("-" * 40)
        
        graph_rag.initialize_system()
        
        # Display results
        print(f"\n📊 Processing Results:")
        print(f"   Documents processed: {len(graph_rag.documents)}")
        print(f"   Graph nodes: {graph_rag.knowledge_graph.number_of_nodes()}")
        print(f"   Graph edges: {graph_rag.knowledge_graph.number_of_edges()}")
        
        # Show sample of processed documents
        if graph_rag.documents:
            print(f"\n📝 Sample processed documents:")
            for i, (doc_id, doc) in enumerate(list(graph_rag.documents.items())[:5]):
                print(f"   {i+1}. {doc.title} ({doc.word_count} words, {len(doc.wikilinks)} links)")
        
        print("\n✓ Phase 2 completed successfully!")
        
        # If OpenAI is available, continue with AI features
        if config.openai_api_key:
            print("\n🤖 Starting Enhanced AI Librarian Chat Interface...")
            print("This may take a moment to generate embeddings and setup enhanced features...")
            await graph_rag.start_chat_session()
        else:
            print("\n⚠️  Skipping AI features due to missing OpenAI API key.")
            print("To use the AI librarian, please set your OPENAI_API_KEY environment variable and restart.")
            print("\n💡 For 3D visualization features, run: python graph3d_launcher.py")
        
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    """
    Entry point that runs the async main function.
    """
    asyncio.run(main())
