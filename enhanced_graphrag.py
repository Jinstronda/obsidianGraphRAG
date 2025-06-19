#!/usr/bin/env python3
"""
Enhanced Graph RAG Components
=============================

This module contains advanced Graph RAG implementations based on the latest research:
- Hybrid Search (Vector + BM25 + Sparse)
- Dynamic Retrieval Strategy Selection
- Hierarchical Community Summarization
- Advanced Leiden Clustering
- Tensor-based Reranking (ColBERT-style)
- LazyGraphRAG Implementation
- Multi-granular Indexing

These components are designed to be drop-in replacements or enhancements to the 
existing GraphRAG system while maintaining backward compatibility.

Author: Assistant
License: MIT
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import networkx as nx
from abc import ABC, abstractmethod

# New imports for enhanced features
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not available. Hybrid search will be disabled.")

try:
    import leidenalg
    import igraph as ig
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning("leidenalg not available. Advanced clustering will be disabled.")

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    TENSOR_RERANK_AVAILABLE = True
except ImportError:
    TENSOR_RERANK_AVAILABLE = False
    logging.warning("torch/transformers not available. Tensor reranking will be disabled.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some features will be disabled.")

try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load the model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        logging.warning("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        nlp = None
except ImportError:
    SPACY_AVAILABLE = False
    nlp = None
    logging.warning("spaCy not available. Entity extraction will be limited.")


# =============================================================================
# ENHANCED DATA STRUCTURES
# =============================================================================

@dataclass
class CommunityReport:
    """Represents a comprehensive report for a community of entities."""
    community_id: str
    title: str
    summary: str
    entities: List[str]
    key_themes: List[str]
    relationships: List[str]
    level: int = 0  # Hierarchical level
    size: int = 0
    density: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class SearchResult:
    """Enhanced search result with multiple scoring mechanisms."""
    document_id: str
    content: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    sparse_score: float = 0.0
    tensor_score: float = 0.0
    graph_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# HYBRID SEARCH MANAGER
# =============================================================================

class HybridSearchManager:
    """
    Implements hybrid search combining vector search, BM25, and sparse vectors.
    
    Based on Microsoft's research showing that combining multiple search 
    methods significantly improves retrieval quality.
    """
    
    def __init__(self, config, documents: Dict[str, Any] = None):
        """
        Initialize the hybrid search manager.
        
        Args:
            config: GraphRAG configuration
            documents: Document collection for indexing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Search components
        self.bm25 = None
        self.sparse_vectorizer = None
        self.sparse_matrix = None
        self.document_ids = []
        self.documents = documents or {}
        
        # Initialize components if available
        if BM25_AVAILABLE and SKLEARN_AVAILABLE:
            self._setup_hybrid_search()
        else:
            self.logger.warning("Hybrid search components not available. Falling back to vector-only search.")
    
    def _setup_hybrid_search(self):
        """Set up BM25 and sparse vector components."""
        if not self.documents:
            self.logger.warning("No documents provided for hybrid search setup.")
            return
            
        try:
            # Prepare documents for BM25
            doc_texts = []
            self.document_ids = []
            
            for doc_id, doc in self.documents.items():
                content = doc.content if hasattr(doc, 'content') else str(doc)
                doc_texts.append(content.lower().split())
                self.document_ids.append(doc_id)
            
            # Initialize BM25
            if doc_texts:
                self.bm25 = BM25Okapi(doc_texts)
                self.logger.info(f"BM25 initialized with {len(doc_texts)} documents")
            
            # Initialize sparse vectorizer
            full_texts = [doc.content if hasattr(doc, 'content') else str(doc) 
                         for doc in self.documents.values()]
            
            self.sparse_vectorizer = TfidfVectorizer(
                max_features=self.config.keyword_bipartite_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            if full_texts:
                self.sparse_matrix = self.sparse_vectorizer.fit_transform(full_texts)
                self.logger.info(f"Sparse vectorizer initialized with {self.sparse_matrix.shape[1]} features")
                
        except Exception as e:
            self.logger.error(f"Error setting up hybrid search: {e}")
            self.bm25 = None
            self.sparse_vectorizer = None
    
    def hybrid_search(self, query: str, vector_scores: List[Tuple[str, float]], 
                     top_k: int = 20) -> List[SearchResult]:
        """
        Perform hybrid search combining vector, BM25, and sparse vector scores.
        
        Args:
            query: Search query
            vector_scores: Results from vector search as [(doc_id, score), ...]
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects with combined scores
        """
        if not self.config.enable_hybrid_search:
            # Fall back to vector-only results
            return [SearchResult(
                document_id=doc_id,
                content=self.documents.get(doc_id, {}).get('content', ''),
                vector_score=score,
                combined_score=score
            ) for doc_id, score in vector_scores[:top_k]]
        
        try:
            results = {}
            
            # Start with vector scores
            for doc_id, score in vector_scores:
                results[doc_id] = SearchResult(
                    document_id=doc_id,
                    content=self.documents.get(doc_id, {}).get('content', ''),
                    vector_score=score
                )
            
            # Add BM25 scores
            if self.bm25 and self.document_ids:
                query_tokens = query.lower().split()
                bm25_scores = self.bm25.get_scores(query_tokens)
                
                for i, doc_id in enumerate(self.document_ids):
                    if doc_id in results:
                        results[doc_id].bm25_score = bm25_scores[i]
                    elif i < len(bm25_scores) and bm25_scores[i] > 0:
                        # Add documents that scored well on BM25 but not vector search
                        results[doc_id] = SearchResult(
                            document_id=doc_id,
                            content=self.documents.get(doc_id, {}).get('content', ''),
                            bm25_score=bm25_scores[i]
                        )
            
            # Add sparse vector scores
            if self.sparse_vectorizer and self.sparse_matrix is not None:
                query_sparse = self.sparse_vectorizer.transform([query])
                sparse_scores = (self.sparse_matrix * query_sparse.T).toarray().flatten()
                
                for i, doc_id in enumerate(self.document_ids):
                    if doc_id in results and i < len(sparse_scores):
                        results[doc_id].sparse_score = sparse_scores[i]
            
            # Combine scores
            for result in results.values():
                vector_weight = 1 - self.config.bm25_weight - self.config.sparse_vector_weight
                result.combined_score = (
                    vector_weight * result.vector_score +
                    self.config.bm25_weight * result.bm25_score +
                    self.config.sparse_vector_weight * result.sparse_score
                )
            
            # Sort by combined score and return top_k
            sorted_results = sorted(results.values(), 
                                  key=lambda x: x.combined_score, 
                                  reverse=True)
            
            return sorted_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            # Fall back to vector-only results
            return [SearchResult(
                document_id=doc_id,
                content=self.documents.get(doc_id, {}).get('content', ''),
                vector_score=score,
                combined_score=score
            ) for doc_id, score in vector_scores[:top_k]]


# =============================================================================
# ADVANCED CLUSTERING MANAGER
# =============================================================================

class AdvancedClusteringManager:
    """
    Implements advanced clustering algorithms including Leiden algorithm.
    
    The Leiden algorithm provides better community detection than the basic
    Louvain algorithm, leading to more coherent communities.
    """
    
    def __init__(self, config):
        """
        Initialize the advanced clustering manager.
        
        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def leiden_clustering(self, networkx_graph: nx.Graph) -> Dict[int, List[str]]:
        """
        Apply Leiden algorithm for community detection.
        
        Args:
            networkx_graph: NetworkX graph to cluster
            
        Returns:
            Dictionary mapping community_id -> [node_ids]
        """
        if not self.config.use_leiden_clustering or not LEIDEN_AVAILABLE:
            # Fall back to Louvain clustering
            return self._louvain_clustering(networkx_graph)
        
        try:
            # Convert NetworkX to igraph
            edge_list = list(networkx_graph.edges(data=True))
            node_list = list(networkx_graph.nodes())
            
            if not edge_list:
                # Handle empty graph
                return {i: [node] for i, node in enumerate(node_list)}
            
            g = ig.Graph()
            g.add_vertices(node_list)
            
            # Add edges with weights if available
            edges = [(u, v) for u, v, _ in edge_list]
            weights = [data.get('weight', 1.0) for _, _, data in edge_list]
            
            g.add_edges(edges)
            
            # Apply Leiden algorithm
            partition = leidenalg.find_partition(
                g, 
                leidenalg.ModularityVertexPartition,
                resolution_parameter=self.config.leiden_resolution,
                n_iterations=self.config.leiden_iterations
            )
            
            # Convert back to communities dict
            communities = {}
            for i, community in enumerate(partition):
                communities[i] = [node_list[node] for node in community]
            
            self.logger.info(f"Leiden clustering found {len(communities)} communities")
            return communities
            
        except Exception as e:
            self.logger.error(f"Error in Leiden clustering: {e}")
            # Fall back to Louvain
            return self._louvain_clustering(networkx_graph)
    
    def _louvain_clustering(self, networkx_graph: nx.Graph) -> Dict[int, List[str]]:
        """Fall back to Louvain clustering."""
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(networkx_graph, 
                                                       resolution=self.config.community_resolution)
            
            # Convert to our format
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
            
            self.logger.info(f"Louvain clustering found {len(communities)} communities")
            return communities
            
        except Exception as e:
            self.logger.error(f"Error in Louvain clustering: {e}")
            # Return single community with all nodes
            return {0: list(networkx_graph.nodes())}


# =============================================================================
# COMMUNITY SUMMARIZATION MANAGER
# =============================================================================

class CommunitySummarizationManager:
    """
    Generates comprehensive summaries for communities using LLMs.
    
    Based on Microsoft GraphRAG's community summarization approach,
    this creates hierarchical summaries that enable global search capabilities.
    """
    
    def __init__(self, config, client):
        """
        Initialize the community summarization manager.
        
        Args:
            config: GraphRAG configuration
            client: OpenAI client for LLM operations
        """
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # Prompt templates
        self.community_summary_prompt = """
        Analyze this community of related documents and entities to provide a comprehensive summary.
        
        Community Documents:
        {documents}
        
        Community Entities:
        {entities}
        
        Please provide:
        1. A descriptive title for this community (2-5 words)
        2. Main themes and topics (3-5 key themes)
        3. Key entities and their relationships
        4. A comprehensive summary paragraph (100-200 words)
        5. Important insights or patterns
        
        Format as JSON:
        {{
            "title": "Community Title",
            "themes": ["theme1", "theme2", "theme3"],
            "key_entities": ["entity1", "entity2"],
            "summary": "Comprehensive summary paragraph...",
            "insights": ["insight1", "insight2"]
        }}
        """
    
    async def generate_community_summaries(self, communities: Dict[int, List[str]], 
                                         documents: Dict[str, Any]) -> Dict[int, CommunityReport]:
        """
        Generate summaries for all communities.
        
        Args:
            communities: Dictionary mapping community_id -> [document_ids]
            documents: Document collection
            
        Returns:
            Dictionary mapping community_id -> CommunityReport
        """
        if not self.config.enable_community_summarization:
            return {}
        
        community_reports = {}
        
        for community_id, doc_ids in communities.items():
            try:
                # Skip very small communities
                if len(doc_ids) < 2:
                    continue
                
                # Skip very large communities (too expensive)
                if len(doc_ids) > self.config.max_community_size:
                    # For large communities, sample representative documents
                    import random
                    doc_ids = random.sample(doc_ids, self.config.max_community_size)
                
                report = await self._generate_single_community_summary(
                    community_id, doc_ids, documents
                )
                
                if report:
                    community_reports[community_id] = report
                
            except Exception as e:
                self.logger.error(f"Error generating summary for community {community_id}: {e}")
                continue
        
        self.logger.info(f"Generated {len(community_reports)} community summaries")
        return community_reports
    
    async def _generate_single_community_summary(self, community_id: int, 
                                               doc_ids: List[str], 
                                               documents: Dict[str, Any]) -> Optional[CommunityReport]:
        """Generate summary for a single community."""
        try:
            # Collect document content
            community_docs = []
            for doc_id in doc_ids:
                if doc_id in documents:
                    doc = documents[doc_id]
                    content = doc.content if hasattr(doc, 'content') else str(doc)
                    title = doc.title if hasattr(doc, 'title') else doc_id
                    community_docs.append(f"Title: {title}\nContent: {content[:500]}...")
            
            if not community_docs:
                return None
            
            # Extract entities (simplified)
            entities = self._extract_entities_from_docs(community_docs)
            
            # Generate summary
            documents_text = "\n\n".join(community_docs[:10])  # Limit for token count
            entities_text = ", ".join(entities[:20])  # Limit entities
            
            prompt = self.community_summary_prompt.format(
                documents=documents_text,
                entities=entities_text
            )
            
            response = self.client.chat.completions.create(
                model=self.config.community_summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse response
            import json
            try:
                summary_data = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fall back to text parsing
                content = response.choices[0].message.content
                summary_data = {
                    "title": f"Community {community_id}",
                    "themes": ["Mixed Topics"],
                    "key_entities": entities[:5],
                    "summary": content[:500],
                    "insights": []
                }
            
            # Create community report
            report = CommunityReport(
                community_id=str(community_id),
                title=summary_data.get("title", f"Community {community_id}"),
                summary=summary_data.get("summary", ""),
                entities=summary_data.get("key_entities", []),
                key_themes=summary_data.get("themes", []),
                relationships=[],  # Could be enhanced
                level=0,
                size=len(doc_ids),
                density=0.0,  # Could be calculated
                metadata={"insights": summary_data.get("insights", [])}
            )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error in single community summary: {e}")
            return None
    
    def _extract_entities_from_docs(self, docs: List[str]) -> List[str]:
        """Extract entities from documents using spaCy if available."""
        entities = set()
        
        if SPACY_AVAILABLE and nlp:
            try:
                for doc in docs:
                    # Process text
                    processed = nlp(doc[:1000])  # Limit for performance
                    
                    # Extract named entities
                    for ent in processed.ents:
                        if len(ent.text) > 2 and ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                            entities.add(ent.text)
                
            except Exception as e:
                self.logger.warning(f"Error in entity extraction: {e}")
        
        # Fall back to simple keyword extraction
        if not entities:
            # Simple keyword extraction
            import re
            for doc in docs:
                # Extract capitalized words (likely entities)
                words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc)
                entities.update(words[:10])  # Limit to avoid noise
        
        return list(entities)


# =============================================================================
# TENSOR RERANKING MANAGER
# =============================================================================

class TensorRerankingManager:
    """
    Implements tensor-based reranking using ColBERT-style late interaction.
    
    This provides much better ranking quality than simple cosine similarity
    by considering token-level interactions between query and documents.
    """
    
    def __init__(self, config):
        """
        Initialize the tensor reranking manager.
        
        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        
        # Initialize if available
        if TENSOR_RERANK_AVAILABLE and self.config.enable_tensor_reranking:
            self._setup_reranking_model()
        else:
            self.logger.warning("Tensor reranking not available. Using simple ranking.")
    
    def _setup_reranking_model(self):
        """Set up the tensor reranking model."""
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.reranking_model)
            self.model = AutoModel.from_pretrained(self.config.reranking_model)
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Tensor reranking model loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error setting up reranking model: {e}")
            self.tokenizer = None
            self.model = None
    
    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Rerank search results using tensor-based late interaction.
        
        Args:
            query: Search query
            results: Initial search results
            
        Returns:
            Reranked search results
        """
        if not self.config.enable_tensor_reranking or not self.model:
            return results
        
        try:
            # Only rerank top candidates to save computation
            candidates = results[:self.config.tensor_rerank_top_k]
            
            if len(candidates) <= 1:
                return results
            
            # Compute tensor scores
            for result in candidates:
                tensor_score = self._compute_late_interaction_score(query, result.content)
                result.tensor_score = tensor_score
                
                # Update combined score to include tensor score
                result.combined_score = (
                    0.7 * result.combined_score + 0.3 * tensor_score
                )
            
            # Re-sort by updated combined score
            candidates.sort(key=lambda x: x.combined_score, reverse=True)
            
            # Return reranked candidates + remaining results
            return candidates + results[self.config.tensor_rerank_top_k:]
            
        except Exception as e:
            self.logger.error(f"Error in tensor reranking: {e}")
            return results
    
    def _compute_late_interaction_score(self, query: str, document: str) -> float:
        """Compute late interaction score between query and document."""
        try:
            with torch.no_grad():
                # Tokenize inputs
                query_tokens = self.tokenizer(
                    query, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                doc_tokens = self.tokenizer(
                    document[:2000],  # Limit document length
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Get embeddings
                query_embeddings = self.model(**query_tokens).last_hidden_state
                doc_embeddings = self.model(**doc_tokens).last_hidden_state
                
                # Compute token-level similarities (late interaction)
                similarity_matrix = torch.matmul(
                    query_embeddings, 
                    doc_embeddings.transpose(-2, -1)
                )
                
                # Max pooling over document tokens for each query token
                max_similarities = torch.max(similarity_matrix, dim=-1)[0]
                
                # Sum over query tokens (mean would also work)
                final_score = torch.sum(max_similarities).item()
                
                # Normalize by query length
                final_score = final_score / query_embeddings.size(1)
                
                return final_score
                
        except Exception as e:
            self.logger.error(f"Error computing late interaction score: {e}")
            return 0.0


# =============================================================================
# DYNAMIC RETRIEVAL STRATEGY MANAGER
# =============================================================================

class DynamicRetrievalManager:
    """
    Implements dynamic retrieval strategy selection based on query type.
    
    This automatically chooses between local (entity-focused) and global 
    (theme-focused) search strategies based on the query characteristics.
    """
    
    def __init__(self, config, client):
        """
        Initialize the dynamic retrieval manager.
        
        Args:
            config: GraphRAG configuration
            client: OpenAI client for query classification
        """
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # Query classification prompt
        self.classification_prompt = """
        Classify this query as either LOCAL or GLOBAL:
        
        LOCAL queries are:
        - About specific entities, people, concepts, or documents
        - Asking for details about particular items
        - Focused on narrow, specific information
        
        GLOBAL queries are:
        - About themes, patterns, or overarching topics
        - Asking for summaries or broad understanding
        - Requiring synthesis across multiple sources
        
        Examples:
        LOCAL: "What did I write about Python?", "Tell me about John Smith", "Show me my transformer notes"
        GLOBAL: "What are the main themes in my research?", "How do my interests connect?", "Summarize the key patterns"
        
        Query: {query}
        
        Classification: LOCAL or GLOBAL
        Confidence: [1-10]
        Reasoning: [brief explanation]
        
        Answer in format: "CLASSIFICATION|CONFIDENCE|REASONING"
        """
    
    async def classify_and_route_query(self, query: str) -> Tuple[str, float, str]:
        """
        Classify query type and return routing decision.
        
        Args:
            query: User query to classify
            
        Returns:
            Tuple of (strategy, confidence, reasoning)
        """
        if not self.config.enable_dynamic_retrieval:
            return "LOCAL", 1.0, "Dynamic retrieval disabled"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for classification
                messages=[{
                    "role": "user",
                    "content": self.classification_prompt.format(query=query)
                }],
                temperature=0.0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse response
            parts = content.split("|")
            if len(parts) >= 3:
                strategy = parts[0].strip()
                try:
                    confidence = float(parts[1].strip()) / 10.0  # Normalize to 0-1
                except ValueError:
                    confidence = 0.5
                reasoning = parts[2].strip()
                
                # Validate strategy
                if strategy not in ["LOCAL", "GLOBAL"]:
                    strategy = "LOCAL"  # Default fallback
                    
                return strategy, confidence, reasoning
            else:
                # Fallback parsing
                if "GLOBAL" in content.upper():
                    return "GLOBAL", 0.5, "Fallback classification"
                else:
                    return "LOCAL", 0.5, "Fallback classification"
                    
        except Exception as e:
            self.logger.error(f"Error in query classification: {e}")
            return "LOCAL", 0.5, f"Classification error: {e}"


# =============================================================================
# LAZY GRAPHRAG MANAGER
# =============================================================================

class LazyGraphRAGManager:
    """
    Implements LazyGraphRAG approach for cost-efficient processing.
    
    Instead of pre-generating all summaries, this defers LLM usage until 
    query time, achieving massive cost savings while maintaining quality.
    """
    
    def __init__(self, config, client):
        """
        Initialize the LazyGraphRAG manager.
        
        Args:
            config: GraphRAG configuration  
            client: OpenAI client for on-demand operations
        """
        self.config = config
        self.client = client
        self.logger = logging.getLogger(__name__)
        
        # Caches for on-demand generation
        self.entity_cache = {}
        self.summary_cache = {}
        self.relationship_cache = {}
    
    def extract_entities_lightweight(self, documents: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract entities using lightweight methods (no LLM).
        
        Args:
            documents: Document collection
            
        Returns:
            Dictionary mapping doc_id -> [entities]
        """
        if not self.config.enable_lazy_graphrag:
            return {}
        
        entities = {}
        
        for doc_id, doc in documents.items():
            doc_entities = []
            content = doc.content if hasattr(doc, 'content') else str(doc)
            
            if SPACY_AVAILABLE and nlp:
                try:
                    # Use spaCy for entity extraction
                    processed = nlp(content[:2000])  # Limit for performance
                    
                    for ent in processed.ents:
                        if (len(ent.text) > 2 and 
                            ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']):
                            doc_entities.append(ent.text)
                    
                except Exception as e:
                    self.logger.warning(f"spaCy entity extraction failed: {e}")
            
            # Fall back to simple keyword extraction
            if not doc_entities:
                import re
                # Extract capitalized words and phrases
                words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
                doc_entities = list(set(words))[:20]  # Limit to avoid noise
            
            entities[doc_id] = doc_entities
        
        return entities
    
    async def generate_summary_on_demand(self, community_docs: List[str], 
                                       query: str) -> str:
        """
        Generate community summary only when needed, focused on the query.
        
        Args:
            community_docs: Documents in the community
            query: Current user query for focused summarization
            
        Returns:
            Query-focused community summary
        """
        if not self.config.enable_lazy_graphrag:
            return ""
        
        # Create cache key
        docs_hash = hash(tuple(sorted(community_docs)))
        query_hash = hash(query)
        cache_key = f"{docs_hash}_{query_hash}"
        
        # Check cache first
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]
        
        try:
            # Generate query-focused summary
            docs_text = "\n\n".join(community_docs[:5])  # Limit for tokens
            
            prompt = f"""
            Based on the following documents, provide a focused summary that helps answer this query: "{query}"
            
            Documents:
            {docs_text}
            
            Provide a concise summary (100-150 words) that highlights information most relevant to the query.
            Focus on facts, entities, and relationships that would help answer the question.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300
            )
            
            summary = response.choices[0].message.content
            
            # Cache the result
            self.summary_cache[cache_key] = summary
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating on-demand summary: {e}")
            return "Summary generation failed."


# =============================================================================
# MULTI-GRANULAR INDEX MANAGER
# =============================================================================

class MultiGranularIndexManager:
    """
    Implements multi-granular indexing for different query types.
    
    Maintains both a skeleton graph of key documents and a lightweight
    keyword-bipartite graph for efficient retrieval at different granularities.
    """
    
    def __init__(self, config):
        """
        Initialize the multi-granular index manager.
        
        Args:
            config: GraphRAG configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Index structures
        self.skeleton_graph = None
        self.full_graph = None
        self.keyword_bipartite = {}
        self.centrality_scores = {}
    
    def build_skeleton_graph(self, full_graph: nx.Graph, 
                           documents: Dict[str, Any]) -> nx.Graph:
        """
        Build a skeleton graph containing only the most important documents.
        
        Args:
            full_graph: Complete knowledge graph
            documents: Document collection
            
        Returns:
            Skeleton graph with key documents
        """
        if not self.config.enable_multi_granular:
            return full_graph
        
        try:
            # Calculate centrality scores
            self.centrality_scores = nx.degree_centrality(full_graph)
            
            # Add other centrality measures
            try:
                pagerank_scores = nx.pagerank(full_graph)
                betweenness_scores = nx.betweenness_centrality(full_graph)
                
                # Combine centrality scores
                for node in full_graph.nodes():
                    combined_score = (
                        0.4 * self.centrality_scores.get(node, 0) +
                        0.4 * pagerank_scores.get(node, 0) +
                        0.2 * betweenness_scores.get(node, 0)
                    )
                    self.centrality_scores[node] = combined_score
                    
            except Exception as e:
                self.logger.warning(f"Error computing advanced centrality: {e}")
            
            # Select top nodes for skeleton
            sorted_nodes = sorted(
                self.centrality_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            skeleton_size = max(1, int(len(sorted_nodes) * self.config.skeleton_graph_ratio))
            key_nodes = [node for node, _ in sorted_nodes[:skeleton_size]]
            
            # Create skeleton subgraph
            self.skeleton_graph = full_graph.subgraph(key_nodes).copy()
            
            self.logger.info(f"Built skeleton graph with {len(key_nodes)} key documents")
            return self.skeleton_graph
            
        except Exception as e:
            self.logger.error(f"Error building skeleton graph: {e}")
            return full_graph
    
    def build_keyword_bipartite(self, documents: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Build keyword-bipartite graph as lightweight alternative.
        
        Args:
            documents: Document collection
            
        Returns:
            Dictionary mapping doc_id -> [keywords]
        """
        if not self.config.enable_multi_granular or not SKLEARN_AVAILABLE:
            return {}
        
        try:
            # Prepare documents for TF-IDF
            doc_texts = []
            doc_ids = []
            
            for doc_id, doc in documents.items():
                content = doc.content if hasattr(doc, 'content') else str(doc)
                doc_texts.append(content)
                doc_ids.append(doc_id)
            
            if not doc_texts:
                return {}
            
            # Generate TF-IDF features
            vectorizer = TfidfVectorizer(
                max_features=self.config.keyword_bipartite_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,  # Require at least 2 occurrences
                max_df=0.95  # Ignore very common terms
            )
            
            tfidf_matrix = vectorizer.fit_transform(doc_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Build bipartite connections
            keyword_bipartite = {}
            for i, doc_id in enumerate(doc_ids):
                # Get non-zero features for this document
                doc_features = tfidf_matrix[i].nonzero()[1]
                
                # Get top keywords by TF-IDF score
                feature_scores = [(feature_names[j], tfidf_matrix[i, j]) 
                                for j in doc_features]
                feature_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Store top keywords
                top_keywords = [keyword for keyword, _ in feature_scores[:50]]
                keyword_bipartite[doc_id] = top_keywords
            
            self.keyword_bipartite = keyword_bipartite
            self.logger.info(f"Built keyword bipartite with {len(feature_names)} features")
            
            return keyword_bipartite
            
        except Exception as e:
            self.logger.error(f"Error building keyword bipartite: {e}")
            return {}
    
    def multi_granular_search(self, query: str, strategy: str = "LOCAL") -> List[str]:
        """
        Perform search across multiple granularities.
        
        Args:
            query: Search query
            strategy: Search strategy ("LOCAL" or "GLOBAL")
            
        Returns:
            List of relevant document IDs
        """
        if not self.config.enable_multi_granular:
            return []
        
        relevant_docs = set()
        
        try:
            if strategy == "GLOBAL" and self.skeleton_graph:
                # Use skeleton graph for global search
                # This would integrate with community summaries
                relevant_docs.update(self.skeleton_graph.nodes())
                
            elif strategy == "LOCAL" and self.keyword_bipartite:
                # Use keyword bipartite for local search
                query_keywords = query.lower().split()
                
                for doc_id, keywords in self.keyword_bipartite.items():
                    # Simple keyword matching (could be improved with TF-IDF)
                    keyword_matches = sum(1 for kw in query_keywords 
                                        if any(kw in keyword.lower() for keyword in keywords))
                    
                    if keyword_matches > 0:
                        relevant_docs.add(doc_id)
            
            return list(relevant_docs)
            
        except Exception as e:
            self.logger.error(f"Error in multi-granular search: {e}")
            return [] 