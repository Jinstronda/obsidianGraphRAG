#!/usr/bin/env python3
"""
SOTA Phase 1: Advanced Entity-Based Linking & Disambiguation
============================================================

Implements state-of-the-art entity processing capabilities:
- Advanced NER with contextual disambiguation
- Cross-document entity linking
- Knowledge base integration
- Entity profile management

Based on latest research in entity linking and knowledge graphs.
"""

import os
import re
import json
import logging
import asyncio
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime

# NLP Libraries
try:
    import spacy
    from spacy import displacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForTokenClassification,
        pipeline,
        AutoModel
    )
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Similarity and clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

@dataclass
class EntityMention:
    """Represents a mention of an entity in text"""
    text: str                    # The actual text of the mention
    start_char: int             # Starting character position
    end_char: int               # Ending character position
    document_id: str            # Document containing this mention
    context: str                # Surrounding context
    confidence: float = 1.0     # Confidence score
    entity_type: str = ""       # Predicted entity type
    embeddings: Optional[np.ndarray] = None

@dataclass
class EntityProfile:
    """Complete profile of a disambiguated entity"""
    entity_id: str              # Unique entity identifier
    canonical_name: str         # Primary name for this entity
    entity_type: str           # Type (PERSON, ORG, CONCEPT, etc.)
    aliases: Set[str] = field(default_factory=set)  # Alternative names
    mentions: List[EntityMention] = field(default_factory=list)
    documents: Set[str] = field(default_factory=set)  # Documents mentioning this entity
    description: str = ""       # Generated entity description
    properties: Dict[str, Any] = field(default_factory=dict)
    centrality_score: float = 0.0
    embeddings: Optional[np.ndarray] = None
    created_at: str = ""
    updated_at: str = ""

@dataclass
class EntityRelationship:
    """Relationship between two entities"""
    source_entity: str          # Source entity ID
    target_entity: str          # Target entity ID
    relation_type: str          # Type of relationship
    confidence: float = 1.0     # Confidence in this relationship
    evidence: List[str] = field(default_factory=list)  # Supporting mentions
    properties: Dict[str, Any] = field(default_factory=dict)

class AdvancedNERManager:
    """Advanced Named Entity Recognition with multiple models"""
    
    def __init__(self, config):
        self.config = config
        self.confidence_threshold = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.8"))
        
        # Initialize NER models
        self.spacy_model = None
        self.transformer_model = None
        self.tokenizer = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available NER models"""
        if HAS_SPACY:
            try:
                # Try to load English model
                self.spacy_model = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy English model loaded successfully")
            except OSError:
                logger.warning("⚠️ spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        
        if HAS_TRANSFORMERS:
            try:
                # Use a more advanced transformer model for NER
                model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.transformer_model = AutoModelForTokenClassification.from_pretrained(model_name)
                self.ner_pipeline = pipeline(
                    "ner", 
                    model=self.transformer_model, 
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple"
                )
                logger.info("✅ Transformer NER model loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️ Could not load transformer NER model: {e}")
    
    def extract_entities(self, text: str, document_id: str) -> List[EntityMention]:
        """Extract entities from text using multiple models"""
        mentions = []
        
        # Extract using spaCy
        if self.spacy_model:
            spacy_mentions = self._extract_with_spacy(text, document_id)
            mentions.extend(spacy_mentions)
        
        # Extract using transformers
        if self.transformer_model:
            transformer_mentions = self._extract_with_transformers(text, document_id)
            mentions.extend(transformer_mentions)
        
        # Deduplicate and merge overlapping mentions
        mentions = self._merge_overlapping_mentions(mentions)
        
        # Filter by confidence
        mentions = [m for m in mentions if m.confidence >= self.confidence_threshold]
        
        return mentions
    
    def _extract_with_spacy(self, text: str, document_id: str) -> List[EntityMention]:
        """Extract entities using spaCy"""
        mentions = []
        doc = self.spacy_model(text)
        
        for ent in doc.ents:
            # Get surrounding context
            context_start = max(0, ent.start_char - 100)
            context_end = min(len(text), ent.end_char + 100)
            context = text[context_start:context_end]
            
            mention = EntityMention(
                text=ent.text,
                start_char=ent.start_char,
                end_char=ent.end_char,
                document_id=document_id,
                context=context,
                confidence=0.8,  # spaCy doesn't provide confidence, use default
                entity_type=ent.label_
            )
            mentions.append(mention)
        
        return mentions
    
    def _extract_with_transformers(self, text: str, document_id: str) -> List[EntityMention]:
        """Extract entities using transformer model"""
        mentions = []
        
        try:
            # Split text into chunks if too long
            max_length = 500
            chunks = [text[i:i+max_length] for i in range(0, len(text), max_length-50)]
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_offset = chunk_idx * (max_length - 50)
                entities = self.ner_pipeline(chunk)
                
                for entity in entities:
                    # Calculate absolute positions
                    start_char = chunk_offset + entity['start']
                    end_char = chunk_offset + entity['end']
                    
                    # Get context
                    context_start = max(0, start_char - 100)
                    context_end = min(len(text), end_char + 100)
                    context = text[context_start:context_end]
                    
                    mention = EntityMention(
                        text=entity['word'],
                        start_char=start_char,
                        end_char=end_char,
                        document_id=document_id,
                        context=context,
                        confidence=entity['score'],
                        entity_type=entity['entity_group']
                    )
                    mentions.append(mention)
        
        except Exception as e:
            logger.warning(f"Error in transformer NER: {e}")
        
        return mentions
    
    def _merge_overlapping_mentions(self, mentions: List[EntityMention]) -> List[EntityMention]:
        """Merge overlapping entity mentions"""
        if not mentions:
            return mentions
        
        # Sort by start position
        mentions.sort(key=lambda x: x.start_char)
        
        merged = [mentions[0]]
        
        for current in mentions[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current.start_char <= last.end_char:
                # Overlapping - keep the one with higher confidence
                if current.confidence > last.confidence:
                    merged[-1] = current
            else:
                # No overlap - add to list
                merged.append(current)
        
        return merged

class EntityDisambiguationManager:
    """Disambiguates entity mentions using contextual embeddings"""
    
    def __init__(self, config, embedding_model=None):
        self.config = config
        self.embedding_model = embedding_model
        self.entity_profiles: Dict[str, EntityProfile] = {}
        self.mention_embeddings_cache = {}
        
        # Similarity thresholds
        self.similarity_threshold = 0.85  # High threshold for same entity
        self.disambiguation_threshold = 0.7  # Lower threshold for potential matches
    
    async def disambiguate_mentions(self, mentions: List[EntityMention]) -> Dict[str, EntityProfile]:
        """Disambiguate entity mentions and create/update entity profiles"""
        
        # Generate embeddings for mentions
        await self._generate_mention_embeddings(mentions)
        
        # Group similar mentions
        entity_clusters = self._cluster_mentions(mentions)
        
        # Create or update entity profiles
        updated_profiles = {}
        for cluster_id, cluster_mentions in entity_clusters.items():
            entity_profile = await self._create_or_update_entity_profile(cluster_mentions)
            updated_profiles[entity_profile.entity_id] = entity_profile
            self.entity_profiles[entity_profile.entity_id] = entity_profile
        
        return updated_profiles
    
    async def _generate_mention_embeddings(self, mentions: List[EntityMention]):
        """Generate contextual embeddings for entity mentions"""
        if not self.embedding_model:
            return
        
        for mention in mentions:
            # Use context for better disambiguation
            context_text = f"{mention.text} {mention.context}"
            
            try:
                # Generate embedding using the same model as documents
                embedding = await self.embedding_model.get_embedding(context_text)
                mention.embeddings = np.array(embedding)
                
                # Cache for efficiency
                self.mention_embeddings_cache[f"{mention.document_id}_{mention.start_char}"] = mention.embeddings
                
            except Exception as e:
                logger.warning(f"Could not generate embedding for mention '{mention.text}': {e}")
    
    def _cluster_mentions(self, mentions: List[EntityMention]) -> Dict[str, List[EntityMention]]:
        """Cluster similar mentions into entities"""
        if not mentions:
            return {}
        
        # Filter mentions with embeddings
        mentions_with_embeddings = [m for m in mentions if m.embeddings is not None]
        
        if not mentions_with_embeddings:
            # Fallback to simple text matching
            return self._cluster_by_text_similarity(mentions)
        
        # Create similarity matrix
        embeddings_matrix = np.vstack([m.embeddings for m in mentions_with_embeddings])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Convert to distance matrix for clustering
        distance_matrix = 1 - similarity_matrix
        
        # Use DBSCAN clustering
        clustering = DBSCAN(
            eps=1 - self.similarity_threshold,  # Convert similarity to distance
            min_samples=1,
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group mentions by cluster
        clusters = defaultdict(list)
        for mention, label in zip(mentions_with_embeddings, cluster_labels):
            cluster_id = f"entity_{label}" if label != -1 else f"singleton_{mention.start_char}"
            clusters[cluster_id].append(mention)
        
        return dict(clusters)
    
    def _cluster_by_text_similarity(self, mentions: List[EntityMention]) -> Dict[str, List[EntityMention]]:
        """Fallback clustering using text similarity"""
        clusters = defaultdict(list)
        
        for mention in mentions:
            # Normalize text for comparison
            normalized_text = mention.text.lower().strip()
            
            # Find existing cluster or create new one
            found_cluster = None
            for cluster_id, cluster_mentions in clusters.items():
                for existing_mention in cluster_mentions:
                    existing_normalized = existing_mention.text.lower().strip()
                    
                    # Simple similarity check
                    if (normalized_text == existing_normalized or 
                        normalized_text in existing_normalized or 
                        existing_normalized in normalized_text):
                        found_cluster = cluster_id
                        break
                
                if found_cluster:
                    break
            
            if found_cluster:
                clusters[found_cluster].append(mention)
            else:
                # Create new cluster
                new_cluster_id = f"text_cluster_{len(clusters)}"
                clusters[new_cluster_id].append(mention)
        
        return dict(clusters)
    
    async def _create_or_update_entity_profile(self, mentions: List[EntityMention]) -> EntityProfile:
        """Create or update an entity profile from clustered mentions"""
        
        # Find most representative mention (highest confidence)
        representative_mention = max(mentions, key=lambda x: x.confidence)
        
        # Generate entity ID
        entity_id = f"entity_{hash(representative_mention.text.lower())}_{len(mentions)}"
        
        # Check if entity already exists
        existing_entity = self.entity_profiles.get(entity_id)
        
        if existing_entity:
            # Update existing entity
            existing_entity.mentions.extend(mentions)
            existing_entity.documents.update(m.document_id for m in mentions)
            existing_entity.aliases.update(m.text for m in mentions)
            return existing_entity
        
        # Create new entity profile
        canonical_name = representative_mention.text
        entity_type = representative_mention.entity_type
        
        # Collect aliases (different mention texts)
        aliases = set(m.text for m in mentions)
        
        # Collect documents
        documents = set(m.document_id for m in mentions)
        
        # Generate description
        description = await self._generate_entity_description(mentions)
        
        # Calculate average embedding
        embeddings_list = [m.embeddings for m in mentions if m.embeddings is not None]
        avg_embedding = None
        if embeddings_list:
            avg_embedding = np.mean(embeddings_list, axis=0)
        
        entity_profile = EntityProfile(
            entity_id=entity_id,
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases,
            mentions=mentions,
            documents=documents,
            description=description,
            embeddings=avg_embedding,
            created_at=str(datetime.now()),
            updated_at=str(datetime.now())
        )
        
        return entity_profile
    
    async def _generate_entity_description(self, mentions: List[EntityMention]) -> str:
        """Generate a description for an entity based on its mentions"""
        
        # Collect contexts
        contexts = [m.context for m in mentions[:5]]  # Limit to first 5 for efficiency
        combined_context = " ".join(contexts)
        
        # Simple description based on most common context patterns
        entity_text = mentions[0].text
        entity_type = mentions[0].entity_type
        
        description = f"{entity_text} is a {entity_type.lower()} mentioned across {len(set(m.document_id for m in mentions))} documents."
        
        return description

class CrossDocumentEntityLinker:
    """Links entities across documents in the knowledge graph"""
    
    def __init__(self, config):
        self.config = config
        self.entity_profiles: Dict[str, EntityProfile] = {}
        self.document_entities: Dict[str, Set[str]] = defaultdict(set)
        self.entity_relationships: List[EntityRelationship] = []
    
    def link_entities_across_documents(self, 
                                     entity_profiles: Dict[str, EntityProfile],
                                     knowledge_graph: nx.Graph) -> nx.Graph:
        """Create cross-document entity links in the knowledge graph"""
        
        self.entity_profiles = entity_profiles
        
        # Build document-entity mapping
        for entity_id, profile in entity_profiles.items():
            for doc_id in profile.documents:
                self.document_entities[doc_id].add(entity_id)
        
        # Add entity nodes to graph
        for entity_id, profile in entity_profiles.items():
            knowledge_graph.add_node(
                entity_id,
                type='entity',
                name=profile.canonical_name,
                entity_type=profile.entity_type,
                description=profile.description,
                document_count=len(profile.documents),
                mention_count=len(profile.mentions),
                centrality_score=profile.centrality_score
            )
        
        # Create entity-document relationships
        for entity_id, profile in entity_profiles.items():
            for doc_id in profile.documents:
                if knowledge_graph.has_node(doc_id):
                    knowledge_graph.add_edge(
                        entity_id,
                        doc_id,
                        type='mentions',
                        weight=len([m for m in profile.mentions if m.document_id == doc_id])
                    )
        
        # Create entity-entity relationships based on co-occurrence
        self._create_entity_cooccurrence_links(knowledge_graph)
        
        return knowledge_graph
    
    def _create_entity_cooccurrence_links(self, knowledge_graph: nx.Graph):
        """Create links between entities that co-occur in documents"""
        
        # Find entity pairs that co-occur in the same documents
        entity_pairs = defaultdict(list)
        
        for doc_id, entities in self.document_entities.items():
            entity_list = list(entities)
            
            # Create pairs of entities in the same document
            for i, entity1 in enumerate(entity_list):
                for entity2 in entity_list[i+1:]:
                    pair_key = tuple(sorted([entity1, entity2]))
                    entity_pairs[pair_key].append(doc_id)
        
        # Add edges for entity pairs with sufficient co-occurrence
        min_cooccurrence = 2  # Minimum documents they must share
        
        for (entity1, entity2), shared_docs in entity_pairs.items():
            if len(shared_docs) >= min_cooccurrence:
                
                # Calculate relationship strength
                total_docs1 = len(self.entity_profiles[entity1].documents)
                total_docs2 = len(self.entity_profiles[entity2].documents)
                shared_count = len(shared_docs)
                
                # Jaccard similarity
                union_count = total_docs1 + total_docs2 - shared_count
                jaccard_similarity = shared_count / union_count if union_count > 0 else 0
                
                # Add edge if similarity is significant
                if jaccard_similarity > 0.1:
                    knowledge_graph.add_edge(
                        entity1,
                        entity2,
                        type='co_occurs',
                        weight=jaccard_similarity,
                        shared_documents=shared_count,
                        evidence=shared_docs[:5]  # Limit evidence for performance
                    )

class EntityBasedLinkingManager:
    """Main manager for Phase 1: Entity-Based Linking & Disambiguation"""
    
    def __init__(self, config, embedding_model=None):
        self.config = config
        self.embedding_model = embedding_model
        
        # Sub-managers
        self.ner_manager = AdvancedNERManager(config)
        self.disambiguation_manager = EntityDisambiguationManager(config, embedding_model)
        self.cross_doc_linker = CrossDocumentEntityLinker(config)
        
        # State
        self.is_enabled = os.getenv("ENABLE_ADVANCED_NER", "false").lower() == "true"
        self.entity_profiles: Dict[str, EntityProfile] = {}
        
        logger.info(f"Entity-based linking manager initialized (enabled: {self.is_enabled})")
    
    async def process_documents(self, documents: Dict[str, Any], knowledge_graph: nx.Graph) -> nx.Graph:
        """Process all documents for entity extraction and linking"""
        
        if not self.is_enabled:
            logger.info("Entity-based linking disabled, skipping...")
            return knowledge_graph
        
        logger.info(f"🔍 Starting entity extraction and linking for {len(documents)} documents...")
        
        all_mentions = []
        
        # Extract entities from all documents
        for doc_id, document in documents.items():
            try:
                content = getattr(document, 'content', str(document))
                mentions = self.ner_manager.extract_entities(content, doc_id)
                all_mentions.extend(mentions)
                
                if len(mentions) > 0:
                    logger.debug(f"Found {len(mentions)} entities in {doc_id}")
                
            except Exception as e:
                logger.warning(f"Error processing document {doc_id}: {e}")
        
        logger.info(f"📊 Extracted {len(all_mentions)} entity mentions total")
        
        # Disambiguate entities
        self.entity_profiles = await self.disambiguation_manager.disambiguate_mentions(all_mentions)
        
        logger.info(f"🎯 Disambiguated into {len(self.entity_profiles)} unique entities")
        
        # Link entities across documents
        enhanced_graph = self.cross_doc_linker.link_entities_across_documents(
            self.entity_profiles, knowledge_graph
        )
        
        logger.info(f"🔗 Enhanced graph with entity-based linking")
        logger.info(f"   Graph nodes: {enhanced_graph.number_of_nodes()}")
        logger.info(f"   Graph edges: {enhanced_graph.number_of_edges()}")
        
        return enhanced_graph
    
    def get_entity_profiles(self) -> Dict[str, EntityProfile]:
        """Get all entity profiles"""
        return self.entity_profiles
    
    def get_entities_for_document(self, document_id: str) -> List[EntityProfile]:
        """Get all entities mentioned in a specific document"""
        return [
            profile for profile in self.entity_profiles.values()
            if document_id in profile.documents
        ]
    
    def search_entities(self, query: str, top_k: int = 10) -> List[EntityProfile]:
        """Search for entities by name or description"""
        query_lower = query.lower()
        
        candidates = []
        for profile in self.entity_profiles.values():
            score = 0
            
            # Exact name match
            if query_lower == profile.canonical_name.lower():
                score = 100
            # Name contains query
            elif query_lower in profile.canonical_name.lower():
                score = 80
            # Alias match
            elif any(query_lower in alias.lower() for alias in profile.aliases):
                score = 60
            # Description contains query
            elif query_lower in profile.description.lower():
                score = 40
            
            if score > 0:
                candidates.append((profile, score))
        
        # Sort by score and return top K
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [profile for profile, score in candidates[:top_k]]

# Example usage and testing
if __name__ == "__main__":
    # This would be integrated into the main GraphRAG system
    print("SOTA Phase 1: Entity-Based Linking & Disambiguation")
    print("This module provides advanced entity processing capabilities")
    print("- Advanced NER with multiple models")
    print("- Contextual entity disambiguation") 
    print("- Cross-document entity linking")
    print("- Entity profile management") 