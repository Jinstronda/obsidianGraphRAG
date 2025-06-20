#!/usr/bin/env python3
"""
SOTA Phase 2: Co-occurrence Analysis with TF-IDF Weighting
==========================================================

Implements advanced co-occurrence analysis capabilities:
- TF-IDF weighted relationship scoring
- Temporal co-occurrence patterns
- Statistical significance testing
- Context window analysis
- Semantic relationship discovery

Based on latest research in information retrieval and knowledge graph construction.
"""

import os
import re
import logging
import asyncio
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np
import networkx as nx
from pathlib import Path
import math

# Statistical and ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.stats import chi2_contingency, fisher_exact
    from scipy.sparse import csr_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)

@dataclass
class CooccurrenceRelationship:
    """Represents a co-occurrence relationship between terms/entities"""
    term1: str
    term2: str
    cooccurrence_count: int
    term1_total_count: int
    term2_total_count: int
    total_documents: int
    document_contexts: List[str] = field(default_factory=list)
    tfidf_score: float = 0.0
    pmi_score: float = 0.0  # Pointwise Mutual Information
    chi_square_score: float = 0.0
    significance_p_value: float = 1.0
    temporal_pattern: Dict[str, int] = field(default_factory=dict)
    context_windows: List[str] = field(default_factory=list)

@dataclass
class TemporalPattern:
    """Temporal pattern of term co-occurrence"""
    term_pair: Tuple[str, str]
    time_periods: Dict[str, int] = field(default_factory=dict)  # period -> count
    trend_direction: str = "stable"  # increasing, decreasing, stable
    seasonal_pattern: bool = False
    peak_periods: List[str] = field(default_factory=list)

class TFIDFCooccurrenceAnalyzer:
    """Analyzes term co-occurrence using TF-IDF weighting"""
    
    def __init__(self, config):
        self.config = config
        self.window_size = int(os.getenv("COOCCURRENCE_WINDOW_SIZE", "5"))
        self.min_frequency = int(os.getenv("COOCCURRENCE_MIN_FREQUENCY", "3"))
        self.significance_threshold = 0.05
        
        # TF-IDF components
        self.tfidf_vectorizer = None
        self.term_document_matrix = None
        self.vocabulary = {}
        
        # Co-occurrence data
        self.cooccurrence_matrix = None
        self.term_frequencies = Counter()
        self.document_frequencies = Counter()
        self.relationships: Dict[Tuple[str, str], CooccurrenceRelationship] = {}
        
        if HAS_SKLEARN:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        logger.info(f"TF-IDF co-occurrence analyzer initialized")
    
    def analyze_documents(self, documents: Dict[str, Any]) -> Dict[Tuple[str, str], CooccurrenceRelationship]:
        """Analyze co-occurrence patterns in documents"""
        
        if not HAS_SKLEARN:
            logger.warning("sklearn not available, using simplified co-occurrence analysis")
            return self._simple_cooccurrence_analysis(documents)
        
        logger.info(f"🔍 Analyzing co-occurrence patterns in {len(documents)} documents...")
        
        # Extract text content
        document_texts = []
        document_ids = []
        
        for doc_id, document in documents.items():
            content = getattr(document, 'content', str(document))
            document_texts.append(content)
            document_ids.append(doc_id)
        
        # Build TF-IDF matrix
        self.term_document_matrix = self.tfidf_vectorizer.fit_transform(document_texts)
        self.vocabulary = self.tfidf_vectorizer.vocabulary_
        
        logger.info(f"📊 Built TF-IDF matrix: {self.term_document_matrix.shape}")
        
        # Extract co-occurrence relationships
        self._extract_cooccurrence_relationships(document_texts, document_ids)
        
        # Calculate statistical significance
        self._calculate_statistical_significance()
        
        # Filter by significance and frequency
        significant_relationships = {
            pair: rel for pair, rel in self.relationships.items()
            if (rel.cooccurrence_count >= self.min_frequency and 
                rel.significance_p_value < self.significance_threshold)
        }
        
        logger.info(f"🎯 Found {len(significant_relationships)} significant co-occurrence relationships")
        
        return significant_relationships
    
    def _extract_cooccurrence_relationships(self, document_texts: List[str], document_ids: List[str]):
        """Extract co-occurrence relationships with context windows"""
        
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        for doc_idx, (text, doc_id) in enumerate(zip(document_texts, document_ids)):
            # Tokenize and process text
            words = self._preprocess_text(text)
            
            # Extract co-occurrences within sliding windows
            for i, word1 in enumerate(words):
                if word1 not in self.vocabulary:
                    continue
                
                # Look within window
                window_start = max(0, i - self.window_size)
                window_end = min(len(words), i + self.window_size + 1)
                
                for j in range(window_start, window_end):
                    if i == j:
                        continue
                    
                    word2 = words[j]
                    if word2 not in self.vocabulary:
                        continue
                    
                    # Create ordered pair (alphabetically)
                    pair = tuple(sorted([word1, word2]))
                    
                    # Update relationship
                    if pair not in self.relationships:
                        self.relationships[pair] = CooccurrenceRelationship(
                            term1=pair[0],
                            term2=pair[1],
                            cooccurrence_count=0,
                            term1_total_count=0,
                            term2_total_count=0,
                            total_documents=len(document_texts)
                        )
                    
                    rel = self.relationships[pair]
                    rel.cooccurrence_count += 1
                    
                    # Extract context window
                    context_start = max(0, min(i, j) - 10)
                    context_end = min(len(words), max(i, j) + 10)
                    context = " ".join(words[context_start:context_end])
                    rel.context_windows.append(context)
                    
                    if doc_id not in rel.document_contexts:
                        rel.document_contexts.append(doc_id)
        
        # Calculate term frequencies
        for pair, rel in self.relationships.items():
            # Count total occurrences of each term
            rel.term1_total_count = sum(1 for doc_text in document_texts 
                                       if rel.term1.lower() in doc_text.lower())
            rel.term2_total_count = sum(1 for doc_text in document_texts 
                                       if rel.term2.lower() in doc_text.lower())
            
            # Calculate TF-IDF weighted score
            rel.tfidf_score = self._calculate_tfidf_cooccurrence_score(rel)
            
            # Calculate Pointwise Mutual Information
            rel.pmi_score = self._calculate_pmi_score(rel)
    
    def _calculate_tfidf_cooccurrence_score(self, rel: CooccurrenceRelationship) -> float:
        """Calculate TF-IDF weighted co-occurrence score"""
        
        if rel.term1 not in self.vocabulary or rel.term2 not in self.vocabulary:
            return 0.0
        
        # Get TF-IDF scores for both terms
        term1_idx = self.vocabulary[rel.term1]
        term2_idx = self.vocabulary[rel.term2]
        
        # Calculate average TF-IDF scores across documents where they co-occur
        total_score = 0.0
        count = 0
        
        for doc_idx in range(self.term_document_matrix.shape[0]):
            term1_tfidf = self.term_document_matrix[doc_idx, term1_idx]
            term2_tfidf = self.term_document_matrix[doc_idx, term2_idx]
            
            if term1_tfidf > 0 and term2_tfidf > 0:
                # Geometric mean of TF-IDF scores
                total_score += math.sqrt(term1_tfidf * term2_tfidf)
                count += 1
        
        return total_score / count if count > 0 else 0.0
    
    def _calculate_pmi_score(self, rel: CooccurrenceRelationship) -> float:
        """Calculate Pointwise Mutual Information score"""
        
        total_docs = rel.total_documents
        
        # Probabilities
        p_term1 = rel.term1_total_count / total_docs
        p_term2 = rel.term2_total_count / total_docs
        p_cooccurrence = len(rel.document_contexts) / total_docs
        
        # PMI calculation
        if p_term1 > 0 and p_term2 > 0 and p_cooccurrence > 0:
            pmi = math.log2(p_cooccurrence / (p_term1 * p_term2))
            return max(0, pmi)  # Positive PMI
        
        return 0.0
    
    def _calculate_statistical_significance(self):
        """Calculate statistical significance using chi-square test"""
        
        for pair, rel in self.relationships.items():
            # Create contingency table
            # [co-occur, not_co-occur]
            # [term1_present, term1_absent]
            
            cooccur = len(rel.document_contexts)
            term1_only = rel.term1_total_count - cooccur
            term2_only = rel.term2_total_count - cooccur
            neither = rel.total_documents - cooccur - term1_only - term2_only
            
            # Ensure non-negative values
            term1_only = max(0, term1_only)
            term2_only = max(0, term2_only)
            neither = max(0, neither)
            
            # Contingency table
            contingency_table = np.array([
                [cooccur, term1_only],
                [term2_only, neither]
            ])
            
            try:
                # Chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                rel.chi_square_score = chi2
                rel.significance_p_value = p_value
                
            except (ValueError, ZeroDivisionError):
                # Fallback for edge cases
                rel.chi_square_score = 0.0
                rel.significance_p_value = 1.0
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for co-occurrence analysis"""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Split into words
        words = text.split()
        
        # Filter out short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'}
        
        filtered_words = [
            word for word in words 
            if len(word) >= 3 and word not in stop_words
        ]
        
        return filtered_words
    
    def _simple_cooccurrence_analysis(self, documents: Dict[str, Any]) -> Dict[Tuple[str, str], CooccurrenceRelationship]:
        """Simplified co-occurrence analysis without sklearn"""
        
        logger.info("Using simplified co-occurrence analysis...")
        
        relationships = {}
        
        for doc_id, document in documents.items():
            content = getattr(document, 'content', str(document))
            words = self._preprocess_text(content)
            
            # Simple window-based co-occurrence
            for i, word1 in enumerate(words):
                window_start = max(0, i - self.window_size)
                window_end = min(len(words), i + self.window_size + 1)
                
                for j in range(window_start, window_end):
                    if i == j:
                        continue
                    
                    word2 = words[j]
                    pair = tuple(sorted([word1, word2]))
                    
                    if pair not in relationships:
                        relationships[pair] = CooccurrenceRelationship(
                            term1=pair[0],
                            term2=pair[1],
                            cooccurrence_count=0,
                            term1_total_count=0,
                            term2_total_count=0,
                            total_documents=len(documents)
                        )
                    
                    relationships[pair].cooccurrence_count += 1
        
        # Filter by minimum frequency
        return {
            pair: rel for pair, rel in relationships.items()
            if rel.cooccurrence_count >= self.min_frequency
        }

class TemporalCooccurrenceAnalyzer:
    """Analyzes temporal patterns in co-occurrence relationships"""
    
    def __init__(self, config):
        self.config = config
        self.temporal_window_days = int(os.getenv("TEMPORAL_WINDOW_SIZE", "30"))
        self.patterns: Dict[Tuple[str, str], TemporalPattern] = {}
    
    def analyze_temporal_patterns(self, 
                                documents: Dict[str, Any],
                                relationships: Dict[Tuple[str, str], CooccurrenceRelationship]) -> Dict[Tuple[str, str], TemporalPattern]:
        """Analyze temporal patterns in co-occurrence relationships"""
        
        logger.info("🕒 Analyzing temporal co-occurrence patterns...")
        
        # Extract document timestamps
        document_timestamps = self._extract_document_timestamps(documents)
        
        # Group relationships by time periods
        for pair, rel in relationships.items():
            pattern = TemporalPattern(term_pair=pair)
            
            # Count occurrences by time period
            for doc_id in rel.document_contexts:
                if doc_id in document_timestamps:
                    timestamp = document_timestamps[doc_id]
                    period = self._get_time_period(timestamp)
                    pattern.time_periods[period] = pattern.time_periods.get(period, 0) + 1
            
            # Analyze trend
            pattern.trend_direction = self._analyze_trend(pattern.time_periods)
            pattern.seasonal_pattern = self._detect_seasonal_pattern(pattern.time_periods)
            pattern.peak_periods = self._find_peak_periods(pattern.time_periods)
            
            self.patterns[pair] = pattern
        
        logger.info(f"📈 Analyzed temporal patterns for {len(self.patterns)} relationship pairs")
        
        return self.patterns
    
    def _extract_document_timestamps(self, documents: Dict[str, Any]) -> Dict[str, datetime]:
        """Extract timestamps from documents"""
        
        timestamps = {}
        
        for doc_id, document in documents.items():
            # Try to get modification time or creation time
            if hasattr(document, 'modified_at'):
                timestamps[doc_id] = document.modified_at
            elif hasattr(document, 'created_at'):
                timestamps[doc_id] = document.created_at
            else:
                # Use current time as fallback
                timestamps[doc_id] = datetime.now()
        
        return timestamps
    
    def _get_time_period(self, timestamp: datetime) -> str:
        """Convert timestamp to time period (e.g., "2024-01" for monthly periods)"""
        return timestamp.strftime("%Y-%m")
    
    def _analyze_trend(self, time_periods: Dict[str, int]) -> str:
        """Analyze trend direction in time series"""
        
        if len(time_periods) < 3:
            return "stable"
        
        # Sort by time period
        sorted_periods = sorted(time_periods.items())
        
        # Calculate simple trend
        values = [count for period, count in sorted_periods]
        
        # Linear regression slope approximation
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * v for i, v in enumerate(values))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
        
        return "stable"
    
    def _detect_seasonal_pattern(self, time_periods: Dict[str, int]) -> bool:
        """Detect if there's a seasonal pattern"""
        
        if len(time_periods) < 12:  # Need at least a year of data
            return False
        
        # Simple seasonal detection: check if certain months consistently have higher values
        monthly_totals = defaultdict(list)
        
        for period, count in time_periods.items():
            try:
                month = period.split('-')[1]  # Extract month from "YYYY-MM"
                monthly_totals[month].append(count)
            except:
                continue
        
        # Calculate average for each month
        monthly_averages = {
            month: sum(counts) / len(counts)
            for month, counts in monthly_totals.items()
            if len(counts) > 1
        }
        
        if not monthly_averages:
            return False
        
        # Check if variance between months is significant
        avg_values = list(monthly_averages.values())
        if len(avg_values) < 6:
            return False
        
        mean_avg = sum(avg_values) / len(avg_values)
        variance = sum((v - mean_avg) ** 2 for v in avg_values) / len(avg_values)
        coefficient_of_variation = (variance ** 0.5) / mean_avg if mean_avg > 0 else 0
        
        return coefficient_of_variation > 0.3  # Threshold for seasonal pattern
    
    def _find_peak_periods(self, time_periods: Dict[str, int]) -> List[str]:
        """Find peak periods with highest activity"""
        
        if not time_periods:
            return []
        
        # Find periods with counts above average + standard deviation
        counts = list(time_periods.values())
        mean_count = sum(counts) / len(counts)
        variance = sum((c - mean_count) ** 2 for c in counts) / len(counts)
        std_dev = variance ** 0.5
        
        threshold = mean_count + std_dev
        
        peak_periods = [
            period for period, count in time_periods.items()
            if count > threshold
        ]
        
        return sorted(peak_periods)

class CooccurrenceAnalysisManager:
    """Main manager for Phase 2: Co-occurrence Analysis"""
    
    def __init__(self, config):
        self.config = config
        self.is_enabled = os.getenv("ENABLE_COOCCURRENCE_ANALYSIS", "false").lower() == "true"
        self.enable_temporal = os.getenv("ENABLE_TEMPORAL_COOCCURRENCE", "false").lower() == "true"
        self.enable_statistical = os.getenv("ENABLE_STATISTICAL_SIGNIFICANCE", "false").lower() == "true"
        
        # Sub-analyzers
        self.tfidf_analyzer = TFIDFCooccurrenceAnalyzer(config)
        self.temporal_analyzer = TemporalCooccurrenceAnalyzer(config)
        
        # Results
        self.cooccurrence_relationships: Dict[Tuple[str, str], CooccurrenceRelationship] = {}
        self.temporal_patterns: Dict[Tuple[str, str], TemporalPattern] = {}
        
        logger.info(f"Co-occurrence analysis manager initialized (enabled: {self.is_enabled})")
    
    async def process_documents(self, documents: Dict[str, Any], knowledge_graph: nx.Graph) -> nx.Graph:
        """Process documents for co-occurrence analysis and enhance knowledge graph"""
        
        if not self.is_enabled:
            logger.info("Co-occurrence analysis disabled, skipping...")
            return knowledge_graph
        
        logger.info(f"🔍 Starting co-occurrence analysis for {len(documents)} documents...")
        
        # Analyze TF-IDF weighted co-occurrences
        self.cooccurrence_relationships = self.tfidf_analyzer.analyze_documents(documents)
        
        # Analyze temporal patterns if enabled
        if self.enable_temporal:
            self.temporal_patterns = self.temporal_analyzer.analyze_temporal_patterns(
                documents, self.cooccurrence_relationships
            )
        
        # Enhance knowledge graph with co-occurrence relationships
        enhanced_graph = self._enhance_graph_with_cooccurrences(knowledge_graph)
        
        logger.info(f"🔗 Enhanced graph with {len(self.cooccurrence_relationships)} co-occurrence relationships")
        
        return enhanced_graph
    
    def _enhance_graph_with_cooccurrences(self, knowledge_graph: nx.Graph) -> nx.Graph:
        """Add co-occurrence relationships to the knowledge graph"""
        
        for pair, rel in self.cooccurrence_relationships.items():
            term1, term2 = pair
            
            # Add term nodes if they don't exist
            if not knowledge_graph.has_node(term1):
                knowledge_graph.add_node(term1, type='term', name=term1)
            
            if not knowledge_graph.has_node(term2):
                knowledge_graph.add_node(term2, type='term', name=term2)
            
            # Add co-occurrence edge
            edge_attributes = {
                'type': 'cooccurrence',
                'weight': rel.tfidf_score,
                'cooccurrence_count': rel.cooccurrence_count,
                'pmi_score': rel.pmi_score,
                'chi_square': rel.chi_square_score,
                'p_value': rel.significance_p_value,
                'context_samples': rel.context_windows[:3]  # Sample contexts
            }
            
            # Add temporal information if available
            if pair in self.temporal_patterns:
                pattern = self.temporal_patterns[pair]
                edge_attributes.update({
                    'temporal_trend': pattern.trend_direction,
                    'seasonal': pattern.seasonal_pattern,
                    'peak_periods': pattern.peak_periods[:5]  # Top 5 peak periods
                })
            
            knowledge_graph.add_edge(term1, term2, **edge_attributes)
        
        return knowledge_graph
    
    def get_strongest_relationships(self, top_k: int = 20) -> List[Tuple[Tuple[str, str], CooccurrenceRelationship]]:
        """Get the strongest co-occurrence relationships by TF-IDF score"""
        
        sorted_relationships = sorted(
            self.cooccurrence_relationships.items(),
            key=lambda x: x[1].tfidf_score,
            reverse=True
        )
        
        return sorted_relationships[:top_k]
    
    def get_temporal_trends(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get relationships grouped by temporal trend"""
        
        trends = defaultdict(list)
        
        for pair, pattern in self.temporal_patterns.items():
            trends[pattern.trend_direction].append(pair)
        
        return dict(trends)
    
    def search_cooccurrences(self, term: str) -> List[Tuple[str, CooccurrenceRelationship]]:
        """Find all terms that co-occur with the given term"""
        
        results = []
        
        for (term1, term2), rel in self.cooccurrence_relationships.items():
            if term.lower() in term1.lower():
                results.append((term2, rel))
            elif term.lower() in term2.lower():
                results.append((term1, rel))
        
        # Sort by TF-IDF score
        results.sort(key=lambda x: x[1].tfidf_score, reverse=True)
        
        return results

# Example usage and testing
if __name__ == "__main__":
    print("SOTA Phase 2: Co-occurrence Analysis with TF-IDF Weighting")
    print("This module provides advanced co-occurrence analysis capabilities:")
    print("- TF-IDF weighted relationship scoring")
    print("- Temporal co-occurrence patterns")
    print("- Statistical significance testing")
    print("- Context window analysis") 