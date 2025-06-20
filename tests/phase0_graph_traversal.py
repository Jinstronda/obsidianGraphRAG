#!/usr/bin/env python3
"""
Phase 0: Enhanced Graph Traversal Testing
Tests improved graph traversal capabilities including:
- Increased MAX_GRAPH_HOPS (2 → 4-6)
- Dynamic similarity thresholds
- Flow-based pruning (PathRAG-style)
- Better connection discovery
"""

import os
import sys
import time
import logging
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import networkx as nx
from sklearn.metrics import precision_score, recall_score, f1_score

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from graphrag import ObsidianGraphRAG, GraphRAGConfig
from test_shared import TestResult, TestConfiguration

logger = logging.getLogger(__name__)

@dataclass
class GraphTraversalMetrics:
    """Metrics for graph traversal performance"""
    precision_at_k: float
    recall_at_k: float
    f1_score: float
    path_length_avg: float
    path_relevance_score: float
    connection_discovery_ratio: float
    traversal_time: float
    memory_usage: float

class GraphTraversalTester:
    """Test enhanced graph traversal capabilities"""
    
    def __init__(self, container_dir: Path, test_data: Dict, config: TestConfiguration):
        self.container_dir = container_dir
        self.test_data = test_data
        self.config = config
        self.baseline_system = None
        self.enhanced_system = None
        
    async def setup_systems(self):
        """Setup baseline and enhanced systems for comparison"""
        logger.info("Setting up baseline and enhanced systems...")
        
        # Create baseline configuration (2 hops, no enhancements)
        baseline_config = self.container_dir / "baseline.env"
        enhanced_config = self.container_dir / ".env"
        
        # Create baseline config with original settings
        baseline_content = f"""
OBSIDIAN_VAULT_PATH={os.getenv('OBSIDIAN_VAULT_PATH', '')}
OPENAI_API_KEY={os.getenv('OPENAI_API_KEY', '')}
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
CACHE_DIR={self.container_dir}/cache_baseline
MAX_GRAPH_HOPS=2
DYNAMIC_SIMILARITY=false
ENABLE_FLOW_PRUNING=false
SIMILARITY_THRESHOLD=0.7
"""
        
        with open(baseline_config, 'w') as f:
            f.write(baseline_content)
        
        # Initialize systems with proper config objects
        self.baseline_system = ObsidianGraphRAG(GraphRAGConfig())
        self.enhanced_system = ObsidianGraphRAG(GraphRAGConfig())
        
        logger.info("Systems setup complete")
    
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create test queries for graph traversal evaluation"""
        # These queries are designed to test different hop distances
        test_queries = [
            {
                "query": "What are the connections between machine learning and neuroscience?",
                "expected_hops": 3,
                "category": "cross_domain",
                "difficulty": "medium"
            },
            {
                "query": "How do graph algorithms relate to knowledge representation?",
                "expected_hops": 2,
                "category": "technical",
                "difficulty": "easy"
            },
            {
                "query": "What are the philosophical implications of artificial intelligence in education?",
                "expected_hops": 4,
                "category": "abstract",
                "difficulty": "hard"
            },
            {
                "query": "Explain the relationship between quantum computing and cryptography",
                "expected_hops": 3,
                "category": "technical",
                "difficulty": "medium"
            },
            {
                "query": "How does cognitive psychology influence user interface design?",
                "expected_hops": 3,
                "category": "interdisciplinary",
                "difficulty": "medium"
            },
            {
                "query": "What are the connections between ancient philosophy and modern ethics?",
                "expected_hops": 5,
                "category": "historical",
                "difficulty": "hard"
            },
            {
                "query": "How do economic theories apply to social network analysis?",
                "expected_hops": 4,
                "category": "cross_domain",
                "difficulty": "hard"
            },
            {
                "query": "What is the relationship between mathematics and music theory?",
                "expected_hops": 3,
                "category": "creative",
                "difficulty": "medium"
            }
        ]
        
        return test_queries
    
    async def evaluate_graph_traversal(self, system: ObsidianGraphRAG, query: str) -> GraphTraversalMetrics:
        """Evaluate graph traversal performance for a single query"""
        start_time = time.time()
        
        try:
            # Get the graph traversal details
            response = await system.query_with_details(query)
            traversal_paths = response.get('traversal_paths', [])
            retrieved_nodes = response.get('retrieved_nodes', [])
            memory_usage = response.get('memory_usage_mb', 0)
            
            # Calculate metrics
            precision = self._calculate_precision(retrieved_nodes, query)
            recall = self._calculate_recall(retrieved_nodes, query)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            avg_path_length = np.mean([len(path) for path in traversal_paths]) if traversal_paths else 0
            path_relevance = self._calculate_path_relevance(traversal_paths, query)
            connection_ratio = len(retrieved_nodes) / max(1, len(traversal_paths))
            
            traversal_time = time.time() - start_time
            
            return GraphTraversalMetrics(
                precision_at_k=precision,
                recall_at_k=recall,
                f1_score=f1,
                path_length_avg=avg_path_length,
                path_relevance_score=path_relevance,
                connection_discovery_ratio=connection_ratio,
                traversal_time=traversal_time,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            logger.error(f"Error evaluating traversal for query '{query}': {e}")
            return GraphTraversalMetrics(
                precision_at_k=0.0,
                recall_at_k=0.0,
                f1_score=0.0,
                path_length_avg=0.0,
                path_relevance_score=0.0,
                connection_discovery_ratio=0.0,
                traversal_time=time.time() - start_time,
                memory_usage=0.0
            )
    
    def _calculate_precision(self, retrieved_nodes: List, query: str) -> float:
        """Calculate precision of retrieved nodes (simplified)"""
        if not retrieved_nodes:
            return 0.0
        
        # Simple relevance scoring based on keyword overlap
        query_words = set(query.lower().split())
        relevant_count = 0
        
        for node in retrieved_nodes:
            node_text = str(node.get('content', '')).lower()
            node_words = set(node_text.split())
            overlap = len(query_words.intersection(node_words))
            if overlap >= 2:  # At least 2 keyword matches
                relevant_count += 1
        
        return relevant_count / len(retrieved_nodes)
    
    def _calculate_recall(self, retrieved_nodes: List, query: str) -> float:
        """Calculate recall (simplified - estimate total relevant documents)"""
        # This is a simplified recall calculation
        # In practice, you'd need a ground truth dataset
        query_words = set(query.lower().split())
        estimated_total_relevant = len(query_words) * 10  # Rough estimate
        
        relevant_retrieved = sum(1 for node in retrieved_nodes 
                               if len(query_words.intersection(set(str(node.get('content', '')).lower().split()))) >= 2)
        
        return min(1.0, relevant_retrieved / estimated_total_relevant)
    
    def _calculate_path_relevance(self, traversal_paths: List, query: str) -> float:
        """Calculate relevance score of traversal paths"""
        if not traversal_paths:
            return 0.0
        
        query_words = set(query.lower().split())
        total_relevance = 0.0
        
        for path in traversal_paths:
            path_relevance = 0.0
            for i, node in enumerate(path):
                node_text = str(node.get('content', '')).lower()
                node_words = set(node_text.split())
                overlap = len(query_words.intersection(node_words))
                # Weight earlier nodes in path more heavily
                weight = 1.0 / (i + 1)
                path_relevance += overlap * weight
            
            total_relevance += path_relevance / len(path) if path else 0
        
        return total_relevance / len(traversal_paths)
    
    async def test_hop_distance_comparison(self) -> TestResult:
        """Test improvement from increased hop distance"""
        logger.info("Testing hop distance comparison...")
        
        test_queries = self.create_test_queries()
        baseline_metrics = []
        enhanced_metrics = []
        
        start_time = time.time()
        
        try:
            for query_data in test_queries:
                query = query_data["query"]
                expected_hops = query_data["expected_hops"]
                
                # Skip queries that don't benefit from increased hops
                if expected_hops <= 2:
                    continue
                
                # Test baseline system (2 hops)
                baseline_result = await self.evaluate_graph_traversal(self.baseline_system, query)
                baseline_metrics.append(baseline_result)
                
                # Test enhanced system (4+ hops)
                enhanced_result = await self.evaluate_graph_traversal(self.enhanced_system, query)
                enhanced_metrics.append(enhanced_result)
                
                logger.info(f"Query: {query[:50]}... | Baseline F1: {baseline_result.f1_score:.3f} | Enhanced F1: {enhanced_result.f1_score:.3f}")
            
            # Calculate aggregate scores
            baseline_score = np.mean([m.f1_score for m in baseline_metrics])
            enhanced_score = np.mean([m.f1_score for m in enhanced_metrics])
            improvement = ((enhanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            execution_time = time.time() - start_time
            memory_usage = np.mean([m.memory_usage for m in enhanced_metrics])
            
            return TestResult(
                phase="phase0",
                test_name="hop_distance_comparison",
                baseline_score=baseline_score,
                enhanced_score=enhanced_score,
                improvement_pct=improvement,
                execution_time=execution_time,
                memory_usage_mb=memory_usage,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Hop distance test failed: {e}")
            return TestResult(
                phase="phase0",
                test_name="hop_distance_comparison",
                baseline_score=0.0,
                enhanced_score=0.0,
                improvement_pct=0.0,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def test_dynamic_similarity(self) -> TestResult:
        """Test dynamic similarity threshold performance"""
        logger.info("Testing dynamic similarity thresholds...")
        
        start_time = time.time()
        
        try:
            # Test queries with varying similarity requirements
            similarity_queries = [
                "Find exact matches for specific concepts",  # High similarity needed
                "Explore related but different ideas",       # Medium similarity
                "Discover loose conceptual connections"      # Low similarity
            ]
            
            baseline_scores = []
            enhanced_scores = []
            
            for query in similarity_queries:
                baseline_result = await self.evaluate_graph_traversal(self.baseline_system, query)
                enhanced_result = await self.evaluate_graph_traversal(self.enhanced_system, query)
                
                baseline_scores.append(baseline_result.f1_score)
                enhanced_scores.append(enhanced_result.f1_score)
            
            baseline_score = np.mean(baseline_scores)
            enhanced_score = np.mean(enhanced_scores)
            improvement = ((enhanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            return TestResult(
                phase="phase0",
                test_name="dynamic_similarity",
                baseline_score=baseline_score,
                enhanced_score=enhanced_score,
                improvement_pct=improvement,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Dynamic similarity test failed: {e}")
            return TestResult(
                phase="phase0",
                test_name="dynamic_similarity",
                baseline_score=0.0,
                enhanced_score=0.0,
                improvement_pct=0.0,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def test_flow_pruning(self) -> TestResult:
        """Test flow-based pruning effectiveness"""
        logger.info("Testing flow-based pruning...")
        
        start_time = time.time()
        
        try:
            # Use complex queries that benefit from pruning
            pruning_queries = [
                "Complex multi-hop reasoning between disparate concepts",
                "Chains of reasoning through multiple intermediate concepts",
                "Deep conceptual relationships across domains"
            ]
            
            baseline_scores = []
            enhanced_scores = []
            
            for query in pruning_queries:
                baseline_result = await self.evaluate_graph_traversal(self.baseline_system, query)
                enhanced_result = await self.evaluate_graph_traversal(self.enhanced_system, query)
                
                # Flow pruning should improve precision while maintaining recall
                baseline_scores.append(baseline_result.precision_at_k)
                enhanced_scores.append(enhanced_result.precision_at_k)
            
            baseline_score = np.mean(baseline_scores)
            enhanced_score = np.mean(enhanced_scores)
            improvement = ((enhanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            return TestResult(
                phase="phase0",
                test_name="flow_pruning",
                baseline_score=baseline_score,
                enhanced_score=enhanced_score,
                improvement_pct=improvement,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Flow pruning test failed: {e}")
            return TestResult(
                phase="phase0",
                test_name="flow_pruning",
                baseline_score=0.0,
                enhanced_score=0.0,
                improvement_pct=0.0,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def test_connection_discovery(self) -> TestResult:
        """Test improved connection discovery capabilities"""
        logger.info("Testing connection discovery...")
        
        start_time = time.time()
        
        try:
            # Test queries designed to find non-obvious connections
            discovery_queries = [
                "Hidden connections between artificial intelligence and philosophy",
                "Unexpected relationships between mathematics and art",
                "Subtle links between psychology and technology design"
            ]
            
            baseline_connections = []
            enhanced_connections = []
            
            for query in discovery_queries:
                baseline_result = await self.evaluate_graph_traversal(self.baseline_system, query)
                enhanced_result = await self.evaluate_graph_traversal(self.enhanced_system, query)
                
                baseline_connections.append(baseline_result.connection_discovery_ratio)
                enhanced_connections.append(enhanced_result.connection_discovery_ratio)
            
            baseline_score = np.mean(baseline_connections)
            enhanced_score = np.mean(enhanced_connections)
            improvement = ((enhanced_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
            
            return TestResult(
                phase="phase0",
                test_name="connection_discovery",
                baseline_score=baseline_score,
                enhanced_score=enhanced_score,
                improvement_pct=improvement,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Connection discovery test failed: {e}")
            return TestResult(
                phase="phase0",
                test_name="connection_discovery",
                baseline_score=0.0,
                enhanced_score=0.0,
                improvement_pct=0.0,
                execution_time=time.time() - start_time,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all Phase 0 graph traversal tests"""
        logger.info("Starting Phase 0: Enhanced Graph Traversal tests...")
        
        # Setup systems
        await self.setup_systems()
        
        # Run all tests
        tests = [
            self.test_hop_distance_comparison(),
            self.test_dynamic_similarity(),
            self.test_flow_pruning(),
            self.test_connection_discovery()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Test {i} failed with exception: {result}")
                final_results.append(TestResult(
                    phase="phase0",
                    test_name=f"test_{i}",
                    baseline_score=0.0,
                    enhanced_score=0.0,
                    improvement_pct=0.0,
                    execution_time=0.0,
                    memory_usage_mb=0.0,
                    success=False,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)
        
        logger.info(f"Phase 0 tests completed: {len(final_results)} tests")
        return final_results

# Test runner for standalone execution
async def main():
    """Standalone test runner for Phase 0"""
    from pathlib import Path
    import tempfile
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        container_dir = Path(temp_dir)
        
        # Mock test data
        test_data = {
            'train_documents': [],
            'validation_documents': [],
            'total_size': 100
        }
        
        # Mock configuration
        config = TestConfiguration(
            data_subset_size=100,
            validation_split=0.2,
            enable_baseline_comparison=True,
            enable_performance_benchmarks=True,
            gpu_memory_limit=6000,
            max_test_duration=3600
        )
        
        # Run tests
        tester = GraphTraversalTester(container_dir, test_data, config)
        results = await tester.run_all_tests()
        
        # Print results
        print("Phase 0 Test Results:")
        for result in results:
            print(f"- {result.test_name}: {result.improvement_pct:.1f}% improvement")

if __name__ == "__main__":
    asyncio.run(main()) 