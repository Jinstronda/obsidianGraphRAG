#!/usr/bin/env python3
"""
SOTA Graph RAG Features Testing Framework
Containerized testing for advanced Graph RAG implementations

Tests 5 phases of state-of-the-art Graph RAG features:
- Phase 0: Enhanced Graph Traversal (immediate improvement)
- Phase 1: Entity-Based Linking & Disambiguation  
- Phase 2: Co-occurrence Analysis with TF-IDF
- Phase 3: Hierarchical Structuring (RAPTOR-style)
- Phase 4: Temporal Graph Analysis & Knowledge Evolution
- Phase 5: Advanced Integration Features
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import asyncio
import subprocess
import tempfile
import shutil

# Import test modules (we'll create these)
from tests.phase0_graph_traversal import GraphTraversalTester
from tests.phase1_entity_linking import EntityLinkingTester  
from tests.phase2_cooccurrence import CooccurrenceTester
from tests.phase3_hierarchical import HierarchicalTester
from tests.phase4_temporal import TemporalTester
from tests.phase5_integration import IntegrationTester

# Import main GraphRAG system
from graphrag import ObsidianGraphRAG, GraphRAGConfig

# Import shared classes
from test_shared import TestResult, TestConfiguration

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sota_testing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# TestResult and TestConfiguration classes now imported from test_shared.py

class SOTAGraphRAGTester:
    """Main testing orchestrator for SOTA Graph RAG features"""
    
    def __init__(self, config_path: str = ".env"):
        """Initialize the testing framework"""
        self.config_path = config_path
        self.test_results: List[TestResult] = []
        self.config = self._load_configuration()
        self.test_data_dir = Path("./test_data")
        self.containers_dir = Path("./test_containers")
        self.baseline_system = None
        self.enhanced_system = None
        
    def _load_configuration(self) -> TestConfiguration:
        """Load test configuration from environment"""
        # Read .env file
        env_vars = {}
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        # Remove comments from value
                        if '#' in value:
                            value = value.split('#')[0]
                        env_vars[key] = value.strip()
        
        return TestConfiguration(
            data_subset_size=int(env_vars.get('TEST_DATA_SUBSET_SIZE', 500)),
            validation_split=float(env_vars.get('TEST_VALIDATION_SPLIT', 0.2)),
            enable_baseline_comparison=env_vars.get('ENABLE_BASELINE_COMPARISON', 'true').lower() == 'true',
            enable_performance_benchmarks=env_vars.get('TEST_PERFORMANCE_BENCHMARKS', 'true').lower() == 'true',
            gpu_memory_limit=int(env_vars.get('MAX_GPU_MEMORY', 6000)),
            max_test_duration=int(env_vars.get('MAX_TEST_DURATION', 3600))  # 1 hour default
        )
    
    def setup_test_environment(self):
        """Setup isolated test environment"""
        logger.info("Setting up containerized test environment...")
        
        # Create test directories
        self.test_data_dir.mkdir(exist_ok=True)
        self.containers_dir.mkdir(exist_ok=True)
        
        # Create individual test containers
        for phase in ['phase0', 'phase1', 'phase2', 'phase3', 'phase4', 'phase5']:
            container_dir = self.containers_dir / phase
            container_dir.mkdir(exist_ok=True)
            
            # Create isolated cache for each test
            (container_dir / "cache").mkdir(exist_ok=True)
            
            # Create test-specific configuration
            self._create_test_config(container_dir, phase)
        
        logger.info("Test environment setup complete")
    
    def _create_test_config(self, container_dir: Path, phase: str):
        """Create test-specific configuration for each phase"""
        config_content = f"""
# Test Configuration for {phase.upper()}
# Isolated testing environment

OBSIDIAN_VAULT_PATH={os.environ.get('OBSIDIAN_VAULT_PATH', '')}
OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}
OPENAI_MODEL=gpt-4o-mini  # Use cheaper model for testing
EMBEDDING_MODEL=text-embedding-3-large

# Test-specific settings
CACHE_DIR={container_dir}/cache
LOG_LEVEL=DEBUG
TEST_MODE=true
TEST_PHASE={phase}

# Memory limits for testing
MAX_GPU_MEMORY={self.config.gpu_memory_limit}
EMBEDDING_BATCH_SIZE=8  # Smaller batches for testing

# Phase-specific feature flags
"""
        
        # Add phase-specific configurations
        if phase == 'phase0':
            config_content += """
MAX_GRAPH_HOPS=4
ENABLE_FLOW_PRUNING=true
DYNAMIC_SIMILARITY=true
"""
        elif phase == 'phase1':
            config_content += """
ENABLE_ADVANCED_NER=true
ENABLE_ENTITY_LINKING=true
ENABLE_ENTITY_DISAMBIGUATION=true
"""
        elif phase == 'phase2':
            config_content += """
ENABLE_COOCCURRENCE_ANALYSIS=true
ENABLE_TEMPORAL_COOCCURRENCE=true
ENABLE_STATISTICAL_SIGNIFICANCE=true
"""
        elif phase == 'phase3':
            config_content += """
ENABLE_HIERARCHICAL_CONCEPTS=true
ENABLE_CONCEPT_TAXONOMIES=true
ENABLE_RAPTOR_CLUSTERING=true
"""
        elif phase == 'phase4':
            config_content += """
ENABLE_TEMPORAL_REASONING=true
ENABLE_KNOWLEDGE_EVOLUTION=true
ENABLE_TEMPORAL_DRIFT_DETECTION=true
"""
        elif phase == 'phase5':
            config_content += """
ENABLE_MULTI_HOP_REASONING=true
ENABLE_GRAPH_ATTENTION=true
ENABLE_PATHRAG_INTEGRATION=true
"""
        
        with open(container_dir / ".env", 'w') as f:
            f.write(config_content)
    
    def prepare_test_data(self):
        """Prepare subset of data for testing"""
        logger.info(f"Preparing test dataset (size: {self.config.data_subset_size})")
        
        # Initialize baseline system to get document list
        baseline_config = GraphRAGConfig()
        self.baseline_system = ObsidianGraphRAG(baseline_config)
        
        # Get subset of documents for testing
        all_documents = self.baseline_system.get_all_documents()
        test_documents = all_documents[:self.config.data_subset_size]
        
        # Split into train/validation
        split_point = int(len(test_documents) * (1 - self.config.validation_split))
        train_docs = test_documents[:split_point]
        validation_docs = test_documents[split_point:]
        
        # Save test data
        test_data = {
            'train_documents': [doc.dict() for doc in train_docs],
            'validation_documents': [doc.dict() for doc in validation_docs],
            'total_size': len(test_documents),
            'train_size': len(train_docs),
            'validation_size': len(validation_docs),
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.test_data_dir / "test_dataset.json", 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Test data prepared: {len(train_docs)} train, {len(validation_docs)} validation")
        return test_data
    
    async def run_phase_test(self, phase: str, tester_class, test_data: Dict) -> List[TestResult]:
        """Run tests for a specific phase"""
        logger.info(f"Running {phase.upper()} tests...")
        
        try:
            # Initialize phase-specific tester
            container_dir = self.containers_dir / phase
            tester = tester_class(
                container_dir=container_dir,
                test_data=test_data,
                config=self.config
            )
            
            # Run phase tests
            results = await tester.run_all_tests()
            
            logger.info(f"{phase.upper()} tests completed: {len(results)} tests")
            return results
            
        except Exception as e:
            logger.error(f"Error in {phase} tests: {str(e)}")
            return [TestResult(
                phase=phase,
                test_name="overall",
                baseline_score=0.0,
                enhanced_score=0.0,
                improvement_pct=0.0,
                execution_time=0.0,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )]
    
    async def run_all_tests(self) -> Dict[str, List[TestResult]]:
        """Run all SOTA feature tests"""
        logger.info("Starting comprehensive SOTA Graph RAG testing...")
        
        # Setup environment
        self.setup_test_environment()
        
        # Prepare test data
        test_data = self.prepare_test_data()
        
        # Define test phases
        test_phases = [
            ('phase0', GraphTraversalTester),
            ('phase1', EntityLinkingTester),
            ('phase2', CooccurrenceTester), 
            ('phase3', HierarchicalTester),
            ('phase4', TemporalTester),
            ('phase5', IntegrationTester)
        ]
        
        # Run tests for each phase
        all_results = {}
        
        for phase, tester_class in test_phases:
            # Check if this phase should be tested
            env_var = f"RUN_{phase.upper()}_TEST"
            if os.getenv(env_var, 'false').lower() != 'true':
                logger.info(f"Skipping {phase} tests (not enabled)")
                continue
            
            phase_results = await self.run_phase_test(phase, tester_class, test_data)
            all_results[phase] = phase_results
            self.test_results.extend(phase_results)
        
        return all_results
    
    def generate_test_report(self, results: Dict[str, List[TestResult]]):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")
        
        report = {
            'test_summary': {
                'total_tests': len(self.test_results),
                'successful_tests': len([r for r in self.test_results if r.success]),
                'failed_tests': len([r for r in self.test_results if not r.success]),
                'overall_success_rate': len([r for r in self.test_results if r.success]) / len(self.test_results) if self.test_results else 0,
                'total_execution_time': sum(r.execution_time for r in self.test_results),
                'generated_at': datetime.now().isoformat()
            },
            'phase_results': {},
            'performance_summary': {},
            'recommendations': []
        }
        
        # Analyze results by phase
        for phase, phase_results in results.items():
            successful_results = [r for r in phase_results if r.success]
            
            if successful_results:
                avg_improvement = sum(r.improvement_pct for r in successful_results) / len(successful_results)
                
                report['phase_results'][phase] = {
                    'tests_run': len(phase_results),
                    'successful_tests': len(successful_results),
                    'average_improvement': avg_improvement,
                    'best_improvement': max(r.improvement_pct for r in successful_results),
                    'total_execution_time': sum(r.execution_time for r in phase_results),
                    'details': [
                        {
                            'test_name': r.test_name,
                            'baseline_score': r.baseline_score,
                            'enhanced_score': r.enhanced_score,
                            'improvement_pct': r.improvement_pct,
                            'execution_time': r.execution_time,
                            'success': r.success
                        } for r in phase_results
                    ]
                }
                
                # Generate recommendations
                if avg_improvement > 10:
                    report['recommendations'].append(f"✅ {phase.upper()}: Significant improvement ({avg_improvement:.1f}%) - RECOMMENDED FOR IMPLEMENTATION")
                elif avg_improvement > 5:
                    report['recommendations'].append(f"⚡ {phase.upper()}: Moderate improvement ({avg_improvement:.1f}%) - Consider implementation")
                else:
                    report['recommendations'].append(f"⚠️ {phase.upper()}: Minimal improvement ({avg_improvement:.1f}%) - May not justify complexity")
        
        # Save report
        report_file = f"sota_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable summary
        self._generate_human_readable_report(report, report_file.replace('.json', '_summary.md'))
        
        logger.info(f"Test report saved: {report_file}")
        return report
    
    def _generate_human_readable_report(self, report: Dict, filename: str):
        """Generate human-readable markdown report"""
        content = f"""# SOTA Graph RAG Testing Report

**Generated:** {report['test_summary']['generated_at']}

## 📊 Overall Results

- **Total Tests:** {report['test_summary']['total_tests']}
- **Successful:** {report['test_summary']['successful_tests']}
- **Failed:** {report['test_summary']['failed_tests']}
- **Success Rate:** {report['test_summary']['overall_success_rate']:.1%}
- **Total Execution Time:** {report['test_summary']['total_execution_time']:.2f}s

## 🚀 Implementation Recommendations

"""
        
        for rec in report['recommendations']:
            content += f"- {rec}\n"
        
        content += "\n## 📈 Phase-by-Phase Results\n\n"
        
        for phase, results in report['phase_results'].items():
            content += f"### {phase.upper()}\n\n"
            content += f"- **Tests Run:** {results['tests_run']}\n"
            content += f"- **Average Improvement:** {results['average_improvement']:.1f}%\n"
            content += f"- **Best Improvement:** {results['best_improvement']:.1f}%\n"
            content += f"- **Execution Time:** {results['total_execution_time']:.2f}s\n\n"
            
            content += "#### Detailed Results:\n\n"
            for detail in results['details']:
                status = "✅" if detail['success'] else "❌"
                content += f"- {status} **{detail['test_name']}**: {detail['improvement_pct']:.1f}% improvement ({detail['execution_time']:.2f}s)\n"
            
            content += "\n"
        
        with open(filename, 'w') as f:
            f.write(content)
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment...")
        
        try:
            if self.containers_dir.exists():
                shutil.rmtree(self.containers_dir)
            logger.info("Test environment cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up test environment: {e}")

async def main():
    """Main testing function"""
    tester = SOTAGraphRAGTester()
    
    try:
        # Run all tests
        results = await tester.run_all_tests()
        
        # Generate report
        report = tester.generate_test_report(results)
        
        # Print summary
        print("\n" + "="*60)
        print("SOTA GRAPH RAG TESTING COMPLETED")
        print("="*60)
        print(f"Total Tests: {len(tester.test_results)}")
        print(f"Successful: {len([r for r in tester.test_results if r.success])}")
        print(f"Failed: {len([r for r in tester.test_results if not r.success])}")
        
        if tester.test_results:
            avg_improvement = sum(r.improvement_pct for r in tester.test_results if r.success) / len([r for r in tester.test_results if r.success])
            print(f"Average Improvement: {avg_improvement:.1f}%")
        
        print("\nCheck the generated report files for detailed results!")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise
    finally:
        # Cleanup
        tester.cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main()) 