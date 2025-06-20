#!/usr/bin/env python3
"""
Quick Test Runner for Phase 0: Enhanced Graph Traversal
Tests the immediate improvement from increased MAX_GRAPH_HOPS (2 → 4)
"""

import os
import asyncio
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def enable_phase0_testing():
    """Enable Phase 0 testing in .env file"""
    env_file = Path('.env')
    
    if not env_file.exists():
        logger.error(".env file not found!")
        return False
    
    # Read current content
    with open(env_file, 'r') as f:
        content = f.read()
    
    # Enable Phase 0 testing
    if 'RUN_PHASE0_TEST=false' in content:
        content = content.replace('RUN_PHASE0_TEST=false', 'RUN_PHASE0_TEST=true')
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        logger.info("✅ Enabled Phase 0 testing in .env file")
        return True
    elif 'RUN_PHASE0_TEST=true' in content:
        logger.info("✅ Phase 0 testing already enabled")
        return True
    else:
        logger.error("❌ Could not find RUN_PHASE0_TEST setting in .env file")
        return False

async def run_phase0_test():
    """Run Phase 0 enhanced graph traversal test"""
    logger.info("🚀 Starting Phase 0: Enhanced Graph Traversal Test")
    logger.info("Testing: MAX_GRAPH_HOPS increased from 2 → 4")
    logger.info("Expected improvement: Better connection discovery, deeper relationships")
    
    try:
        # Import and run the test framework
        from test_sota_features import SOTAGraphRAGTester
        
        tester = SOTAGraphRAGTester()
        results = await tester.run_all_tests()
        
        # Print Phase 0 results
        if 'phase0' in results:
            phase0_results = results['phase0']
            
            print("\n" + "="*60)
            print("📊 PHASE 0: ENHANCED GRAPH TRAVERSAL RESULTS")
            print("="*60)
            
            total_improvement = 0
            successful_tests = 0
            
            for result in phase0_results:
                if result.success:
                    status = "✅"
                    total_improvement += result.improvement_pct
                    successful_tests += 1
                else:
                    status = "❌"
                
                print(f"{status} {result.test_name}: {result.improvement_pct:.1f}% improvement")
                print(f"   Baseline: {result.baseline_score:.3f} | Enhanced: {result.enhanced_score:.3f}")
                print(f"   Execution time: {result.execution_time:.2f}s")
                
                if result.error_message:
                    print(f"   Error: {result.error_message}")
                print()
            
            if successful_tests > 0:
                avg_improvement = total_improvement / successful_tests
                print(f"🎯 AVERAGE IMPROVEMENT: {avg_improvement:.1f}%")
                
                if avg_improvement > 15:
                    print("🏆 EXCELLENT! Significant improvement - RECOMMENDED FOR IMPLEMENTATION")
                elif avg_improvement > 8:
                    print("⚡ GOOD! Moderate improvement - Consider implementation")
                else:
                    print("⚠️  MINIMAL improvement - May not justify the change")
            
            print("="*60)
        else:
            logger.warning("No Phase 0 results found. Check if RUN_PHASE0_TEST=true in .env")
        
        # Generate report
        report = tester.generate_test_report(results)
        logger.info("📄 Detailed report generated - check the .json and .md files!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure you've rebuilt the graph with the new .env settings")
        print("2. Check that your system has enough memory")
        print("3. Verify all dependencies are installed")
        print("4. Try running: python start_chat.py and choose option 3 to rebuild")

def main():
    """Main function"""
    print("🧪 PHASE 0 ENHANCED GRAPH TRAVERSAL TESTER")
    print("="*50)
    print("This will test the immediate improvement from:")
    print("• MAX_GRAPH_HOPS: 2 → 4 (find deeper connections)")  
    print("• DYNAMIC_SIMILARITY: Enhanced adaptive thresholds")
    print("• ENABLE_FLOW_PRUNING: PathRAG-style optimization")
    print()
    
    # Enable testing
    if not enable_phase0_testing():
        return
    
    # Run the test
    try:
        asyncio.run(run_phase0_test())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 