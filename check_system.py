#!/usr/bin/env python3
"""
Obsidian Graph RAG System Diagnostics
=====================================

This script provides comprehensive diagnostics for the Obsidian Graph RAG system.
Use this when experiencing issues with graph loading or system initialization.

Features:
- Check vault status and file counts
- Verify cache file integrity
- Test data loading capabilities
- Provide detailed troubleshooting recommendations
- Option to fix common issues automatically

Usage:
    python check_system.py
    python check_system.py --vault-path "path/to/vault"
    python check_system.py --fix-issues

Author: Assistant
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

# Import our main system
from graphrag import GraphRAGConfig, DataPersistenceManager, ObsidianGraphRAG


def setup_logging():
    """Setup logging for diagnostics."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


def print_header():
    """Print diagnostic header."""
    print("\n" + "="*70)
    print("🔍 OBSIDIAN GRAPH RAG SYSTEM DIAGNOSTICS")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


def print_section(title: str):
    """Print section header."""
    print(f"\n📋 {title}")
    print("-" * (len(title) + 4))


def check_vault_status(vault_path: str) -> dict:
    """Check vault status and return information."""
    print_section("VAULT STATUS")
    
    vault_info = {}
    vault_path_obj = Path(vault_path)
    
    if not vault_path_obj.exists():
        print(f"❌ Vault path does not exist: {vault_path}")
        vault_info['exists'] = False
        return vault_info
    
    print(f"[OK] Vault path exists: {vault_path}")
    vault_info['exists'] = True
    vault_info['path'] = str(vault_path_obj.resolve())
    
    # Count markdown files
    try:
        md_files = list(vault_path_obj.rglob("*.md"))
        vault_info['markdown_files'] = len(md_files)
        print(f"[OK] Found {len(md_files)} markdown files")
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in md_files if f.exists())
        vault_info['total_size_mb'] = round(total_size / 1024 / 1024, 2)
        print(f"[OK] Total size: {vault_info['total_size_mb']} MB")
        
        # Sample files
        if md_files:
            sample_files = [f.name for f in md_files[:5]]
            print(f"[OK] Sample files: {', '.join(sample_files)}")
            vault_info['sample_files'] = sample_files
    
    except Exception as e:
        print(f"⚠️  Error scanning vault: {e}")
        vault_info['scan_error'] = str(e)
    
    return vault_info


def check_cache_status(config: GraphRAGConfig) -> dict:
    """Check cache status and file integrity."""
    print_section("CACHE STATUS")
    
    persistence_manager = DataPersistenceManager(config)
    cache_info = {}
    
    # Check cache directory
    cache_dir = persistence_manager.data_dir
    cache_info['cache_dir'] = str(cache_dir)
    
    if not cache_dir.exists():
        print(f"❌ Cache directory does not exist: {cache_dir}")
        cache_info['exists'] = False
        return cache_info
    
    print(f"[OK] Cache directory exists: {cache_dir}")
    cache_info['exists'] = True
    
    # Check required files
    required_files = [
        ('documents.pkl', persistence_manager.documents_file),
        ('knowledge_graph.gpickle', persistence_manager.graph_file),
        ('embeddings.pkl', persistence_manager.embeddings_file),
        ('metadata.json', persistence_manager.metadata_file)
    ]
    
    cache_info['files'] = {}
    all_files_exist = True
    
    for file_name, file_path in required_files:
        file_info = {'exists': file_path.exists()}
        
        if file_path.exists():
            try:
                stat = file_path.stat()
                file_info.update({
                    'size_mb': round(stat.st_size / 1024 / 1024, 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                })
                print(f"[OK] {file_name}: {file_info['size_mb']} MB, modified {file_info['modified']}")
            except Exception as e:
                print(f"[WARN] {file_name}: exists but cannot read stats - {e}")
                file_info['stat_error'] = str(e)
        else:
            print(f"[ERROR] {file_name}: missing")
            all_files_exist = False
        
        cache_info['files'][file_name] = file_info
    
    cache_info['all_files_exist'] = all_files_exist
    return cache_info


def test_data_loading(config: GraphRAGConfig) -> dict:
    """Test actual data loading capabilities."""
    print_section("DATA LOADING TEST")
    
    loading_info = {'success': False}
    
    try:
        persistence_manager = DataPersistenceManager(config)
        
        print("🧪 Attempting to load cached data...")
        documents, knowledge_graph, embeddings = persistence_manager.load_data()
        
        print(f"[OK] Documents loaded: {len(documents)}")
        print(f"[OK] Graph loaded: {knowledge_graph.number_of_nodes()} nodes, {knowledge_graph.number_of_edges()} edges")
        print(f"[OK] Embeddings loaded: {len(embeddings)}")
        
        loading_info.update({
            'success': True,
            'documents_count': len(documents),
            'graph_nodes': knowledge_graph.number_of_nodes(),
            'graph_edges': knowledge_graph.number_of_edges(),
            'embeddings_count': len(embeddings)
        })
        
        # Basic consistency checks
        doc_ids = set(documents.keys())
        graph_nodes = set(knowledge_graph.nodes())
        embedding_ids = set(embeddings.keys())
        
        if doc_ids == graph_nodes:
            print("[OK] Document-graph consistency: perfect match")
        else:
            print(f"[WARN] Document-graph consistency: {len(doc_ids - graph_nodes)} docs missing from graph, {len(graph_nodes - doc_ids)} extra graph nodes")
        
        if embedding_ids:
            coverage = len(embedding_ids) / len(doc_ids) * 100
            print(f"[OK] Embedding coverage: {coverage:.1f}% ({len(embedding_ids)}/{len(doc_ids)})")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        loading_info['error'] = str(e)
        loading_info['error_type'] = type(e).__name__
    
    return loading_info


def run_comprehensive_diagnosis(config: GraphRAGConfig) -> dict:
    """Run comprehensive system diagnosis."""
    print_section("COMPREHENSIVE DIAGNOSIS")
    
    try:
        persistence_manager = DataPersistenceManager(config)
        diagnosis = persistence_manager.diagnose_system_status(config.vault_path)
        
        print("🔍 Running comprehensive diagnosis...")
        
        # Display key findings
        if diagnosis['issues']:
            print(f"\n❌ Issues found ({len(diagnosis['issues'])}):")
            for issue in diagnosis['issues']:
                print(f"   - {issue}")
        else:
            print("[OK] No major issues detected")
        
        if diagnosis['recommendations']:
            print(f"\n[INFO] Recommendations ({len(diagnosis['recommendations'])}):")
            for rec in diagnosis['recommendations']:
                print(f"   - {rec}")
        
        return diagnosis
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        return {'error': str(e)}


def suggest_fixes(vault_info: dict, cache_info: dict, loading_info: dict) -> list:
    """Suggest fixes based on diagnostic results."""
    print_section("SUGGESTED FIXES")
    
    fixes = []
    
    if not vault_info.get('exists', False):
        fixes.append("Fix vault path - the specified vault directory does not exist")
        print("🔧 Fix 1: Correct the vault path in your configuration")
    
    if not cache_info.get('exists', False):
        fixes.append("Initialize cache - run initial vault processing")
        print("🔧 Fix 2: Run 'python graphrag.py' to initialize the cache")
    
    elif not cache_info.get('all_files_exist', False):
        fixes.append("Rebuild cache - some cache files are missing")
        print("🔧 Fix 3: Delete cache directory and rebuild: rm -rf ./cache")
    
    elif not loading_info.get('success', False):
        error_type = loading_info.get('error_type', '')
        if 'pickle' in error_type.lower() or 'unpickling' in loading_info.get('error', '').lower():
            fixes.append("Cache corruption - rebuild from scratch")
            print("🔧 Fix 4: Cache files corrupted, delete cache and rebuild")
        else:
            fixes.append("Loading error - check file permissions and Python environment")
            print("🔧 Fix 5: Check file permissions and Python environment")
    
    if not fixes:
        print("[OK] No fixes needed - system appears healthy!")
    
    return fixes


def offer_auto_fixes(fixes: list, config: GraphRAGConfig) -> None:
    """Offer to automatically apply some fixes."""
    if not fixes:
        return
    
    print_section("AUTO-FIX OPTIONS")
    
    fixable_issues = []
    
    for fix in fixes:
        if "cache" in fix.lower() and "rebuild" in fix.lower():
            fixable_issues.append("rebuild_cache")
        elif "delete cache" in fix.lower():
            fixable_issues.append("delete_cache")
    
    if not fixable_issues:
        print("No auto-fixable issues detected.")
        return
    
    print("Available auto-fixes:")
    for i, issue in enumerate(fixable_issues, 1):
        if issue == "rebuild_cache":
            print(f"  {i}. Rebuild cache from scratch")
        elif issue == "delete_cache":
            print(f"  {i}. Delete corrupted cache files")
    
    try:
        choice = input(f"\nApply auto-fix? (1-{len(fixable_issues)}/n): ").strip().lower()
        
        if choice == 'n':
            print("Skipping auto-fixes.")
            return
        
        try:
            choice_num = int(choice) - 1
            if 0 <= choice_num < len(fixable_issues):
                issue = fixable_issues[choice_num]
                
                if issue == "delete_cache":
                    cache_dir = Path(config.cache_dir)
                    if cache_dir.exists():
                        import shutil
                        shutil.rmtree(cache_dir)
                        print(f"[OK] Deleted cache directory: {cache_dir}")
                        print("Run 'python graphrag.py' to rebuild the cache.")
                    else:
                        print("Cache directory doesn't exist.")
                
                elif issue == "rebuild_cache":
                    print("[INFO] Rebuilding cache from scratch...")
                    try:
                        graph_rag = ObsidianGraphRAG(config)
                        graph_rag.scan_and_parse_vault()
                        
                        # Save the rebuilt data
                        persistence_manager = DataPersistenceManager(config)
                        persistence_manager.save_data(
                            graph_rag.documents,
                            graph_rag.knowledge_graph,
                            {},  # Empty embeddings for now
                            config.vault_path
                        )
                        print("[OK] Cache rebuilt successfully!")
                        
                    except Exception as e:
                        print(f"❌ Rebuild failed: {e}")
            else:
                print("Invalid choice.")
                
        except ValueError:
            print("Invalid input.")
            
    except KeyboardInterrupt:
        print("\nAuto-fix cancelled.")


def save_diagnostic_report(vault_info: dict, cache_info: dict, loading_info: dict, 
                          diagnosis: dict, output_file: str = None) -> None:
    """Save diagnostic report to file."""
    if output_file is None:
        output_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'vault_info': vault_info,
        'cache_info': cache_info,
        'loading_info': loading_info,
        'diagnosis': diagnosis
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n💾 Diagnostic report saved: {output_file}")
    except Exception as e:
        print(f"⚠️  Could not save report: {e}")


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(description="Obsidian Graph RAG System Diagnostics")
    parser.add_argument('--vault-path', type=str, 
                       default="",
                       help="Path to Obsidian vault (or set OBSIDIAN_VAULT_PATH environment variable)")
    parser.add_argument('--fix-issues', action='store_true',
                       help="Offer to automatically fix detected issues")
    parser.add_argument('--save-report', type=str, nargs='?', const=True,
                       help="Save diagnostic report to file")
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    print_header()
    
    # Configure system
    config = GraphRAGConfig()
    
    # Use command line arg or environment variable for vault path
    if args.vault_path:
        config.vault_path = args.vault_path
    elif not config.vault_path:
        print("❌ No vault path specified.")
        print("Set OBSIDIAN_VAULT_PATH environment variable or use --vault-path argument.")
        print("Example: python check_system.py --vault-path C:\\path\\to\\your\\vault")
        return
    
    print(f"Target vault: {config.vault_path}")
    print(f"Cache directory: {config.cache_dir}")
    
    # Run diagnostics
    vault_info = check_vault_status(config.vault_path)
    cache_info = check_cache_status(config)
    loading_info = test_data_loading(config) if cache_info.get('all_files_exist', False) else {'success': False, 'error': 'Cache files missing'}
    diagnosis = run_comprehensive_diagnosis(config)
    
    # Analyze and suggest fixes
    fixes = suggest_fixes(vault_info, cache_info, loading_info)
    
    # Offer auto-fixes if requested
    if args.fix_issues:
        offer_auto_fixes(fixes, config)
    
    # Save report if requested
    if args.save_report:
        output_file = args.save_report if isinstance(args.save_report, str) else None
        save_diagnostic_report(vault_info, cache_info, loading_info, diagnosis, output_file)
    
    # Final summary
    print_section("SUMMARY")
    
    if vault_info.get('exists') and cache_info.get('all_files_exist') and loading_info.get('success'):
        print("🎉 System Status: HEALTHY")
        print("Your Obsidian Graph RAG system is working correctly!")
        print("You can run 'python graphrag.py' to start the AI librarian.")
    else:
        print("⚠️  System Status: NEEDS ATTENTION")
        print("Issues detected that need to be resolved.")
        if not args.fix_issues:
            print("Run with --fix-issues to get help resolving them.")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main() 