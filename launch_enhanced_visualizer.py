#!/usr/bin/env python3
"""
Enhanced Graph Visualizer Launcher
==================================

A simple launcher for enhanced graph visualization with community clustering.
"""

import sys
from pathlib import Path
import logging

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main launcher function."""
    try:
        # Import modules
        from enhanced_graph_visualizer import EnhancedGraphVisualizer, VisualizationConfig
        from graphrag import ObsidianGraphRAG, GraphRAGConfig
        
        print("🚀 Enhanced Graph Visualizer")
        print("=" * 40)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize GraphRAG system
        print("📚 Loading GraphRAG system...")
        config = GraphRAGConfig()
        graphrag_system = ObsidianGraphRAG(config)
        graphrag_system.initialize_system()
        
        if not graphrag_system.documents:
            print("❌ No documents found. Please run the main GraphRAG system first.")
            return
        
        print(f"✅ Loaded {len(graphrag_system.documents)} documents")
        print(f"📊 Graph: {graphrag_system.knowledge_graph.number_of_nodes()} nodes, {graphrag_system.knowledge_graph.number_of_edges()} edges")
        
        # Initialize enhanced visualizer
        print("🎨 Creating enhanced visualization...")
        viz_config = VisualizationConfig()
        visualizer = EnhancedGraphVisualizer(viz_config)
        
        # Load data
        if not visualizer.load_graphrag_data(graphrag_system):
            print("❌ Failed to load GraphRAG data")
            return
        
        # Create visualization
        fig = visualizer.create_enhanced_visualization(
            layout_algorithm='community_semantic_3d',
            title="Enhanced Knowledge Graph - Community Clustering"
        )
        
        print("✅ Enhanced visualization created successfully!")
        print("📁 Check the 'images' directory for output files")
        
        # Also create 2D overview
        print("🎨 Creating 2D overview...")
        fig_2d = visualizer.create_2d_overview()
        
        # Save 2D version
        output_path = Path("images") / "enhanced_graph_2d_overview.html"
        output_path.parent.mkdir(exist_ok=True)
        fig_2d.write_html(str(output_path))
        
        print("✅ 2D overview created successfully!")
        print("🎉 All visualizations completed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 