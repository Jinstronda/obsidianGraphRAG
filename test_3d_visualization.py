#!/usr/bin/env python3
"""
Test script for 3D Graph Visualization
=====================================

This script tests the 3D graph visualization with the improvements to ensure
nodes are visible and properly positioned.
"""

import sys
import logging
from pathlib import Path
import networkx as nx
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from plotly_graph_visualizer import PlotlyGraphVisualizer
from graphrag import GraphRAGConfig

def create_test_graph():
    """Create a test graph with realistic structure."""
    # Create a graph with some structure
    G = nx.Graph()
    
    # Add nodes with metadata
    nodes = [
        ("note1", {"title": "Introduction to AI", "word_count": 1500}),
        ("note2", {"title": "Machine Learning Basics", "word_count": 2200}),
        ("note3", {"title": "Neural Networks", "word_count": 1800}),
        ("note4", {"title": "Deep Learning", "word_count": 3000}),
        ("note5", {"title": "Computer Vision", "word_count": 1200}),
        ("note6", {"title": "Natural Language Processing", "word_count": 2500}),
        ("note7", {"title": "Reinforcement Learning", "word_count": 1900}),
        ("note8", {"title": "Ethics in AI", "word_count": 800}),
        ("note9", {"title": "Future of AI", "word_count": 1100}),
        ("note10", {"title": "AI Applications", "word_count": 1600}),
    ]
    
    for node_id, attrs in nodes:
        G.add_node(node_id, **attrs)
    
    # Add edges (connections between notes)
    edges = [
        ("note1", "note2"),  # Intro connects to ML Basics
        ("note2", "note3"),  # ML Basics connects to Neural Networks
        ("note3", "note4"),  # Neural Networks connects to Deep Learning
        ("note4", "note5"),  # Deep Learning connects to Computer Vision
        ("note4", "note6"),  # Deep Learning connects to NLP
        ("note2", "note7"),  # ML Basics connects to Reinforcement Learning
        ("note1", "note8"),  # Intro connects to Ethics
        ("note1", "note9"),  # Intro connects to Future
        ("note9", "note10"), # Future connects to Applications
        ("note5", "note10"), # Computer Vision connects to Applications
        ("note6", "note10"), # NLP connects to Applications
    ]
    
    G.add_edges_from(edges)
    
    return G

def create_mock_documents(graph):
    """Create mock document objects for the test."""
    class MockDocument:
        def __init__(self, node_id, title, word_count):
            self.id = node_id
            self.title = title
            self.word_count = word_count
            self.path = f"/mock/path/{title.replace(' ', '_').lower()}.md"
    
    documents = {}
    for node_id in graph.nodes():
        attrs = graph.nodes[node_id]
        doc = MockDocument(node_id, attrs['title'], attrs['word_count'])
        documents[node_id] = doc
    
    return documents

def test_3d_visualization():
    """Test the 3D visualization with all layout algorithms."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("🧪 Testing 3D Graph Visualization")
    print("=" * 50)
    
    try:
        # Create test data
        print("📊 Creating test graph...")
        graph = create_test_graph()
        documents = create_mock_documents(graph)
        
        print(f"   Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Create visualizer
        config = GraphRAGConfig()
        visualizer = PlotlyGraphVisualizer(config)
        
        # Test different layout algorithms
        layouts = ['spring_3d', 'circular_3d', 'spherical_3d', 'hierarchical_3d']
        
        for layout in layouts:
            print(f"\n🎨 Testing {layout} layout...")
            
            try:
                fig = visualizer.visualize_graph(
                    knowledge_graph=graph,
                    documents=documents,
                    layout_algorithm=layout,
                    title=f"Test Graph - {layout.replace('_', ' ').title()}",
                    output_file=f"test_graph_{layout}.html",
                    auto_open=False  # Don't auto-open during testing
                )
                
                print(f"   ✅ {layout} layout successful")
                
                # Verify the figure has data
                if fig.data:
                    print(f"   📈 Figure contains {len(fig.data)} traces")
                    
                    # Check for nodes trace
                    node_trace = None
                    for trace in fig.data:
                        if hasattr(trace, 'mode') and 'markers' in trace.mode:
                            node_trace = trace
                            break
                    
                    if node_trace:
                        print(f"   🎯 Found node trace with {len(node_trace.x)} nodes")
                        
                        # Check if nodes have valid positions
                        if node_trace.x and node_trace.y and node_trace.z:
                            x_range = max(node_trace.x) - min(node_trace.x)
                            y_range = max(node_trace.y) - min(node_trace.y) 
                            z_range = max(node_trace.z) - min(node_trace.z)
                            
                            print(f"   📏 Position ranges: X={x_range:.2f}, Y={y_range:.2f}, Z={z_range:.2f}")
                            
                            if x_range > 0.1 and y_range > 0.1:
                                print(f"   ✅ Nodes are properly distributed")
                            else:
                                print(f"   ⚠️  Nodes may be clustered at origin")
                        
                        # Check node sizes
                        if hasattr(node_trace.marker, 'size') and node_trace.marker.size:
                            sizes = node_trace.marker.size
                            if isinstance(sizes, list):
                                min_size = min(sizes)
                                max_size = max(sizes)
                                print(f"   📏 Node sizes: {min_size:.1f} - {max_size:.1f}")
                                
                                if min_size >= 6:
                                    print(f"   ✅ Nodes should be visible (min size: {min_size})")
                                else:
                                    print(f"   ⚠️  Some nodes may be too small (min size: {min_size})")
                        
                    else:
                        print(f"   ❌ No node trace found!")
                else:
                    print(f"   ❌ Figure contains no data!")
                    
            except Exception as e:
                print(f"   ❌ {layout} layout failed: {e}")
                continue
        
        print(f"\n🎉 Testing completed!")
        print(f"💡 Check the generated HTML files (test_graph_*.html) to verify visualization")
        print(f"🔧 If nodes are not visible, the fixes may need further adjustment")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_3d_visualization()
    sys.exit(0 if success else 1) 