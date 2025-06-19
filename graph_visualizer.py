#!/usr/bin/env python3
"""
Graph RAG Visualizer - Interactive Knowledge Graph Explorer

This tool provides visual exploration of your Obsidian Graph RAG knowledge graph.
Features:
- Interactive network visualization
- Query simulation to see Graph RAG in action
- Community detection and analysis
- Document relationship exploration

Usage: python graph_visualizer.py
"""

import os
import pickle
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import logging

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from community import community_louvain

# Import data structures from main GraphRAG system
try:
    from graphrag import Document, TextChunk, Entity, Relationship, Community, QueryResult
    HAS_GRAPHRAG_CLASSES = True
except ImportError:
    print("⚠️  Could not import GraphRAG classes. Creating minimal versions...")
    HAS_GRAPHRAG_CLASSES = False
    
    # Minimal Document class for compatibility
    from dataclasses import dataclass, field
    from datetime import datetime
    from typing import Set, Dict, Any, Optional
    
    @dataclass
    class Document:
        id: str
        path: str
        title: str
        content: str
        frontmatter: Dict[str, Any] = field(default_factory=dict)
        tags: Set[str] = field(default_factory=set)
        wikilinks: Set[str] = field(default_factory=set)
        backlinks: Set[str] = field(default_factory=set)
        created_at: datetime = field(default_factory=datetime.now)
        modified_at: datetime = field(default_factory=datetime.now)
        word_count: int = 0
        embedding: Optional[np.ndarray] = None

# For Graph RAG simulation
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from openai import OpenAI
    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    print("⚠️  Some features require: pip install scikit-learn openai")


class GraphVisualizer:
    """
    Interactive visualizer for Graph RAG knowledge graphs.
    """
    
    def __init__(self, cache_dir: str = "./cache/processed_data"):
        """Initialize the visualizer."""
        self.cache_dir = Path(cache_dir)
        self.documents = {}
        self.graph = None
        self.embeddings = {}
        self.communities = {}
        self.node_positions = {}
        
        # Visualization settings
        self.layout_cache = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> bool:
        """Load cached graph data."""
        try:
            self.logger.info("Loading cached graph data...")
            
            # Check if files exist
            docs_file = self.cache_dir / "documents.pkl"
            graph_file = self.cache_dir / "knowledge_graph.gpickle"
            embeddings_file = self.cache_dir / "embeddings.pkl"
            
            if not all(f.exists() for f in [docs_file, graph_file]):
                self.logger.error("Required cache files not found. Run graphrag.py first.")
                return False
            
            # Load documents
            with open(docs_file, 'rb') as f:
                self.documents = pickle.load(f)
            
            # Load graph
            with open(graph_file, 'rb') as f:
                self.graph = pickle.load(f)
            
            # Load embeddings if available
            if embeddings_file.exists():
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                self.logger.info(f"Loaded {len(self.embeddings)} embeddings")
            
            self.logger.info(f"Loaded {len(self.documents)} documents and graph with {self.graph.number_of_edges()} edges")
            
            # Compute communities
            self._compute_communities()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def _compute_communities(self):
        """Compute community structure of the graph."""
        try:
            self.logger.info("Computing community structure...")
            self.communities = community_louvain.best_partition(self.graph)
            num_communities = len(set(self.communities.values()))
            self.logger.info(f"Found {num_communities} communities")
        except Exception as e:
            self.logger.warning(f"Error computing communities: {e}")
            self.communities = {node: 0 for node in self.graph.nodes()}
    
    def _compute_layout(self, layout_type: str = "spring", sample_size: Optional[int] = None):
        """Compute node positions for visualization."""
        if layout_type in self.layout_cache:
            return self.layout_cache[layout_type]
        
        try:
            self.logger.info(f"Computing {layout_type} layout...")
            
            # For large graphs, sample nodes based on importance
            if sample_size and len(self.graph.nodes()) > sample_size:
                degrees = dict(self.graph.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                sample_nodes = [node for node, _ in sorted_nodes[:sample_size]]
                subgraph = self.graph.subgraph(sample_nodes)
            else:
                subgraph = self.graph
            
            if layout_type == "spring":
                pos = nx.spring_layout(subgraph, k=1, iterations=50)
            elif layout_type == "circular":
                pos = nx.circular_layout(subgraph)
            elif layout_type == "kamada_kawai":
                # Use only for smaller graphs
                if len(subgraph.nodes()) < 500:
                    pos = nx.kamada_kawai_layout(subgraph)
                else:
                    pos = nx.spring_layout(subgraph)
            else:
                pos = nx.spring_layout(subgraph)
            
            self.layout_cache[layout_type] = pos
            return pos
            
        except Exception as e:
            self.logger.error(f"Error computing layout: {e}")
            # Fallback to random positions
            nodes = list(self.graph.nodes())[:1000]  # Limit for performance
            return {node: (np.random.random(), np.random.random()) for node in nodes}
    
    def create_overview_visualization(self, layout: str = "spring", max_nodes: int = 300):
        """Create an overview visualization of the entire graph."""
        try:
            self.logger.info(f"Creating overview visualization with {max_nodes} nodes...")
            
            # Performance warning for large graphs
            if len(self.graph.nodes()) > 1000:
                self.logger.warning(f"Large graph detected ({len(self.graph.nodes())} nodes). Using sample of {max_nodes} for performance.")
            
            # Get node positions
            pos = self._compute_layout(layout, max_nodes)
            
            # Prepare node data
            node_ids = list(pos.keys())
            x_coords = [pos[node][0] for node in node_ids]
            y_coords = [pos[node][1] for node in node_ids]
            
            # Node attributes
            node_texts = []
            node_colors = []
            node_sizes = []
            hover_texts = []
            
            for node_id in node_ids:
                doc = self.documents.get(node_id)
                if doc:
                    title = doc.title[:20] + "..." if len(doc.title) > 20 else doc.title
                    node_texts.append(title)
                    node_colors.append(self.communities.get(node_id, 0))
                    # Size based on degree centrality
                    degree = self.graph.degree(node_id)
                    node_sizes.append(max(8, min(30, degree + 5)))
                    hover_texts.append(
                        f"<b>{doc.title}</b><br>"
                        f"Words: {doc.word_count}<br>"
                        f"Connections: {degree}<br>"
                        f"Community: {self.communities.get(node_id, 0)}<br>"
                        f"Tags: {', '.join(list(doc.tags)[:3])}"
                    )
                else:
                    node_texts.append(node_id[:15])
                    node_colors.append(0)
                    node_sizes.append(8)
                    hover_texts.append(f"Document ID: {node_id}")
            
            # Create edge traces
            edge_x = []
            edge_y = []
            
            for edge in self.graph.edges():
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            # Create figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(width=0.5, color='rgba(128,128,128,0.5)'),
                hoverinfo='none',
                showlegend=False,
                name='Wiki-links'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Community", thickness=15),
                    line=dict(width=1, color='white'),
                    opacity=0.8
                ),
                text=node_texts,
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False,
                name='Documents'
            ))
            
            fig.update_layout(
                title={
                    'text': f"Obsidian Knowledge Graph Overview<br><sub>{len(node_ids)} nodes • {self.graph.number_of_edges()} connections</sub>",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=60),
                annotations=[
                    dict(
                        text="💡 Node size = connections • Color = community • Hover for details",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.5, y=-0.05,
                        xanchor="center", yanchor="top",
                        font=dict(color="gray", size=11)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                font=dict(family="Arial", size=12)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating overview: {e}")
            return go.Figure()
    
    def simulate_query_retrieval(self, query: str, api_key: Optional[str] = None):
        """Simulate Graph RAG retrieval for a query."""
        semantic_results = []
        graph_expanded_results = []
        
        try:
            self.logger.info(f"Simulating query: '{query}'")
            
            # Semantic search simulation
            if api_key and self.embeddings and HAS_ML_LIBS:
                semantic_results = self._semantic_search(query, api_key)
            else:
                # Fallback: simple text matching
                semantic_results = self._text_search(query)
            
            # Graph expansion simulation
            graph_expanded_results = self._graph_expansion(semantic_results)
            
            # Create visualization
            fig = self._create_query_visualization(query, semantic_results, graph_expanded_results)
            
            return semantic_results, graph_expanded_results, fig
            
        except Exception as e:
            self.logger.error(f"Error simulating query: {e}")
            return [], [], go.Figure()
    
    def _semantic_search(self, query: str, api_key: str, top_k: int = 5):
        """Perform semantic search using embeddings."""
        try:
            client = OpenAI(api_key=api_key)
            
            # Generate query embedding
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = np.array(response.data[0].embedding)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                if doc_id in self.documents:
                    similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                    similarities.append((doc_id, similarity))
            
            # Sort and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [doc_id for doc_id, _ in similarities[:top_k]]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _text_search(self, query: str, top_k: int = 8):
        """Fallback text-based search."""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        scores = []
        for doc_id, doc in self.documents.items():
            content = f"{doc.title} {doc.content}".lower()
            score = 0
            
            # Count word matches
            for word in query_words:
                if word in content:
                    score += content.count(word)
            
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score and return top results
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scores[:top_k]]
    
    def _graph_expansion(self, seed_nodes: List[str], max_hops: int = 2, max_per_hop: int = 3):
        """Expand search results using graph traversal."""
        expanded = set(seed_nodes)
        current_layer = set(seed_nodes)
        
        for hop in range(max_hops):
            next_layer = set()
            for node in current_layer:
                if node in self.graph:
                    neighbors = list(self.graph.neighbors(node))
                    # Sort by degree to prioritize important nodes
                    neighbors.sort(key=lambda x: self.graph.degree(x), reverse=True)
                    next_layer.update(neighbors[:max_per_hop])
            
            next_layer = next_layer - expanded  # Remove already included nodes
            expanded.update(next_layer)
            current_layer = next_layer
            
            if not current_layer:  # No more nodes to expand
                break
        
        return list(expanded)
    
    def _create_query_visualization(self, query: str, semantic_results: List[str], 
                                  graph_expanded_results: List[str]):
        """Create visualization showing query results."""
        try:
            # Get layout for visualization (sample for performance)
            pos = self._compute_layout("spring", 800)
            
            # Categorize nodes
            semantic_set = set(semantic_results)
            expanded_set = set(graph_expanded_results) - semantic_set
            other_nodes = set(pos.keys()) - set(graph_expanded_results)
            
            fig = go.Figure()
            
            # Add edges with different styles for result paths
            result_edges_x, result_edges_y = [], []
            other_edges_x, other_edges_y = [], []
            
            for edge in self.graph.edges():
                if edge[0] in pos and edge[1] in pos:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    
                    # Highlight edges connected to results
                    if (edge[0] in graph_expanded_results or edge[1] in graph_expanded_results):
                        result_edges_x.extend([x0, x1, None])
                        result_edges_y.extend([y0, y1, None])
                    else:
                        other_edges_x.extend([x0, x1, None])
                        other_edges_y.extend([y0, y1, None])
            
            # Add regular edges
            fig.add_trace(go.Scatter(
                x=other_edges_x, y=other_edges_y,
                mode='lines',
                line=dict(width=0.3, color='rgba(200,200,200,0.3)'),
                hoverinfo='none',
                showlegend=False,
                name='Other connections'
            ))
            
            # Add result path edges
            fig.add_trace(go.Scatter(
                x=result_edges_x, y=result_edges_y,
                mode='lines',
                line=dict(width=1.5, color='rgba(255,100,100,0.6)'),
                hoverinfo='none',
                showlegend=False,
                name='Result connections'
            ))
            
            # Add different node types
            node_configs = [
                (other_nodes, 'lightgray', 'Other Documents', 6, 0.4),
                (expanded_set, '#FFA500', 'Graph Expanded', 12, 0.8),
                (semantic_set, '#FF4444', 'Semantic Match', 16, 1.0)
            ]
            
            for nodes, color, name, size, opacity in node_configs:
                if nodes:
                    node_list = [n for n in nodes if n in pos]
                    if not node_list:
                        continue
                        
                    x_coords = [pos[node][0] for node in node_list]
                    y_coords = [pos[node][1] for node in node_list]
                    
                    hover_texts = []
                    for node in node_list:
                        doc = self.documents.get(node)
                        if doc:
                            hover_texts.append(f"<b>{doc.title}</b><br>Type: {name}<br>Words: {doc.word_count}")
                        else:
                            hover_texts.append(f"<b>{node}</b><br>Type: {name}")
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords, y=y_coords,
                        mode='markers',
                        marker=dict(
                            size=size,
                            color=color,
                            opacity=opacity,
                            line=dict(width=1, color='white')
                        ),
                        hovertext=hover_texts,
                        hoverinfo='text',
                        name=name,
                        showlegend=True
                    ))
            
            fig.update_layout(
                title={
                    'text': f"Graph RAG Query Simulation<br><sub>Query: '{query}' • {len(semantic_results)} semantic • {len(expanded_set)} expanded</sub>",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                hovermode='closest',
                legend=dict(
                    x=0.02, y=0.98,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                ),
                margin=dict(b=20,l=5,r=5,t=80),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                font=dict(family="Arial", size=12)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating query visualization: {e}")
            return go.Figure()
    
    def create_analysis_dashboard(self):
        """Create a dashboard with various graph analysis metrics."""
        try:
            # Compute graph metrics
            degrees = dict(self.graph.degree())
            
            # Sample for betweenness centrality calculation (expensive for large graphs)
            sample_size = min(1000, len(self.graph.nodes()))
            sample_nodes = list(degrees.keys())[:sample_size]
            subgraph = self.graph.subgraph(sample_nodes)
            betweenness = nx.betweenness_centrality(subgraph)
            
            # Community sizes
            community_sizes = {}
            for node, comm in self.communities.items():
                community_sizes[comm] = community_sizes.get(comm, 0) + 1
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Document Connection Distribution', 
                    'Community Sizes', 
                    'Most Connected Documents', 
                    'Bridge Documents (High Betweenness)'
                ),
                specs=[[{"type": "histogram"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "bar"}]]
            )
            
            # Degree distribution
            fig.add_trace(
                go.Histogram(
                    x=list(degrees.values()), 
                    nbinsx=30, 
                    name="Connections",
                    marker_color='skyblue'
                ),
                row=1, col=1
            )
            
            # Community sizes (top 20)
            top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:20]
            fig.add_trace(
                go.Bar(
                    x=[f"C{comm}" for comm, _ in top_communities], 
                    y=[size for _, size in top_communities], 
                    name="Documents in Community",
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
            
            # Top degree nodes (most connected)
            top_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
            top_degree_titles = []
            for node, degree in top_degree:
                doc = self.documents.get(node)
                title = doc.title[:25] + "..." if doc and len(doc.title) > 25 else (doc.title if doc else node[:25])
                top_degree_titles.append(f"{title} ({degree})")
            
            fig.add_trace(
                go.Bar(
                    y=top_degree_titles,
                    x=[deg for _, deg in top_degree], 
                    orientation='h', 
                    name="Connections",
                    marker_color='orange'
                ),
                row=2, col=1
            )
            
            # Top betweenness nodes (bridge documents)
            top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:15]
            top_bet_titles = []
            for node, bet in top_betweenness:
                doc = self.documents.get(node)
                title = doc.title[:25] + "..." if doc and len(doc.title) > 25 else (doc.title if doc else node[:25])
                top_bet_titles.append(f"{title} ({bet:.3f})")
            
            fig.add_trace(
                go.Bar(
                    y=top_bet_titles,
                    x=[bet for _, bet in top_betweenness], 
                    orientation='h', 
                    name="Betweenness",
                    marker_color='coral'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title={
                    'text': "Knowledge Graph Analysis Dashboard",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                height=700,
                showlegend=False,
                font=dict(family="Arial", size=11)
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Number of Connections", row=1, col=1)
            fig.update_yaxes(title_text="Number of Documents", row=1, col=1)
            fig.update_xaxes(title_text="Community ID", row=1, col=2)
            fig.update_yaxes(title_text="Documents in Community", row=1, col=2)
            fig.update_xaxes(title_text="Number of Connections", row=2, col=1)
            fig.update_xaxes(title_text="Betweenness Centrality", row=2, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating analysis dashboard: {e}")
            return go.Figure()

def main():
    """Main function to run the visualizer."""
    print("🎨 Graph RAG Knowledge Graph Visualizer")
    print("=" * 45)
    
    # Initialize visualizer
    visualizer = GraphVisualizer()
    
    # Load data
    print("📚 Loading cached graph data...")
    if not visualizer.load_data():
        print("❌ Failed to load data. Please run graphrag.py first to process your vault.")
        return
    
    num_docs = len(visualizer.documents)
    num_edges = visualizer.graph.number_of_edges()
    num_communities = len(set(visualizer.communities.values()))
    
    print(f"✅ Successfully loaded:")
    print(f"   📄 {num_docs:,} documents")
    print(f"   🔗 {num_edges:,} connections")
    print(f"   🏘️  {num_communities} communities")
    
    # Performance recommendations
    if num_docs > 2000:
        print(f"\n⚠️  Large graph detected! For better performance:")
        print(f"   • Network visualization will sample nodes")
        print(f"   • Consider exploring smaller subsets first")
        print(f"   • Use the analysis dashboard for overview statistics")
    
    # User choice menu
    print(f"\n🎯 What would you like to visualize?")
    print(f"   1. Quick overview (300 most connected documents)")
    print(f"   2. Medium overview (500 documents)")
    print(f"   3. Large overview (1000 documents) ⚠️  Slower")
    print(f"   4. Analysis dashboard (statistics)")
    print(f"   5. Query simulation")
    print(f"   6. Generate all")
    
    try:
        choice = input("\nEnter your choice (1-6) or press Enter for quick overview: ").strip()
        if not choice:
            choice = "1"
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return
    
    try:
        if choice in ["1", "6"]:
            print("\n📊 Generating quick overview (300 nodes)...")
            overview_fig = visualizer.create_overview_visualization(max_nodes=300)
            overview_fig.write_html("graph_overview_quick.html")
            print("   ✅ Saved as graph_overview_quick.html")
        
        if choice in ["2", "6"]:
            print("\n📊 Generating medium overview (500 nodes)...")
            overview_fig = visualizer.create_overview_visualization(max_nodes=500)
            overview_fig.write_html("graph_overview_medium.html")
            print("   ✅ Saved as graph_overview_medium.html")
            
        if choice in ["3", "6"]:
            print("\n📊 Generating large overview (1000 nodes) - this may take a while...")
            overview_fig = visualizer.create_overview_visualization(max_nodes=1000)
            overview_fig.write_html("graph_overview_large.html")
            print("   ✅ Saved as graph_overview_large.html")
        
        if choice in ["4", "6"]:
            print("\n📈 Building analysis dashboard...")
            analysis_fig = visualizer.create_analysis_dashboard()
            analysis_fig.write_html("graph_analysis.html")
            print("   ✅ Saved as graph_analysis.html")
        
        if choice in ["5", "6"]:
            print("\n🔍 Setting up query simulation...")
            
            # Get API key for advanced query simulation
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("🔑 For advanced semantic search:")
                api_key = input("   Enter OpenAI API key (or press Enter for text search): ").strip()
            
            # Get query from user
            if choice == "5":
                query = input("Enter your test query: ").strip()
                if not query:
                    query = "learning" if not api_key else "machine learning artificial intelligence"
            else:
                query = "learning" if not api_key else "machine learning artificial intelligence"
                
            print(f"   Running query: '{query}'...")
            semantic_results, expanded_results, query_fig = visualizer.simulate_query_retrieval(query, api_key)
            
            print(f"   📊 Results: {len(semantic_results)} direct matches, {len(expanded_results) - len(semantic_results)} graph-expanded")
            query_fig.write_html("query_simulation.html")
            print("   ✅ Saved as query_simulation.html")
        
        # Summary
        print(f"\n🎉 Visualization complete!")
        
        if choice == "1":
            print("   📈 graph_overview_quick.html - Quick network overview")
            filename = "graph_overview_quick.html"
        elif choice == "2":
            print("   📈 graph_overview_medium.html - Medium network overview") 
            filename = "graph_overview_medium.html"
        elif choice == "3":
            print("   📈 graph_overview_large.html - Large network overview")
            filename = "graph_overview_large.html"
        elif choice == "4":
            print("   📊 graph_analysis.html - Statistical analysis")
            filename = "graph_analysis.html"
        elif choice == "5":
            print("   🔍 query_simulation.html - Query demonstration")
            filename = "query_simulation.html"
        else:  # choice == "6"
            print("   📈 graph_overview_*.html - Network overviews (3 sizes)")
            print("   📊 graph_analysis.html - Statistical analysis")
            print("   🔍 query_simulation.html - Query demonstration")
            filename = "graph_overview_quick.html"
        
        print(f"\n💡 Pro tips:")
        print("   • Start with the quick overview to get oriented")
        print("   • Use zoom/pan to explore the network")
        print("   • Hover over nodes for document details")
        print("   • Analysis dashboard shows important statistics")
        
        # Auto-open visualization
        try:
            import webbrowser
            print(f"\n🌐 Opening {filename} in your browser...")
            webbrowser.open(filename)
        except Exception as e:
            print(f"\n🌐 Open {filename} in your browser to start exploring!")
            
    except Exception as e:
        print(f"\n❌ Error generating visualizations: {e}")
        print("Try choosing a smaller visualization option.")

if __name__ == "__main__":
    main()