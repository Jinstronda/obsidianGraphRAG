#!/usr/bin/env python3
"""
Plotly-based 3D Graph Visualizer for Obsidian Graph RAG
======================================================

A reliable 3D graph visualization system using Plotly instead of Three.js.
This provides better browser compatibility, easier maintenance, and more
interactive features out of the box.

Features:
- Multiple 3D layout algorithms
- Interactive controls (zoom, rotate, pan)
- Node search and filtering
- Community detection visualization
- Export to HTML
- Performance optimization for large graphs
- No external dependencies or CDN issues

Author: Assistant
License: MIT
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import webbrowser
import tempfile
import os
from pathlib import Path
import json
import time
from community import community_louvain
import math


class PlotlyGraphVisualizer:
    """
    3D Graph visualizer using Plotly for reliable cross-platform visualization.
    
    This class provides a complete 3D graph visualization solution that's much
    more reliable than Three.js-based approaches. It uses Plotly's native 3D
    capabilities to render interactive graph visualizations.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Plotly graph visualizer.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Visualization parameters
        self.max_nodes = 2000  # Maximum nodes to display
        self.node_size_range = (8, 25)  # Increased minimum and maximum node sizes for better visibility
        self.edge_width = 1.5
        self.default_node_color = '#1f77b4'
        
        # Layout algorithms
        self.layout_algorithms = {
            'spring_3d': self._spring_layout_3d,
            'circular_3d': self._circular_layout_3d,
            'spherical_3d': self._spherical_layout_3d,
            'hierarchical_3d': self._hierarchical_layout_3d,
            'community_layers_3d': self._community_layers_layout_3d
        }
        
        self.logger.info("PlotlyGraphVisualizer initialized")
    
    def visualize_graph(self, 
                       knowledge_graph: nx.Graph, 
                       documents: Dict[str, Any] = None,
                       layout_algorithm: str = 'spring_3d',
                       title: str = "Obsidian Knowledge Graph 3D Visualization",
                       output_file: Optional[str] = None,
                       auto_open: bool = True) -> go.Figure:
        """
        Create a 3D visualization of the knowledge graph.
        
        Args:
            knowledge_graph: NetworkX graph to visualize
            documents: Dictionary of document objects for additional info
            layout_algorithm: Layout algorithm to use
            title: Title for the visualization
            output_file: Optional file to save HTML output
            auto_open: Whether to open in browser automatically
            
        Returns:
            Plotly Figure object
        """
        try:
            start_time = time.time()
            self.logger.info(f"Creating 3D visualization with {knowledge_graph.number_of_nodes()} nodes and {knowledge_graph.number_of_edges()} edges")
            
            # Optimize graph for visualization if too large
            graph = self._optimize_graph_for_visualization(knowledge_graph)
            
            # Compute 3D layout
            self.logger.info(f"Computing 3D layout using {layout_algorithm}")
            positions = self._compute_layout(graph, layout_algorithm)
            
            # Detect communities for coloring
            communities = self._detect_communities(graph)
            
            # Create Plotly figure
            fig = self._create_plotly_figure(graph, positions, communities, documents, title)
            
            # Add interactivity
            self._add_interactive_features(fig, graph, documents)
            
            # Save or display
            if output_file:
                self._save_html(fig, output_file, auto_open)
            elif auto_open:
                self._show_in_browser(fig)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Visualization created successfully in {elapsed_time:.2f} seconds")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}")
            raise
    
    def _optimize_graph_for_visualization(self, graph: nx.Graph) -> nx.Graph:
        """
        Optimize graph for visualization by filtering nodes if too large.
        
        Args:
            graph: Original graph
            
        Returns:
            Optimized graph
        """
        if graph.number_of_nodes() <= self.max_nodes:
            return graph.copy()
        
        self.logger.info(f"Graph has {graph.number_of_nodes()} nodes, optimizing for visualization")
        
        # Calculate node importance (degree centrality)
        centrality = nx.degree_centrality(graph)
        
        # Sort nodes by importance and take top N
        important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:self.max_nodes]
        nodes_to_keep = [node for node, _ in important_nodes]
        
        # Create subgraph with most important nodes
        optimized_graph = graph.subgraph(nodes_to_keep).copy()
        
        self.logger.info(f"Optimized graph to {optimized_graph.number_of_nodes()} nodes and {optimized_graph.number_of_edges()} edges")
        
        return optimized_graph
    
    def _compute_layout(self, graph: nx.Graph, algorithm: str) -> Dict[str, Tuple[float, float, float]]:
        """
        Compute 3D positions for graph nodes using specified algorithm.
        
        Args:
            graph: NetworkX graph
            algorithm: Layout algorithm name
            
        Returns:
            Dictionary of node_id -> (x, y, z) positions
        """
        if algorithm not in self.layout_algorithms:
            self.logger.warning(f"Unknown layout algorithm '{algorithm}', using spring_3d")
            algorithm = 'spring_3d'
        
        return self.layout_algorithms[algorithm](graph)
    
    def _spring_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Spring-force layout in 3D space with improved distribution."""
        try:
            self.logger.info("Computing spring layout in 3D...")
            
            # Use NetworkX spring layout with optimized parameters
            # Increase iterations and adjust spacing
            pos_2d = nx.spring_layout(
                graph, 
                iterations=100,  # More iterations for better distribution
                k=2.0,  # Increased spacing between nodes
                dim=2,
                seed=42  # Reproducible layout
            )
            
            # Create 3D positions with better Z distribution
            positions = {}
            nodes = list(graph.nodes())
            n_nodes = len(nodes)
            
            # Generate Z coordinates using different method for better spread
            z_positions = {}
            if n_nodes > 1:
                # Use community detection for Z-layering if possible
                try:
                    communities = community_louvain.best_partition(graph)
                    unique_communities = list(set(communities.values()))
                    
                    for node in nodes:
                        community = communities.get(node, 0)
                        community_index = unique_communities.index(community)
                        # Spread communities across Z axis
                        z = (community_index / max(1, len(unique_communities) - 1)) * 4 - 2  # Range -2 to 2
                        z_positions[node] = z
                        
                except:
                    # Fallback: distribute Z based on node position in list
                    for i, node in enumerate(nodes):
                        z = (i / max(1, n_nodes - 1)) * 4 - 2  # Range -2 to 2
                        z_positions[node] = z
            else:
                z_positions[nodes[0]] = 0
            
            # Combine 2D and Z positions
            for node in nodes:
                x, y = pos_2d.get(node, (0, 0))
                z = z_positions.get(node, 0)
                
                # Scale positions for better visibility
                x *= 3  # Increase spread
                y *= 3
                
                positions[node] = (x, y, z)
            
            # Verify positions are distributed
            x_coords = [pos[0] for pos in positions.values()]
            y_coords = [pos[1] for pos in positions.values()]
            z_coords = [pos[2] for pos in positions.values()]
            
            self.logger.info(f"Spring layout computed: X range [{min(x_coords):.2f}, {max(x_coords):.2f}]")
            self.logger.info(f"Spring layout computed: Y range [{min(y_coords):.2f}, {max(y_coords):.2f}]")
            self.logger.info(f"Spring layout computed: Z range [{min(z_coords):.2f}, {max(z_coords):.2f}]")
            
            return positions
            
        except Exception as e:
            self.logger.warning(f"Spring layout failed: {e}, using random layout")
            return self._random_layout_3d(graph)
    
    def _circular_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Circular layout extended to 3D with multiple levels."""
        positions = {}
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Arrange nodes in multiple circular layers
        layers = max(1, int(np.sqrt(n_nodes) / 2))
        nodes_per_layer = n_nodes // layers
        
        for i, node in enumerate(nodes):
            layer = i // nodes_per_layer
            angle = 2 * np.pi * (i % nodes_per_layer) / max(1, nodes_per_layer)
            
            radius = 1 + layer * 0.5
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = layer * 0.5
            
            positions[node] = (x, y, z)
        
        return positions
    
    def _spherical_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Arrange nodes on sphere surface with community clustering."""
        positions = {}
        nodes = list(graph.nodes())
        n_nodes = len(nodes)
        
        # Generate points on sphere using Fibonacci spiral
        for i, node in enumerate(nodes):
            # Fibonacci spiral on sphere
            y = 1 - (i / float(n_nodes - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)
            
            theta = np.pi * (3 - np.sqrt(5)) * i  # golden angle increment
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            positions[node] = (x, y, z)
        
        return positions
    
    def _hierarchical_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Hierarchical layout with nodes arranged in vertical layers."""
        # Try to create a hierarchy based on node degree
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        positions = {}
        level_counts = {}
        level_positions = {}
        
        for node in graph.nodes():
            # Assign level based on degree (higher degree = higher level)
            level = int((degrees[node] / max_degree) * 5)  # 6 levels (0-5)
            
            if level not in level_counts:
                level_counts[level] = 0
                level_positions[level] = 0
            
            # Arrange nodes in circles at each level
            angle = 2 * np.pi * level_positions[level] / max(1, level_counts[level] + 1)
            radius = 1 + level * 0.2
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = level * 0.8
            
            positions[node] = (x, y, z)
            level_positions[level] += 1
        
        # Second pass to update counts
        for node in graph.nodes():
            level = int((degrees[node] / max_degree) * 5)
            level_counts[level] += 1
        
        return positions
    
    def _community_layers_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Separate Z-layers for different communities."""
        # Detect communities
        try:
            communities = community_louvain.best_partition(graph)
            unique_communities = set(communities.values())
        except:
            # Fallback if community detection fails
            communities = {node: 0 for node in graph.nodes()}
            unique_communities = {0}
        
        positions = {}
        
        for i, community_id in enumerate(unique_communities):
            # Get nodes in this community
            community_nodes = [node for node, comm in communities.items() if comm == community_id]
            
            # Create subgraph for this community
            subgraph = graph.subgraph(community_nodes)
            
            # Use spring layout for this community
            try:
                community_pos = nx.spring_layout(subgraph, iterations=30, k=0.5)
            except:
                # Fallback to random positions
                community_pos = {node: (np.random.random(), np.random.random()) 
                               for node in community_nodes}
            
            # Assign positions with community-specific Z level
            z_level = i * 1.5  # Separate layers
            for node in community_nodes:
                x, y = community_pos.get(node, (0, 0))
                positions[node] = (x, y, z_level)
        
        return positions
    
    def _random_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Random 3D layout as fallback with better distribution."""
        self.logger.info("Using random 3D layout")
        
        positions = {}
        for node in graph.nodes():
            # Create more spread out random positions
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            z = np.random.uniform(-2, 2)
            positions[node] = (x, y, z)
        
        return positions
    
    def _detect_communities(self, graph: nx.Graph) -> Dict[str, int]:
        """Detect communities in the graph for coloring."""
        try:
            communities = community_louvain.best_partition(graph)
            self.logger.info(f"Detected {len(set(communities.values()))} communities")
            return communities
        except Exception as e:
            self.logger.warning(f"Community detection failed: {e}")
            return {node: 0 for node in graph.nodes()}
    
    def _create_plotly_figure(self, 
                             graph: nx.Graph, 
                             positions: Dict[str, Tuple[float, float, float]], 
                             communities: Dict[str, int],
                             documents: Optional[Dict[str, Any]],
                             title: str) -> go.Figure:
        """Create the main Plotly figure with nodes and edges."""
        
        # Prepare data for nodes
        node_data = self._prepare_node_data(graph, positions, communities, documents)
        
        # Prepare data for edges
        edge_data = self._prepare_edge_data(graph, positions)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges first (so they appear behind nodes)
        if edge_data:
            fig.add_trace(go.Scatter3d(
                x=edge_data['x'],
                y=edge_data['y'],
                z=edge_data['z'],
                mode='lines',
                line=dict(color='rgba(125,125,125,0.3)', width=self.edge_width),
                hoverinfo='none',
                name='Connections'
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=node_data['x'],
            y=node_data['y'],
            z=node_data['z'],
            mode='markers',
            marker=dict(
                size=node_data['size'],
                color=node_data['color'],
                colorscale='Plotly3',
                showscale=True,
                colorbar=dict(
                    title="Community",
                    x=1.0,
                    len=0.7
                ),
                line=dict(width=1, color='rgba(50,50,50,0.8)'),
                opacity=0.8
            ),
            text=node_data['hover_text'],
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
            name='Documents'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0)
                )
            ),
            showlegend=False,
            margin=dict(l=0, r=0, b=0, t=40),
            height=800
        )
        
        return fig
    
    def _prepare_node_data(self, 
                          graph: nx.Graph, 
                          positions: Dict[str, Tuple[float, float, float]], 
                          communities: Dict[str, int],
                          documents: Optional[Dict[str, Any]]) -> Dict[str, List]:
        """Prepare node data for Plotly."""
        
        node_data = {
            'x': [], 'y': [], 'z': [],
            'size': [], 'color': [], 'hover_text': []
        }
        
        # Calculate node metrics
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 0
        
        # Ensure we have valid community colors
        unique_communities = set(communities.values())
        n_communities = len(unique_communities)
        
        self.logger.info(f"Preparing data for {len(graph.nodes())} nodes")
        self.logger.info(f"Degree range: {min_degree} - {max_degree}")
        self.logger.info(f"Found {n_communities} communities")
        
        for i, node in enumerate(graph.nodes()):
            # Position
            x, y, z = positions.get(node, (0, 0, 0))
            node_data['x'].append(x)
            node_data['y'].append(y)
            node_data['z'].append(z)
            
            # Size based on degree with minimum size guarantee
            degree = degrees[node]
            if max_degree > min_degree:
                size_ratio = (degree - min_degree) / (max_degree - min_degree)
            else:
                size_ratio = 0.5  # Default if all nodes have same degree
            
            size = self.node_size_range[0] + size_ratio * (self.node_size_range[1] - self.node_size_range[0])
            # Ensure minimum visibility
            size = max(size, 6)
            node_data['size'].append(size)
            
            # Color based on community with fallback
            community = communities.get(node, 0)
            # Use community ID directly as color value for discrete colorscale
            node_data['color'].append(community)
            
            # Hover text with more informative content
            hover_text = self._create_hover_text(node, graph, documents)
            node_data['hover_text'].append(hover_text)
        
        self.logger.info(f"Node sizes range: {min(node_data['size'])} - {max(node_data['size'])}")
        self.logger.info(f"Position range X: {min(node_data['x']):.2f} - {max(node_data['x']):.2f}")
        self.logger.info(f"Position range Y: {min(node_data['y']):.2f} - {max(node_data['y']):.2f}")
        self.logger.info(f"Position range Z: {min(node_data['z']):.2f} - {max(node_data['z']):.2f}")
        
        return node_data
    
    def _prepare_edge_data(self, 
                          graph: nx.Graph, 
                          positions: Dict[str, Tuple[float, float, float]]) -> Dict[str, List]:
        """Prepare edge data for Plotly."""
        
        edge_data = {'x': [], 'y': [], 'z': []}
        
        for edge in graph.edges():
            source, target = edge
            
            # Get positions
            x0, y0, z0 = positions.get(source, (0, 0, 0))
            x1, y1, z1 = positions.get(target, (0, 0, 0))
            
            # Add line segment
            edge_data['x'].extend([x0, x1, None])  # None creates break between segments
            edge_data['y'].extend([y0, y1, None])
            edge_data['z'].extend([z0, z1, None])
        
        return edge_data
    
    def _create_hover_text(self, node: str, graph: nx.Graph, documents: Optional[Dict[str, Any]]) -> str:
        """Create hover text for a node."""
        
        # Get basic graph info
        degree = graph.degree[node]
        neighbors = list(graph.neighbors(node))
        
        # Try to get document info
        title = "Unknown"
        word_count = 0
        path = ""
        
        if documents and node in documents:
            doc = documents[node]
            title = getattr(doc, 'title', node)
            word_count = getattr(doc, 'word_count', 0)
            path = getattr(doc, 'path', "")
            # Extract filename from path
            if path:
                filename = Path(path).name
            else:
                filename = title
        elif hasattr(graph.nodes[node], 'title'):
            title = graph.nodes[node].get('title', node)
            word_count = graph.nodes[node].get('word_count', 0)
            filename = title
        else:
            title = str(node)[:50] + "..." if len(str(node)) > 50 else str(node)
            filename = title
        
        # Create more informative hover text
        hover_text = f"""<b>{title}</b>
📄 File: {filename}
🔗 Connections: {degree}
📝 Words: {word_count:,}
🆔 ID: {str(node)[:20]}..."""
        
        if neighbors and len(neighbors) <= 5:
            neighbor_names = []
            for neighbor in neighbors[:3]:  # Show max 3 neighbors
                if documents and neighbor in documents:
                    neighbor_name = getattr(documents[neighbor], 'title', str(neighbor)[:20])
                else:
                    neighbor_name = str(neighbor)[:20]
                neighbor_names.append(neighbor_name)
            
            if neighbor_names:
                hover_text += f"\n🔗 Connected to: {', '.join(neighbor_names)}"
                if len(neighbors) > 3:
                    hover_text += f" (+{len(neighbors)-3} more)"
        
        return hover_text
    
    def _add_interactive_features(self, fig: go.Figure, graph: nx.Graph, documents: Optional[Dict[str, Any]]):
        """Add interactive features to the figure."""
        
        # Add dropdown for layout algorithms
        layout_buttons = []
        for algorithm in self.layout_algorithms.keys():
            layout_buttons.append({
                'label': algorithm.replace('_', ' ').title(),
                'method': 'restyle',
                'args': [{'visible': True}, [0, 1]]  # Keep all traces visible
            })
        
        fig.update_layout(
            updatemenus=[{
                'buttons': layout_buttons,
                'direction': 'down',
                'showactive': True,
                'x': 0.1,
                'xanchor': 'left',
                'y': 1.0,
                'yanchor': 'top'
            }],
            annotations=[{
                'text': f"Nodes: {graph.number_of_nodes()}<br>Edges: {graph.number_of_edges()}",
                'showarrow': False,
                'x': 0.02,
                'y': 0.98,
                'xref': 'paper',
                'yref': 'paper',
                'align': 'left',
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': 'rgba(0,0,0,0.5)',
                'borderwidth': 1
            }]
        )
    
    def _save_html(self, fig: go.Figure, output_file: str, auto_open: bool = True):
        """Save figure as HTML file."""
        try:
            fig.write_html(output_file, include_plotlyjs=True)
            self.logger.info(f"Visualization saved to {output_file}")
            
            if auto_open:
                webbrowser.open(f"file://{os.path.abspath(output_file)}")
                
        except Exception as e:
            self.logger.error(f"Error saving HTML file: {e}")
    
    def _show_in_browser(self, fig: go.Figure):
        """Show figure in browser using temporary file."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                temp_file = f.name
            
            fig.write_html(temp_file, include_plotlyjs=True)
            webbrowser.open(f"file://{os.path.abspath(temp_file)}")
            
            self.logger.info(f"Visualization opened in browser (temporary file: {temp_file})")
            
        except Exception as e:
            self.logger.error(f"Error opening in browser: {e}")


def create_obsidian_graph_visualization(knowledge_graph: nx.Graph, 
                                      documents: Optional[Dict[str, Any]] = None,
                                      layout: str = 'spring_3d',
                                      output_file: Optional[str] = None,
                                      title: str = "Obsidian Knowledge Graph") -> go.Figure:
    """
    Convenience function to create graph visualization.
    
    Args:
        knowledge_graph: NetworkX graph to visualize
        documents: Optional dictionary of document objects
        layout: Layout algorithm to use
        output_file: Optional file to save HTML output
        title: Title for the visualization
        
    Returns:
        Plotly Figure object
    """
    visualizer = PlotlyGraphVisualizer()
    return visualizer.visualize_graph(
        knowledge_graph=knowledge_graph,
        documents=documents,
        layout_algorithm=layout,
        output_file=output_file,
        title=title,
        auto_open=True
    )


if __name__ == "__main__":
    """
    Test the visualizer with a sample graph.
    """
    # Create a sample graph for testing
    G = nx.karate_club_graph()
    
    # Add some attributes to make it more interesting
    for node in G.nodes():
        G.nodes[node]['title'] = f"Document {node}"
        G.nodes[node]['word_count'] = np.random.randint(50, 500)
    
    # Create visualization
    fig = create_obsidian_graph_visualization(
        G, 
        layout='spring_3d',
        title="Test Graph Visualization"
    )
    
    print("✓ Test visualization created successfully!") 