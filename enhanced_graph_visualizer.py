#!/usr/bin/env python3
"""
Enhanced Graph Visualizer for GraphRAG System
==============================================

A containerized graph visualization system that integrates with the enhanced
GraphRAG community detection and summarization features. This provides:

- Semantic clustering visualization using AI-generated community summaries
- Color coding by community themes rather than arbitrary IDs
- Rich hover information with community details
- Multiple visualization modes (2D HTML, 3D interactive)
- Complete isolation from existing working code

Features:
- Enhanced community data integration
- Theme-based semantic coloring
- Interactive 3D and 2D visualizations
- Community summary tooltips
- Export to multiple formats
- Containerized design for safety

Author: Assistant
License: MIT
"""

import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import webbrowser
import tempfile
import os
from pathlib import Path
import json
import time
import math
import colorsys
from dataclasses import dataclass
from datetime import datetime

# Import GraphRAG components for data access
try:
    from graphrag import ObsidianGraphRAG, GraphRAGConfig
    from enhanced_graphrag import CommunityReport
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False
    logging.warning("GraphRAG components not available for enhanced visualization")


@dataclass
class VisualizationConfig:
    """Configuration for enhanced graph visualization."""
    # Display settings
    max_nodes: int = 2000
    max_edges: int = 5000
    node_size_range: Tuple[int, int] = (8, 30)
    edge_width_range: Tuple[float, float] = (0.5, 3.0)
    
    # Color settings
    use_semantic_coloring: bool = True
    color_by_themes: bool = True
    community_color_opacity: float = 0.8
    
    # Layout settings
    default_layout: str = 'community_semantic_3d'
    layout_iterations: int = 100
    community_separation: float = 2.0
    
    # Information display
    show_community_info: bool = True
    show_document_preview: bool = True
    preview_length: int = 200
    
    # Export settings
    output_directory: str = "images"
    auto_open_browser: bool = True
    save_html: bool = True
    save_interactive: bool = True


class EnhancedGraphVisualizer:
    """
    Enhanced graph visualizer that integrates with GraphRAG community data.
    
    This visualizer provides semantic clustering visualization using AI-generated
    community summaries and themes, offering rich interactive exploration of
    the knowledge graph structure.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize the enhanced graph visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Visualization state
        self.graph = None
        self.documents = None
        self.community_summaries = None
        self.semantic_themes = {}
        self.theme_colors = {}
        
        # Layout algorithms
        self.layout_algorithms = {
            'community_semantic_3d': self._community_semantic_layout_3d,
            'theme_hierarchical_3d': self._theme_hierarchical_layout_3d,
            'enhanced_spring_3d': self._enhanced_spring_layout_3d,
            'community_layers_3d': self._community_layers_layout_3d
        }
        
        self.logger.info("Enhanced Graph Visualizer initialized")
    
    def load_graphrag_data(self, graphrag_system: 'ObsidianGraphRAG') -> bool:
        """
        Load data from GraphRAG system.
        
        Args:
            graphrag_system: Initialized GraphRAG system
            
        Returns:
            True if data loaded successfully
        """
        try:
            self.logger.info("Loading data from GraphRAG system...")
            
            # Load basic graph and documents
            self.graph = graphrag_system.knowledge_graph
            self.documents = graphrag_system.documents
            
            # Load enhanced community data
            if hasattr(graphrag_system, 'community_summaries'):
                self.community_summaries = graphrag_system.community_summaries
                self.logger.info(f"Loaded {len(self.community_summaries)} community summaries")
            else:
                self.logger.warning("No community summaries found in GraphRAG system")
                self.community_summaries = {}
            
            # Extract semantic themes from community data
            self._extract_semantic_themes()
            
            # Generate theme-based colors
            self._generate_theme_colors()
            
            self.logger.info(f"GraphRAG data loaded: {self.graph.number_of_nodes()} nodes, "
                           f"{self.graph.number_of_edges()} edges, "
                           f"{len(self.semantic_themes)} themes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading GraphRAG data: {e}")
            return False
    
    def _extract_semantic_themes(self):
        """Extract semantic themes from community summaries."""
        if not self.community_summaries:
            self.logger.warning("No community summaries available for theme extraction")
            return
        
        try:
            # Extract themes from community reports
            all_themes = []
            community_themes = {}
            
            for community_id, report in self.community_summaries.items():
                if hasattr(report, 'key_themes') and report.key_themes:
                    themes = report.key_themes
                elif hasattr(report, 'title'):
                    # Use title as theme if no themes available
                    themes = [report.title]
                else:
                    themes = [f"Community {community_id}"]
                
                community_themes[community_id] = themes
                all_themes.extend(themes)
            
            # Group similar themes
            unique_themes = list(set(all_themes))
            
            # Create theme mapping
            self.semantic_themes = {}
            for community_id, themes in community_themes.items():
                # Use primary theme for coloring
                primary_theme = themes[0] if themes else "Unknown"
                self.semantic_themes[community_id] = primary_theme
            
            self.logger.info(f"Extracted {len(unique_themes)} unique themes: {unique_themes}")
            
        except Exception as e:
            self.logger.error(f"Error extracting semantic themes: {e}")
            self.semantic_themes = {}
    
    def _generate_theme_colors(self):
        """Generate distinct colors for each theme."""
        if not self.semantic_themes:
            self.logger.warning("No semantic themes available for color generation")
            return
        
        try:
            # Get unique themes
            unique_themes = list(set(self.semantic_themes.values()))
            num_themes = len(unique_themes)
            
            # Generate colors using HSV color space for better distribution
            self.theme_colors = {}
            
            for i, theme in enumerate(unique_themes):
                # Generate hue based on theme index
                hue = i / max(1, num_themes)
                saturation = 0.7  # High saturation for vibrant colors
                value = 0.9  # High value for brightness
                
                # Convert HSV to RGB
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                rgb_255 = tuple(int(c * 255) for c in rgb)
                
                # Convert to hex color
                hex_color = f"#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}"
                
                self.theme_colors[theme] = hex_color
            
            self.logger.info(f"Generated colors for {num_themes} themes")
            
        except Exception as e:
            self.logger.error(f"Error generating theme colors: {e}")
            self.theme_colors = {}
    
    def create_enhanced_visualization(self, 
                                    layout_algorithm: str = None,
                                    title: str = "Enhanced Knowledge Graph",
                                    output_file: Optional[str] = None) -> go.Figure:
        """
        Create enhanced visualization with semantic clustering.
        
        Args:
            layout_algorithm: Layout algorithm to use
            title: Visualization title
            output_file: Optional output file path
            
        Returns:
            Plotly Figure object
        """
        if not self.graph or not self.documents:
            raise ValueError("No graph data loaded. Call load_graphrag_data() first.")
        
        try:
            start_time = time.time()
            layout_algorithm = layout_algorithm or self.config.default_layout
            
            self.logger.info(f"Creating enhanced visualization with {self.graph.number_of_nodes()} nodes")
            
            # Optimize graph for visualization
            optimized_graph = self._optimize_graph_for_visualization()
            
            # Compute enhanced layout
            positions = self._compute_enhanced_layout(optimized_graph, layout_algorithm)
            
            # Prepare enhanced node and edge data
            node_data = self._prepare_enhanced_node_data(optimized_graph, positions)
            edge_data = self._prepare_enhanced_edge_data(optimized_graph, positions)
            
            # Create Plotly figure
            fig = self._create_enhanced_plotly_figure(node_data, edge_data, title)
            
            # Add interactive features
            self._add_enhanced_interactive_features(fig)
            
            # Save or display
            if output_file or self.config.save_html:
                self._save_enhanced_html(fig, output_file)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Enhanced visualization created in {elapsed_time:.2f} seconds")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating enhanced visualization: {e}")
            raise 

    def _optimize_graph_for_visualization(self) -> nx.Graph:
        """Optimize graph for visualization by filtering if too large."""
        if self.graph.number_of_nodes() <= self.config.max_nodes:
            return self.graph.copy()
        
        self.logger.info(f"Optimizing graph from {self.graph.number_of_nodes()} to {self.config.max_nodes} nodes")
        
        # Calculate node importance
        centrality = nx.degree_centrality(self.graph)
        
        # Sort nodes by importance and take top N
        important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        nodes_to_keep = [node for node, _ in important_nodes[:self.config.max_nodes]]
        
        # Create optimized subgraph
        optimized = self.graph.subgraph(nodes_to_keep).copy()
        
        self.logger.info(f"Optimized to {optimized.number_of_nodes()} nodes, {optimized.number_of_edges()} edges")
        return optimized
    
    def _compute_enhanced_layout(self, graph: nx.Graph, algorithm: str) -> Dict[str, Tuple[float, float, float]]:
        """Compute 3D positions using enhanced community-aware algorithms."""
        if algorithm not in self.layout_algorithms:
            self.logger.warning(f"Unknown layout algorithm '{algorithm}', using community_semantic_3d")
            algorithm = 'community_semantic_3d'
        
        return self.layout_algorithms[algorithm](graph)
    
    def _community_semantic_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Community-based layout with semantic theme separation."""
        try:
            self.logger.info("Computing community semantic layout...")
            
            # Get community assignments for nodes
            node_communities = {}
            community_nodes = {}
            
            # Map nodes to communities using document membership
            if self.community_summaries:
                for community_id, report in self.community_summaries.items():
                    if hasattr(report, 'entities'):
                        community_entities = set(report.entities)
                        community_nodes[community_id] = []
                        
                        for node in graph.nodes():
                            # Check if node (document) contains community entities
                            if node in self.documents:
                                doc = self.documents[node]
                                doc_content = doc.content.lower()
                                
                                # Count entity matches
                                matches = sum(1 for entity in community_entities 
                                            if entity.lower() in doc_content)
                                
                                if matches > 0:
                                    node_communities[node] = community_id
                                    community_nodes[community_id].append(node)
            
            # Assign unclustered nodes to default community
            default_community = -1
            community_nodes[default_community] = []
            for node in graph.nodes():
                if node not in node_communities:
                    node_communities[node] = default_community
                    community_nodes[default_community].append(node)
            
            positions = {}
            community_centers = {}
            
            # Position communities in 3D space based on themes
            unique_communities = list(community_nodes.keys())
            n_communities = len(unique_communities)
            
            # Arrange communities in 3D space
            for i, community_id in enumerate(unique_communities):
                # Calculate community center position
                if n_communities == 1:
                    center = (0, 0, 0)
                else:
                    # Arrange in sphere pattern
                    phi = 2 * np.pi * i / n_communities  # Azimuth angle
                    theta = np.pi * (i % 3) / 3  # Elevation angle
                    radius = self.config.community_separation
                    
                    x = radius * np.sin(theta) * np.cos(phi)
                    y = radius * np.sin(theta) * np.sin(phi)
                    z = radius * np.cos(theta)
                    center = (x, y, z)
                
                community_centers[community_id] = center
            
            # Position nodes within each community
            for community_id, nodes in community_nodes.items():
                if not nodes:
                    continue
                
                center_x, center_y, center_z = community_centers[community_id]
                
                if len(nodes) == 1:
                    # Single node at center
                    positions[nodes[0]] = (center_x, center_y, center_z)
                else:
                    # Create subgraph for this community
                    subgraph = graph.subgraph(nodes)
                    
                    # Use spring layout for community internal structure
                    try:
                        sub_positions = nx.spring_layout(
                            subgraph, 
                            iterations=50,
                            k=0.5,
                            dim=2
                        )
                        
                        # Convert to 3D and offset by community center
                        for node, (x, y) in sub_positions.items():
                            positions[node] = (
                                center_x + x * 0.8,  # Scale down internal layout
                                center_y + y * 0.8,
                                center_z + np.random.uniform(-0.2, 0.2)  # Small Z variation
                            )
                            
                    except:
                        # Fallback: circular arrangement
                        for j, node in enumerate(nodes):
                            angle = 2 * np.pi * j / len(nodes)
                            radius = 0.5
                            positions[node] = (
                                center_x + radius * np.cos(angle),
                                center_y + radius * np.sin(angle),
                                center_z
                            )
            
            self.logger.info(f"Community semantic layout: {n_communities} communities positioned")
            return positions
            
        except Exception as e:
            self.logger.error(f"Error in community semantic layout: {e}")
            return self._fallback_spring_layout_3d(graph)
    
    def _theme_hierarchical_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Hierarchical layout organized by semantic themes."""
        try:
            # Group nodes by theme
            theme_nodes = {}
            for node in graph.nodes():
                theme = "Unknown"
                
                # Find theme for this node through community membership
                for community_id, community_theme in self.semantic_themes.items():
                    if self.community_summaries and community_id in self.community_summaries:
                        report = self.community_summaries[community_id]
                        if hasattr(report, 'entities'):
                            # Check if node relates to this community
                            if node in self.documents:
                                doc = self.documents[node]
                                content = doc.content.lower()
                                
                                entity_matches = sum(1 for entity in report.entities 
                                                   if entity.lower() in content)
                                if entity_matches > 0:
                                    theme = community_theme
                                    break
                
                if theme not in theme_nodes:
                    theme_nodes[theme] = []
                theme_nodes[theme].append(node)
            
            positions = {}
            
            # Arrange themes in vertical layers
            themes = list(theme_nodes.keys())
            n_themes = len(themes)
            
            for i, theme in enumerate(themes):
                nodes = theme_nodes[theme]
                
                # Calculate layer height
                z_level = (i - n_themes / 2) * 1.5
                
                # Arrange nodes in this theme layer
                if len(nodes) == 1:
                    positions[nodes[0]] = (0, 0, z_level)
                else:
                    # Circular arrangement for theme
                    for j, node in enumerate(nodes):
                        angle = 2 * np.pi * j / len(nodes)
                        radius = 1.0 + len(nodes) * 0.1  # Radius based on nodes count
                        
                        x = radius * np.cos(angle)
                        y = radius * np.sin(angle)
                        
                        positions[node] = (x, y, z_level)
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error in theme hierarchical layout: {e}")
            return self._fallback_spring_layout_3d(graph)
    
    def _enhanced_spring_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Enhanced spring layout with community gravity."""
        try:
            # Start with standard spring layout
            pos_2d = nx.spring_layout(
                graph, 
                iterations=self.config.layout_iterations,
                k=1.5,
                dim=2
            )
            
            # Convert to 3D with community-based Z positioning
            positions = {}
            
            for node in graph.nodes():
                x, y = pos_2d.get(node, (0, 0))
                
                # Assign Z based on community
                z = 0
                for community_id, community_theme in self.semantic_themes.items():
                    if self.community_summaries and community_id in self.community_summaries:
                        report = self.community_summaries[community_id]
                        if hasattr(report, 'entities') and node in self.documents:
                            doc = self.documents[node]
                            content = doc.content.lower()
                            
                            entity_matches = sum(1 for entity in report.entities 
                                               if entity.lower() in content)
                            if entity_matches > 0:
                                # Hash theme name to get consistent Z level
                                theme_hash = hash(community_theme) % 10
                                z = (theme_hash - 5) * 0.3  # Range -1.5 to 1.5
                                break
                
                positions[node] = (x * 2, y * 2, z)  # Scale up X,Y
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error in enhanced spring layout: {e}")
            return self._fallback_spring_layout_3d(graph)
    
    def _community_layers_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Separate Z-layers for different communities with enhanced spacing."""
        return self._community_semantic_layout_3d(graph)  # Reuse semantic layout
    
    def _fallback_spring_layout_3d(self, graph: nx.Graph) -> Dict[str, Tuple[float, float, float]]:
        """Fallback spring layout if enhanced methods fail."""
        try:
            pos_2d = nx.spring_layout(graph, iterations=50, k=1.0)
            positions = {}
            
            for node in graph.nodes():
                x, y = pos_2d.get(node, (0, 0))
                z = np.random.uniform(-1, 1)
                positions[node] = (x * 2, y * 2, z)
            
            return positions
            
        except:
            # Final fallback: random positions
            positions = {}
            for node in graph.nodes():
                positions[node] = (
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2),
                    np.random.uniform(-1, 1)
                )
            return positions
    
    def _prepare_enhanced_node_data(self, graph: nx.Graph, positions: Dict) -> Dict[str, List]:
        """Prepare enhanced node data with community information."""
        node_data = {
            'x': [], 'y': [], 'z': [],
            'size': [], 'color': [], 'hover_text': [],
            'node_ids': [], 'community_info': []
        }
        
        # Calculate node metrics
        degrees = dict(graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 1
        
        for node in graph.nodes():
            # Position
            x, y, z = positions.get(node, (0, 0, 0))
            node_data['x'].append(x)
            node_data['y'].append(y)
            node_data['z'].append(z)
            node_data['node_ids'].append(node)
            
            # Size based on degree
            degree = degrees[node]
            if max_degree > min_degree:
                size_ratio = (degree - min_degree) / (max_degree - min_degree)
            else:
                size_ratio = 0.5
            
            size = self.config.node_size_range[0] + size_ratio * (
                self.config.node_size_range[1] - self.config.node_size_range[0]
            )
            node_data['size'].append(size)
            
            # Enhanced color based on community theme
            color = self._get_node_color(node)
            node_data['color'].append(color)
            
            # Enhanced hover text
            hover_text = self._create_enhanced_hover_text(node, graph)
            node_data['hover_text'].append(hover_text)
            
            # Community information
            community_info = self._get_node_community_info(node)
            node_data['community_info'].append(community_info)
        
        return node_data
    
    def _get_node_color(self, node: str) -> str:
        """Get semantic color for node based on community theme or connections."""
        try:
            # First priority: Use semantic community themes if available
            for community_id, community_theme in self.semantic_themes.items():
                if self.community_summaries and community_id in self.community_summaries:
                    report = self.community_summaries[community_id]
                    if hasattr(report, 'entities') and node in self.documents:
                        doc = self.documents[node]
                        content = doc.content.lower()
                        
                        # Check entity matches
                        entity_matches = sum(1 for entity in report.entities 
                                           if entity.lower() in content)
                        if entity_matches > 0:
                            return self.theme_colors.get(community_theme, '#1f77b4')
            
            # Fallback: Use connection-based coloring when no community data
            if not self.semantic_themes and self.graph:
                return self._get_connection_based_color(node)
            
            # Default color for unthemed nodes
            return '#888888'  # Gray
            
        except Exception as e:
            self.logger.warning(f"Error getting node color for {node}: {e}")
            return '#1f77b4'  # Default blue
    
    def _get_connection_based_color(self, node: str) -> str:
        """Get color based on node connections when no community data available."""
        try:
            if not self.graph or node not in self.graph:
                return '#1f77b4'  # Default blue
            
            # Get node degree (number of connections)
            degree = self.graph.degree(node)
            
            # Calculate degree percentile for color mapping
            all_degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            max_degree = max(all_degrees) if all_degrees else 1
            min_degree = min(all_degrees) if all_degrees else 0
            
            if max_degree > min_degree:
                degree_ratio = (degree - min_degree) / (max_degree - min_degree)
            else:
                degree_ratio = 0.5
            
            # Map degree to color using HSV color space
            # Hue: 0.0 (red) for high connectivity, 0.7 (blue/purple) for low
            hue = 0.7 - (degree_ratio * 0.7)  # Range: 0.7 to 0.0
            saturation = 0.8  # High saturation for vibrant colors
            value = 0.9  # High brightness
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb_255 = tuple(int(c * 255) for c in rgb)
            
            # Convert to hex color
            hex_color = f"#{rgb_255[0]:02x}{rgb_255[1]:02x}{rgb_255[2]:02x}"
            
            return hex_color
            
        except Exception as e:
            self.logger.warning(f"Error getting connection-based color for {node}: {e}")
            return '#1f77b4'  # Default blue
    
    def _get_node_community_info(self, node: str) -> Dict[str, Any]:
        """Get community information for a node."""
        try:
            for community_id, community_theme in self.semantic_themes.items():
                if self.community_summaries and community_id in self.community_summaries:
                    report = self.community_summaries[community_id]
                    if hasattr(report, 'entities') and node in self.documents:
                        doc = self.documents[node]
                        content = doc.content.lower()
                        
                        entity_matches = sum(1 for entity in report.entities 
                                           if entity.lower() in content)
                        if entity_matches > 0:
                            return {
                                'community_id': community_id,
                                'theme': community_theme,
                                'title': getattr(report, 'title', 'Unknown'),
                                'summary': getattr(report, 'summary', 'No summary available'),
                                'entities': getattr(report, 'entities', []),
                                'entity_matches': entity_matches
                            }
            
            return {'community_id': None, 'theme': 'Uncategorized'}
            
        except Exception as e:
            self.logger.warning(f"Error getting community info for {node}: {e}")
            return {'community_id': None, 'theme': 'Unknown'}

    def _create_enhanced_hover_text(self, node: str, graph: nx.Graph) -> str:
        """Create enhanced hover text for a node."""
        try:
            hover_parts = []
            
            # Document information
            if node in self.documents:
                doc = self.documents[node]
                hover_parts.append(f"<b>{doc.title}</b>")
                
                # Document preview
                content_preview = doc.content[:self.config.preview_length]
                if len(doc.content) > self.config.preview_length:
                    content_preview += "..."
                hover_parts.append(f"<i>Preview:</i><br>{content_preview}")
                
                # Document metadata
                hover_parts.append(f"<i>Word Count:</i> {doc.word_count}")
                if hasattr(doc, 'tags') and doc.tags:
                    tags_str = ", ".join(list(doc.tags)[:5])  # Limit tags
                    hover_parts.append(f"<i>Tags:</i> {tags_str}")
            else:
                hover_parts.append(f"<b>{node}</b>")
            
            # Graph metrics
            degree = graph.degree(node)
            hover_parts.append(f"<i>Connections:</i> {degree}")
            
            # Community information
            community_info = self._get_node_community_info(node)
            if community_info.get('community_id') is not None:
                hover_parts.append(f"<i>Community:</i> {community_info['title']}")
                hover_parts.append(f"<i>Theme:</i> {community_info['theme']}")
                
                if self.config.show_community_info and community_info.get('summary'):
                    summary_preview = community_info['summary'][:150]
                    if len(community_info['summary']) > 150:
                        summary_preview += "..."
                    hover_parts.append(f"<i>Community Summary:</i><br>{summary_preview}")
                
                if community_info.get('entities'):
                    entities_str = ", ".join(community_info['entities'][:3])
                    hover_parts.append(f"<i>Key Entities:</i> {entities_str}")
            
            return "<br><br>".join(hover_parts)
            
        except Exception as e:
            self.logger.warning(f"Error creating enhanced hover text for {node}: {e}")
            return f"<b>{node}</b><br><i>Error loading details</i>"
    
    def _prepare_enhanced_edge_data(self, graph: nx.Graph, positions: Dict) -> Dict[str, List]:
        """Prepare enhanced edge data with weights and community relationships."""
        edge_data = {'x': [], 'y': [], 'z': [], 'weights': [], 'colors': []}
        
        max_edges = self.config.max_edges
        edges_processed = 0
        
        # Get edge weights if available
        edge_weights = {}
        for u, v, data in graph.edges(data=True):
            edge_weights[(u, v)] = data.get('weight', 1.0)
        
        # Sort edges by weight to show most important ones
        sorted_edges = sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)
        
        for (u, v), weight in sorted_edges:
            if edges_processed >= max_edges:
                break
                
            if u in positions and v in positions:
                # Get positions
                x1, y1, z1 = positions[u]
                x2, y2, z2 = positions[v]
                
                # Add edge coordinates (line)
                edge_data['x'].extend([x1, x2, None])
                edge_data['y'].extend([y1, y2, None])
                edge_data['z'].extend([z1, z2, None])
                
                # Edge weight for styling
                edge_data['weights'].append(weight)
                
                # Edge color based on community relationship
                edge_color = self._get_edge_color(u, v)
                edge_data['colors'].append(edge_color)
                
                edges_processed += 1
        
        return edge_data
    
    def _get_edge_color(self, node1: str, node2: str) -> str:
        """Get edge color based on community relationship or connection strength."""
        try:
            # First priority: Use community-based coloring if available
            if self.semantic_themes:
                # Get community info for both nodes
                comm1 = self._get_node_community_info(node1)
                comm2 = self._get_node_community_info(node2)
                
                # Same community = stronger/colored edge
                if (comm1.get('community_id') == comm2.get('community_id') and 
                    comm1.get('community_id') is not None):
                    theme = comm1.get('theme', 'Unknown')
                    base_color = self.theme_colors.get(theme, '#1f77b4')
                    return f"{base_color}CC"  # Add alpha for transparency
                else:
                    # Different communities = neutral edge
                    return 'rgba(150,150,150,0.3)'
            
            # Fallback: Use connection-strength based coloring
            else:
                return self._get_connection_strength_edge_color(node1, node2)
                
        except Exception as e:
            return 'rgba(125,125,125,0.4)'  # Default gray
    
    def _get_connection_strength_edge_color(self, node1: str, node2: str) -> str:
        """Get edge color based on connection strength when no community data."""
        try:
            if not self.graph:
                return 'rgba(100,149,237,0.4)'  # Default blue
            
            # Get degrees of both nodes
            degree1 = self.graph.degree(node1) if node1 in self.graph else 0
            degree2 = self.graph.degree(node2) if node2 in self.graph else 0
            
            # Average degree as connection strength indicator
            avg_degree = (degree1 + degree2) / 2
            
            # Calculate strength percentile
            all_degrees = [self.graph.degree(n) for n in self.graph.nodes()]
            max_degree = max(all_degrees) if all_degrees else 1
            min_degree = min(all_degrees) if all_degrees else 0
            
            if max_degree > min_degree:
                strength_ratio = (avg_degree - min_degree) / (max_degree - min_degree)
            else:
                strength_ratio = 0.5
            
            # Map strength to color and opacity
            # Blue for weaker connections, Orange/Red for stronger
            hue = 0.6 - (strength_ratio * 0.4)  # Range: 0.6 (blue) to 0.2 (orange)
            saturation = 0.7
            value = 0.8
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            rgb_255 = tuple(int(c * 255) for c in rgb)
            
            # Opacity based on strength (stronger connections more visible)
            opacity = 0.2 + (strength_ratio * 0.4)  # Range: 0.2 to 0.6
            
            return f"rgba({rgb_255[0]},{rgb_255[1]},{rgb_255[2]},{opacity:.2f})"
            
        except Exception as e:
            self.logger.warning(f"Error getting edge color: {e}")
            return 'rgba(100,149,237,0.4)'  # Default blue
    
    def _create_enhanced_plotly_figure(self, node_data: Dict, edge_data: Dict, title: str) -> go.Figure:
        """Create enhanced Plotly figure with community visualization."""
        fig = go.Figure()
        
        # Add edges first (behind nodes)
        if edge_data['x']:
            # Check if we have individual edge colors
            if 'colors' in edge_data and edge_data['colors']:
                # Use individual edge colors
                for i in range(0, len(edge_data['x']), 3):  # Every 3 points (x1, x2, None)
                    if i//3 < len(edge_data['colors']):
                        edge_color = edge_data['colors'][i//3]
                        fig.add_trace(go.Scatter3d(
                            x=edge_data['x'][i:i+3],
                            y=edge_data['y'][i:i+3],
                            z=edge_data['z'][i:i+3],
                            mode='lines',
                            line=dict(color=edge_color, width=1.5),
                            hoverinfo='none',
                            showlegend=False,
                            name='Connection'
                        ))
            else:
                # Fallback to single color for all edges
                fig.add_trace(go.Scatter3d(
                    x=edge_data['x'],
                    y=edge_data['y'],
                    z=edge_data['z'],
                    mode='lines',
                    line=dict(
                        color='rgba(125,125,125,0.3)',
                        width=1.5
                    ),
                    hoverinfo='none',
                    showlegend=False,
                    name='Connections'
                ))
        
        # Add nodes with enhanced styling
        fig.add_trace(go.Scatter3d(
            x=node_data['x'],
            y=node_data['y'],
            z=node_data['z'],
            mode='markers',
            marker=dict(
                size=node_data['size'],
                color=node_data['color'],
                opacity=self.config.community_color_opacity,
                line=dict(width=1, color='rgba(50,50,50,0.8)'),
                sizemode='diameter'
            ),
            text=node_data['hover_text'],
            hoverinfo='text',
            hovertemplate='%{text}<extra></extra>',
            showlegend=False,
            name='Documents'
        ))
        
        # Create community legend
        if self.theme_colors:
            self._add_community_legend(fig)
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color='darkblue')
            ),
            scene=dict(
                xaxis=dict(
                    showticklabels=False,
                    showgrid=True,
                    zeroline=False,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=True,
                    zeroline=False,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                zaxis=dict(
                    showticklabels=False,
                    showgrid=True,
                    zeroline=False,
                    gridcolor='rgba(200,200,200,0.3)'
                ),
                bgcolor='rgba(240,240,240,0.1)',
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.8),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            height=800,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _add_community_legend(self, fig: go.Figure):
        """Add community theme legend to the figure."""
        try:
            # Add invisible traces for legend
            for theme, color in self.theme_colors.items():
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=theme,
                    showlegend=True,
                    legendgroup='communities'
                ))
        except Exception as e:
            self.logger.warning(f"Error adding community legend: {e}")
    
    def _add_enhanced_interactive_features(self, fig: go.Figure):
        """Add enhanced interactive features to the figure."""
        try:
            # Add annotations for community information
            if self.community_summaries:
                annotations = []
                
                # Add title annotation
                annotations.append(dict(
                    text=f"Communities: {len(self.semantic_themes)} | "
                         f"Themes: {len(set(self.semantic_themes.values()))} | "
                         f"Enhanced GraphRAG Visualization",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=0.02,
                    xanchor='center', yanchor='bottom',
                    font=dict(size=12, color='gray')
                ))
                
                fig.update_layout(annotations=annotations)
            
            # Add control buttons for different views
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=list([
                            dict(
                                args=[{"visible": [True, True]}],
                                label="Show All",
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [False, True]}],
                                label="Nodes Only",
                                method="restyle"
                            ),
                            dict(
                                args=[{"visible": [True, False]}],
                                label="Edges Only",
                                method="restyle"
                            )
                        ]),
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.02,
                        yanchor="top"
                    ),
                ]
            )
            
        except Exception as e:
            self.logger.warning(f"Error adding interactive features: {e}")
    
    def _save_enhanced_html(self, fig: go.Figure, output_file: Optional[str] = None):
        """Save enhanced visualization to HTML file."""
        try:
            # Create output directory
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"enhanced_graph_visualization_{timestamp}.html"
            else:
                output_file = output_dir / output_file
            
            # Save HTML
            fig.write_html(
                str(output_file),
                include_plotlyjs=True,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
                }
            )
            
            self.logger.info(f"Enhanced visualization saved to: {output_file}")
            
            # Auto-open in browser
            if self.config.auto_open_browser:
                webbrowser.open(f"file://{output_file.absolute()}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving enhanced visualization: {e}")
            return None
    
    def create_2d_overview(self, title: str = "Enhanced Graph Overview (2D)") -> go.Figure:
        """Create a 2D overview visualization for web embedding."""
        if not self.graph or not self.documents:
            raise ValueError("No graph data loaded. Call load_graphrag_data() first.")
        
        try:
            # Use 2D spring layout
            pos_2d = nx.spring_layout(self.graph, iterations=100, k=1.5)
            
            # Prepare 2D data
            node_data_2d = {'x': [], 'y': [], 'size': [], 'color': [], 'hover_text': []}
            edge_data_2d = {'x': [], 'y': []}
            
            # Calculate node metrics
            degrees = dict(self.graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            min_degree = min(degrees.values()) if degrees else 1
            
            # Nodes
            for node in self.graph.nodes():
                x, y = pos_2d.get(node, (0, 0))
                node_data_2d['x'].append(x)
                node_data_2d['y'].append(y)
                
                # Size and color
                degree = degrees[node]
                size_ratio = (degree - min_degree) / max(1, max_degree - min_degree)
                size = 8 + size_ratio * 20
                node_data_2d['size'].append(size)
                
                color = self._get_node_color(node)
                node_data_2d['color'].append(color)
                
                # Simplified hover text for 2D
                hover_text = self._create_simple_hover_text(node)
                node_data_2d['hover_text'].append(hover_text)
            
            # Edges
            for u, v in list(self.graph.edges())[:1000]:  # Limit edges for performance
                if u in pos_2d and v in pos_2d:
                    x1, y1 = pos_2d[u]
                    x2, y2 = pos_2d[v]
                    edge_data_2d['x'].extend([x1, x2, None])
                    edge_data_2d['y'].extend([y1, y2, None])
            
            # Create 2D figure
            fig = go.Figure()
            
            # Add edges
            if edge_data_2d['x']:
                fig.add_trace(go.Scatter(
                    x=edge_data_2d['x'],
                    y=edge_data_2d['y'],
                    mode='lines',
                    line=dict(color='rgba(150,150,150,0.3)', width=1),
                    hoverinfo='none',
                    showlegend=False
                ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_data_2d['x'],
                y=node_data_2d['y'],
                mode='markers',
                marker=dict(
                    size=node_data_2d['size'],
                    color=node_data_2d['color'],
                    opacity=0.8,
                    line=dict(width=1, color='rgba(50,50,50,0.5)')
                ),
                text=node_data_2d['hover_text'],
                hoverinfo='text',
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ))
            
            # Layout
            fig.update_layout(
                title=dict(text=title, x=0.5),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=f"Enhanced GraphRAG: {self.graph.number_of_nodes()} documents, "
                             f"{len(self.semantic_themes)} communities",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=12, color='gray')
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating 2D overview: {e}")
            raise
    
    def _create_simple_hover_text(self, node: str) -> str:
        """Create simplified hover text for 2D visualization."""
        try:
            if node in self.documents:
                doc = self.documents[node]
                community_info = self._get_node_community_info(node)
                
                parts = [
                    f"<b>{doc.title}</b>",
                    f"Theme: {community_info.get('theme', 'Unknown')}",
                    f"Connections: {self.graph.degree(node)}"
                ]
                
                return "<br>".join(parts)
            else:
                return f"<b>{node}</b>"
                
        except Exception as e:
            return f"<b>{node}</b>"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_enhanced_visualization(graphrag_system: 'ObsidianGraphRAG',
                                layout: str = 'community_semantic_3d',
                                output_file: Optional[str] = None,
                                config: VisualizationConfig = None) -> go.Figure:
    """
    Convenience function to create enhanced graph visualization.
    
    Args:
        graphrag_system: Initialized GraphRAG system
        layout: Layout algorithm to use
        output_file: Optional output file path
        config: Visualization configuration
        
    Returns:
        Plotly Figure object
    """
    visualizer = EnhancedGraphVisualizer(config)
    
    if not visualizer.load_graphrag_data(graphrag_system):
        raise ValueError("Failed to load GraphRAG data")
    
    return visualizer.create_enhanced_visualization(
        layout_algorithm=layout,
        title="Enhanced Knowledge Graph Visualization",
        output_file=output_file
    )


def create_enhanced_2d_overview(graphrag_system: 'ObsidianGraphRAG',
                              config: VisualizationConfig = None) -> go.Figure:
    """
    Convenience function to create 2D overview visualization.
    
    Args:
        graphrag_system: Initialized GraphRAG system
        config: Visualization configuration
        
    Returns:
        Plotly Figure object
    """
    visualizer = EnhancedGraphVisualizer(config)
    
    if not visualizer.load_graphrag_data(graphrag_system):
        raise ValueError("Failed to load GraphRAG data")
    
    return visualizer.create_2d_overview()


if __name__ == "__main__":
    print("Enhanced Graph Visualizer for GraphRAG")
    print("=" * 50)
    print("This module provides enhanced graph visualization")
    print("with semantic community clustering and rich hover information.")
    print("\nUsage:")
    print("  from enhanced_graph_visualizer import create_enhanced_visualization")
    print("  fig = create_enhanced_visualization(graphrag_system)")
    print("\nNote: This module requires an initialized GraphRAG system.") 