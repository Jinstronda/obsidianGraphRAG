# 🌐 Obsidian Graph RAG 3D Visualizer

An advanced 3D visualization system for exploring your Obsidian knowledge graph in immersive three-dimensional space.

## ✨ Features

### 🎮 Interactive 3D Navigation
- **Mouse Controls**: Rotate, zoom, and pan through your knowledge graph
- **Orbit Controls**: Smooth camera movement with momentum and damping
- **Auto-Focus**: Click any node to automatically center the view
- **Reset Camera**: Instantly return to optimal viewing position

### 🔮 Multiple Layout Algorithms
1. **Force-Directed 3D** - Physics-based layout with natural clustering
2. **Fruchterman-Reingold 3D** - Classic graph layout adapted for 3D space
3. **Spherical 3D** - Nodes arranged on sphere surface with community grouping
4. **Hierarchical 3D** - Layered arrangement based on node importance
5. **Community Layers 3D** - Different communities on separate Z-levels

### 🎨 Visual Features
- **Community Colors** - Automatic color coding based on detected communities
- **Size Scaling** - Node sizes reflect importance (centrality, connections)
- **Edge Visualization** - Customizable connection rendering
- **Real-time Labels** - Toggle node labels on/off
- **Atmospheric Effects** - Fog and lighting for depth perception

### 🔍 Search & Discovery
- **Live Search** - Find nodes by title, content, or tags
- **Instant Highlighting** - Search results highlighted in 3D space
- **Auto-Navigation** - Click search results to fly to nodes
- **Relevance Scoring** - Smart ranking of search results

### 📊 Analytics Dashboard
- **Graph Statistics** - Nodes, edges, density, communities
- **Centrality Metrics** - Betweenness, closeness, degree centrality
- **Community Analysis** - Automatic community detection and visualization
- **Real-time Updates** - Statistics update as you explore

## 🚀 Getting Started

### Prerequisites
```bash
pip install fastapi uvicorn websockets scipy
```

### Launch 3D Visualizer
```python
from graph3d_launcher import Graph3DLauncher
from graphrag import GraphRAGConfig

# Configure for your vault
config = GraphRAGConfig()
config.vault_path = "path/to/your/obsidian/vault"

# Initialize 3D launcher
launcher = Graph3DLauncher(config)
launcher.initialize_system()

# Start 3D visualizer
launcher.start_3d_visualizer()
```

### Command Line Usage
```bash
python graph3d_launcher.py
# Choose option 1 for 3D Visualizer
```

## 🎛️ Controls Guide

### 🖱️ Mouse Controls
- **Left Click + Drag**: Rotate camera around the scene
- **Right Click + Drag**: Pan/translate view
- **Scroll Wheel**: Zoom in/out
- **Click Node**: Select and focus on specific note
- **Double Click**: Center camera on selected node

### ⌨️ Keyboard Shortcuts
- **Space**: Reset camera to default position
- **L**: Toggle labels on/off
- **E**: Toggle edge/connection visibility
- **S**: Focus on search input
- **Escape**: Clear selection

### 🎮 UI Controls
- **Layout Dropdown**: Switch between 3D algorithms
- **Apply Layout**: Recompute positions with selected algorithm
- **Search Box**: Find specific notes in your vault
- **View Options**: Toggle labels, edges, and effects
- **Statistics Panel**: View graph metrics and analytics

## ⚙️ Layout Algorithms

### 🌊 Force-Directed 3D
**Best for**: Natural clustering and organic layouts
- Uses physics simulation with attraction/repulsion forces
- Nodes repel each other while connected nodes attract
- Creates natural communities through emergent behavior
- **Parameters**: Iterations, space size, force strengths

### 🎯 Fruchterman-Reingold 3D
**Best for**: Balanced, aesthetic layouts
- Classic graph layout algorithm adapted for 3D
- Sophisticated cooling schedule for stability
- Good balance between clustering and readability
- **Parameters**: Iterations, space size, cooling rate

### 🌍 Spherical 3D
**Best for**: Community visualization and exploration
- Arranges nodes on sphere surface
- Communities clustered together
- Excellent for understanding vault structure
- **Parameters**: Sphere radius, community clustering

### 🏗️ Hierarchical 3D
**Best for**: Importance-based visualization
- Layers nodes by centrality/importance
- Most important nodes at center
- Clear hierarchical structure
- **Parameters**: Layer height, layer radius

### 📚 Community Layers 3D
**Best for**: Topic separation and organization
- Each community gets its own Z-layer
- Clear topic separation
- Force-directed layout within layers
- **Parameters**: Layer height, internal layout

## 🔧 Advanced Features

### 🌐 Web API Endpoints
- `GET /api/graph/data` - Full graph data with 3D positions
- `GET /api/graph/layouts` - Available layout algorithms
- `POST /api/graph/layout/{algorithm}` - Compute new layout
- `GET /api/search?q={query}` - Search nodes
- `GET /api/node/{id}` - Node details
- `WebSocket /ws/graph` - Real-time updates

### 🎨 Customization Options
```python
# Custom colors and styling
launcher = Graph3DLauncher(config)
launcher.initialize_system()
launcher.start_3d_visualizer(
    host="0.0.0.0",  # Allow external connections
    port=3000,       # Custom port
    open_browser=False  # Don't auto-open browser
)
```

### 📡 Real-time Updates
- WebSocket connections for live updates
- Automatic layout recomputation
- Synchronized multi-user viewing
- Real-time search result highlighting

## 🎯 Use Cases

### 📖 Knowledge Exploration
- **Vault Overview**: Get a bird's-eye view of your entire knowledge base
- **Topic Discovery**: Find unexpected connections between concepts
- **Community Analysis**: Understand how your notes cluster into topics
- **Orphan Detection**: Identify isolated notes that need more connections

### 🔍 Research & Analysis
- **Concept Mapping**: Visualize relationships between ideas
- **Literature Reviews**: See how sources and citations connect
- **Project Planning**: Understand knowledge dependencies
- **Learning Paths**: Discover optimal sequences through topics

### 🧠 Creative Inspiration
- **Serendipitous Discovery**: Find unexpected note combinations
- **Pattern Recognition**: Spot recurring themes across your vault
- **Connection Building**: Identify opportunities for new links
- **Knowledge Gaps**: See areas that need more development

## 🛠️ Technical Architecture

### 🏗️ Backend Components
- **Graph3DVisualizer**: Main controller for 3D layouts
- **Layout Algorithms**: Physics-based positioning systems
- **Web Server**: FastAPI server with real-time capabilities
- **Data Exporters**: JSON serialization for frontend

### 🌐 Frontend Technology
- **Three.js R158+**: Latest 3D rendering engine
- **WebGL 2.0**: Hardware-accelerated graphics
- **Modern JavaScript**: ES2022+ features
- **Responsive Design**: Works on desktop, tablet, mobile

### ⚡ Performance Optimizations
- **Level of Detail (LOD)**: Reduce detail for distant objects
- **Frustum Culling**: Only render visible objects
- **Instanced Rendering**: Efficient batch rendering
- **Layout Caching**: Cache computed positions
- **Chunked Loading**: Progressive loading for large graphs

## 🎨 Styling & Themes

### 🌙 Dark Theme (Default)
- Dark space background with gradient effects
- Neon-style node highlighting
- Subtle connection lines
- High contrast for readability

### 🎨 Color Schemes
- **Community Colors**: 15 distinct colors for communities
- **Importance Scaling**: Size based on centrality metrics
- **Edge Styling**: Thickness reflects relationship strength
- **Selection Effects**: Glow and highlight for active nodes

## 🔍 Troubleshooting

### 🚨 Common Issues

**Server won't start**
```bash
# Check if port is in use
netstat -an | grep :8080
# Try different port - use option 4 in the launcher for custom settings
python graph3d_launcher.py
```

**3D view is empty**
```python
# Ensure graph is built
launcher = Graph3DLauncher(config)
launcher.initialize_system()
print(f"Nodes: {launcher.graph_rag.knowledge_graph.number_of_nodes()}")
```

**Layout computation is slow**
```python
# Reduce iterations for large graphs
python graph3d_launcher.py
# Use "Spherical 3D" for fastest layout
```

**Browser compatibility**
- **Chrome/Edge**: Full WebGL 2.0 support ✅
- **Firefox**: Full support ✅
- **Safari**: WebGL 1.0 (some features limited) ⚠️
- **Mobile**: Basic support 📱

### 📊 Performance Guidelines
- **Small graphs** (< 100 nodes): All algorithms work well
- **Medium graphs** (100-1000 nodes): Use Force-Directed or Spherical
- **Large graphs** (1000+ nodes): Use Spherical or Hierarchical
- **Very large graphs** (5000+ nodes): Consider filtering or sampling

## 🎓 Tips & Best Practices

### 🎯 Optimal Viewing
1. **Start with Spherical 3D** for overview
2. **Switch to Force-Directed** for detailed exploration
3. **Use search** to find specific topics
4. **Toggle labels** based on zoom level
5. **Try different layouts** to reveal different patterns

### 🔍 Effective Exploration
1. **Follow connections** between related notes
2. **Look for color clusters** representing communities
3. **Focus on large nodes** for important concepts
4. **Search for keywords** to find relevant areas
5. **Use camera reset** when you get lost

### 🎨 Visual Optimization
1. **Adjust zoom level** for optimal label readability
2. **Toggle edges** to reduce visual clutter
3. **Use search highlighting** to track topics
4. **Try different layouts** for various perspectives
5. **Focus on specific communities** for detailed study

## 🌟 Future Enhancements

### 🔮 Planned Features
- **VR/AR Support** - WebXR integration for immersive exploration
- **Collaborative Viewing** - Multi-user shared exploration
- **Animation System** - Smooth transitions between layouts
- **Export Options** - Save 3D models and screenshots
- **Plugin Architecture** - Custom visualizations and interactions

### 🎮 Advanced Interactions
- **Node Editing** - Direct manipulation of note content
- **Link Creation** - Visual creation of new connections
- **Timeline View** - See graph evolution over time
- **Filtering System** - Show/hide nodes by criteria
- **Annotation Tools** - Add notes and highlights to 3D view

---

*Happy exploring! 🚀 Your knowledge graph awaits in stunning 3D.* 