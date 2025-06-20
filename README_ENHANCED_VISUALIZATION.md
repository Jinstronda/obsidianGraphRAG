# Enhanced Graph Visualization for GraphRAG

A containerized, enhanced graph visualization system that provides semantic community clustering and rich interactive exploration of your knowledge graph.

## 🌟 Features

### Enhanced Visualizations
- **Semantic Community Clustering**: Colors nodes by AI-generated community themes instead of arbitrary IDs
- **Rich Hover Information**: Document previews, community summaries, key entities, and connection metrics
- **Multiple 3D Layouts**: Community semantic, theme hierarchical, enhanced spring, and community layers
- **2D Overview**: Optimized for web embedding and quick exploration

### Advanced Community Integration
- **AI-Generated Community Summaries**: Uses enhanced GraphRAG community reports
- **Theme-Based Coloring**: Intelligent color schemes based on semantic content
- **Community Separation**: 3D layouts that spatially separate different communities
- **Leiden Clustering**: Advanced clustering algorithm for better community detection

### Safety & Isolation
- **Containerized Design**: Completely isolated from existing working code
- **No File Modifications**: Never modifies your existing GraphRAG system
- **Safe Data Loading**: Read-only access to your graph data
- **Multiple Output Formats**: HTML, interactive 3D, and 2D visualizations

## 🚀 Quick Start

### Basic Usage

```bash
# Run the enhanced visualizer
python launch_enhanced_visualizer.py
```

This will:
1. Load your existing GraphRAG system data
2. Create enhanced 3D visualization with community clustering
3. Generate 2D overview for quick exploration
4. Save HTML files to `images/` directory
5. Auto-open visualizations in your browser

### Example Output
```
🚀 Enhanced Graph Visualizer
========================================
📚 Loading GraphRAG system...
✅ Loaded 5,139 documents
📊 Graph: 5,139 nodes, 15,134 edges
🎨 Creating enhanced visualization...
✅ Enhanced visualization created successfully!
📁 Check the 'images' directory for output files
🎨 Creating 2D overview...
✅ 2D overview created successfully!
🎉 All visualizations completed!
```

## 📊 Visualization Types

### 1. Community Semantic 3D (`community_semantic_3d`)
- **Best for**: Understanding community structure
- **Features**: Separates communities in 3D space, semantic color coding
- **Use Case**: Exploring how different topics cluster in your knowledge base

### 2. Theme Hierarchical 3D (`theme_hierarchical_3d`)
- **Best for**: Hierarchical topic exploration
- **Features**: Vertical layers for different themes
- **Use Case**: Understanding conceptual hierarchies and relationships

### 3. Enhanced Spring 3D (`enhanced_spring_3d`)
- **Best for**: Natural network layout with community hints
- **Features**: Force-directed layout with community-based positioning
- **Use Case**: Balanced view of connections and communities

### 4. 2D Overview
- **Best for**: Quick exploration and web embedding
- **Features**: Optimized performance, simplified hover information
- **Use Case**: Fast browsing and sharing visualizations

## 🎨 Visual Features

### Color Coding
- **Semantic Themes**: Nodes colored by AI-identified community themes
- **HSV Color Space**: Distinct, visually appealing color distribution
- **Community Consistency**: Same-community nodes share colors
- **Gray Fallback**: Uncategorized nodes in neutral gray

### Rich Hover Information
- **Document Details**: Title, content preview, word count, tags
- **Community Info**: Theme, AI-generated summary, key entities
- **Graph Metrics**: Connection count, centrality measures
- **Entity Matches**: How strongly document relates to community

### Interactive Controls
- **3D Navigation**: Zoom, rotate, pan with mouse/touch
- **View Controls**: Show/hide edges, nodes-only view
- **Community Legend**: Color-coded theme legend
- **Responsive Design**: Works on desktop and mobile

## ⚙️ Configuration

### Enabling Enhanced Features

To get full semantic clustering with rich community summaries, you need enhanced GraphRAG features enabled:

1. **Enable Community Summarization**:
   ```bash
   export ENABLE_COMMUNITY_SUMMARIZATION=true
   ```

2. **Enable Advanced Clustering**:
   ```bash
   export USE_LEIDEN_CLUSTERING=true
   ```

3. **Enable Hybrid Search** (optional):
   ```bash
   export ENABLE_HYBRID_SEARCH=true
   ```

### Visualization Configuration

You can customize the visualization by modifying `VisualizationConfig`:

```python
from enhanced_graph_visualizer import VisualizationConfig

config = VisualizationConfig()
config.max_nodes = 3000  # Increase node limit
config.use_semantic_coloring = True  # Enable theme-based colors
config.show_community_info = True  # Rich hover information
config.auto_open_browser = False  # Don't auto-open
```

## 📁 File Structure

```
📁 Project Root
├── enhanced_graph_visualizer.py     # Main visualization engine
├── launch_enhanced_visualizer.py    # Simple launcher script
├── images/                          # Output directory
│   ├── enhanced_graph_*.html        # 3D visualizations
│   ├── enhanced_graph_2d_*.html     # 2D overviews
│   └── community_analysis_*.txt     # Community reports
└── README_ENHANCED_VISUALIZATION.md # This file
```

## 🔧 Advanced Usage

### Programmatic Access

```python
from enhanced_graph_visualizer import create_enhanced_visualization
from graphrag import ObsidianGraphRAG, GraphRAGConfig

# Load your GraphRAG system
config = GraphRAGConfig()
graphrag_system = ObsidianGraphRAG(config)
graphrag_system.initialize_system()

# Create enhanced visualization
fig = create_enhanced_visualization(
    graphrag_system, 
    layout='community_semantic_3d',
    output_file='my_custom_graph.html'
)
```

### Custom Layouts

```python
from enhanced_graph_visualizer import EnhancedGraphVisualizer, VisualizationConfig

visualizer = EnhancedGraphVisualizer()
visualizer.load_graphrag_data(graphrag_system)

# Create multiple visualizations
layouts = ['community_semantic_3d', 'theme_hierarchical_3d', 'enhanced_spring_3d']
for layout in layouts:
    fig = visualizer.create_enhanced_visualization(
        layout_algorithm=layout,
        title=f"Knowledge Graph - {layout}"
    )
```

## 🚨 Troubleshooting

### No Community Summaries Found
**Issue**: "Loaded 0 community summaries"

**Solution**: 
1. Enable community summarization: `export ENABLE_COMMUNITY_SUMMARIZATION=true`
2. Run the main GraphRAG system once to generate summaries
3. Re-run the enhanced visualizer

### Large Graph Performance
**Issue**: Visualization is slow with large graphs

**Solutions**:
- Reduce `max_nodes` in configuration (default: 2000)
- Use 2D overview for quick exploration
- Enable graph optimization features

### Import Errors
**Issue**: "Import error: No module named..."

**Solutions**:
- Install missing dependencies: `pip install plotly networkx numpy`
- Ensure GraphRAG system is properly installed
- Check Python path includes current directory

### Browser Not Opening
**Issue**: Visualizations created but don't open automatically

**Solutions**:
- Manually open HTML files from `images/` directory
- Set `auto_open_browser = False` to disable auto-opening
- Check browser security settings

## 📈 Performance Tips

### Optimization Strategies
1. **Node Limiting**: Default max 2,000 nodes for optimal performance
2. **Edge Filtering**: Automatically filters to most important connections
3. **Community Caching**: Reuses community data between visualizations
4. **Progressive Loading**: Loads data incrementally for better UX

### Large Dataset Handling
- **Automatic Sampling**: Uses centrality-based node selection
- **Memory Management**: Efficient data structures and cleanup
- **Batch Processing**: Processes communities in manageable chunks
- **Fallback Layouts**: Simpler layouts for very large graphs

## 🎯 Use Cases

### Research & Analysis
- **Topic Discovery**: Identify main themes in your knowledge base
- **Concept Mapping**: Visualize relationships between ideas
- **Literature Review**: Explore connections in research notes
- **Knowledge Gaps**: Find isolated or weakly connected areas

### Personal Knowledge Management
- **Note Organization**: Understand how your notes cluster
- **Learning Paths**: Identify natural learning progressions
- **Connection Building**: Discover unexpected relationships
- **Knowledge Health**: Assess completeness and connectivity

### Team Collaboration
- **Shared Understanding**: Visual consensus on topic relationships
- **Onboarding**: Help new team members understand knowledge structure
- **Knowledge Transfer**: Identify expertise areas and dependencies
- **Strategic Planning**: Align on knowledge priorities

## 🔄 Integration with Main System

### Seamless Workflow
1. **Use Main System**: Continue using your regular GraphRAG system
2. **Generate Visualizations**: Run enhanced visualizer anytime
3. **No Interference**: Visualizer never modifies your working system
4. **Data Sync**: Always uses latest data from your GraphRAG system

### When to Use Enhanced Visualizer
- **After Major Updates**: When you've added many new documents
- **For Presentations**: When you need visual knowledge maps
- **During Analysis**: When exploring knowledge structure
- **For Validation**: When checking community detection quality

## 📝 Output Files

### HTML Visualizations
- **3D Interactive**: Full-featured 3D exploration with controls
- **2D Overview**: Fast-loading 2D network visualization
- **Self-Contained**: Include all dependencies for easy sharing
- **Responsive**: Work on desktop, tablet, and mobile devices

### Community Reports
- **Text Analysis**: Detailed community breakdown and statistics
- **Theme Summary**: Key themes and entities per community
- **Relationship Mapping**: How communities connect to each other
- **Growth Tracking**: Changes in community structure over time

## 🎉 Success Indicators

You'll know the enhanced visualizer is working well when you see:

✅ **Rich Community Colors**: Nodes colored by meaningful themes, not random IDs  
✅ **Detailed Hover Info**: Community summaries and entity information in tooltips  
✅ **Semantic Clustering**: Related documents grouped spatially in 3D  
✅ **Performance**: Smooth interaction with large knowledge graphs  
✅ **Visual Clarity**: Clear separation of different topic areas  

## 🤝 Contributing

This enhanced visualization system is designed to be:
- **Extensible**: Easy to add new layout algorithms
- **Configurable**: Customizable for different use cases  
- **Maintainable**: Clean separation from main GraphRAG system
- **Safe**: Never breaks existing functionality

Feel free to customize layouts, colors, and features for your specific needs!

---

**Created by**: AI Assistant  
**Compatibility**: Works with existing GraphRAG systems  
**License**: MIT  
**Status**: Production Ready ✅ 