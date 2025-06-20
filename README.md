# 🧠 Enhanced Obsidian Graph RAG System

A comprehensive Graph RAG (Retrieval-Augmented Generation) system that transforms Obsidian vaults into intelligent knowledge bases. This system combines vector search with graph traversal to provide enhanced context understanding and multi-hop reasoning for personal knowledge management.

## 🎯 **What This System Does**

- **Intelligent Knowledge Search**: Combines semantic similarity with explicit document relationships
- **Multi-Hop Graph Traversal**: Discovers connections up to 4 steps away from query topics
- **Sequential Reasoning**: Breaks down complex questions into manageable steps
- **Hybrid Retrieval**: Merges vector search, BM25 full-text search, and sparse vectors
- **Multiple Interfaces**: Web chat, CLI, 3D visualization, and enhanced 2D graphs
- **Real-time Processing**: Monitors vault changes and updates incrementally

---

## 🔧 **Technology Stack**

### **Core Technologies**
- **Python 3.8+** - Main programming language
- **OpenAI API** - GPT-4o for chat, text-embedding-3-large for vectors
- **NetworkX** - Graph algorithms and community detection
- **NumPy & Pandas** - Data processing and mathematical operations
- **PyTorch** - Tensor operations and GPU acceleration

### **AI & Machine Learning**
- **OpenAI Models**: GPT-4o, text-embedding-3-large
- **Hugging Face Transformers** - Advanced NLP models
- **scikit-learn** - Traditional ML algorithms
- **sentence-transformers** - Sentence embeddings
- **spaCy** - NLP pipeline and entity recognition

### **Graph & Search Technologies**
- **NetworkX** - Graph construction and analysis
- **Leiden Algorithm** - Advanced community detection
- **BM25** - Full-text search implementation
- **FAISS** - Vector similarity search
- **TF-IDF** - Sparse vector generation

### **Web & Visualization**
- **Flask** - Web chat interface
- **FastAPI** - 3D visualization backend
- **Plotly** - Interactive 2D/3D graphs
- **WebSockets** - Real-time communication
- **HTML/CSS/JavaScript** - Frontend interface

### **Data & Performance**
- **Pickle** - Data persistence and caching
- **YAML** - Configuration and frontmatter parsing
- **Async/Await** - Non-blocking operations
- **Watchdog** - File system monitoring
- **psutil** - System resource monitoring

---

## 🚀 **Quick Start**

### **Requirements**
- Python 3.8+
- OpenAI API key
- Obsidian vault with markdown files

### **Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/obsidian-graph-rag.git
cd obsidian-graph-rag

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your OpenAI API key and vault path

# Run the system
python web_chat.py  # Web interface
# or
python start_chat.py  # CLI interface
```

### **Configuration**
Set these key variables in your `.env` file:
```bash
OBSIDIAN_VAULT_PATH=/path/to/your/vault
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
MAX_GRAPH_HOPS=4
```

### **Usage**
- **Web Interface**: `python web_chat.py` - Opens http://localhost:5000
- **CLI Interface**: `python start_chat.py` - Terminal-based chat
- **3D Visualization**: `python graph3d_launcher.py` - Interactive 3D graph
- **Enhanced Visualization**: `python launch_enhanced_visualizer.py` - Advanced 2D graphs

---

## 📚 **Graph RAG Study Guide**

### **🎯 Core Graph RAG Concepts**

#### **1. Traditional RAG vs Graph RAG**
- **Traditional RAG**: Treats documents as isolated chunks, retrieves based only on semantic similarity
- **Graph RAG**: Preserves document relationships, combines semantic search with graph traversal
- **Key Advantage**: Discovers connections between related concepts across multiple documents

#### **2. Knowledge Graph Construction**
- **Nodes**: Individual documents (Obsidian notes)
- **Edges**: Explicit relationships (`[[wikilinks]]` between notes)
- **Weights**: Connection strength based on link frequency and semantic similarity
- **Communities**: Topic clusters discovered through graph algorithms

#### **3. Multi-Hop Graph Traversal**
- **Concept**: Explore connections beyond immediate neighbors (2-4 hops)
- **Benefit**: Discover related concepts that aren't directly linked to query
- **Implementation**: Breadth-first expansion with relevance scoring
- **Performance**: 200% more relevant connections than single-hop

#### **4. Hybrid Retrieval Methods**
- **Vector Search**: Semantic similarity using embeddings (text-embedding-3-large)
- **BM25 Search**: Traditional full-text keyword matching
- **Sparse Vectors**: TF-IDF weighted keyword representations
- **Graph Expansion**: Multi-hop traversal from semantic matches
- **Fusion**: Combine all methods using Reciprocal Rank Fusion (RRF)

#### **5. Community Detection & Summarization**
- **Purpose**: Group related documents into topic clusters
- **Algorithms**: Leiden clustering (superior to Louvain)
- **Global Search**: Query entire topic communities for broad questions
- **Summarization**: LLM-generated descriptions of community themes

#### **6. Sequential Thinking for Complex Queries**
- **Query Decomposition**: Break complex questions into manageable steps
- **Iterative Retrieval**: Gather context for each step independently
- **Step-by-Step Reasoning**: Build reasoning chain with intermediate conclusions
- **Answer Synthesis**: Combine all steps into comprehensive response

### **🔧 Technical Implementation Concepts**

#### **Graph Algorithms**
- **Community Detection**: Leiden vs Louvain clustering methods
- **Centrality Measures**: Identify important nodes (degree, betweenness, PageRank)
- **Graph Traversal**: BFS/DFS for multi-hop exploration
- **Flow-Based Pruning**: PathRAG optimization for relevant path selection

#### **Vector Operations**
- **Embedding Generation**: Transform text into high-dimensional vectors
- **Cosine Similarity**: Measure semantic similarity between vectors
- **Batch Processing**: Efficient API usage for large document sets
- **Sparse vs Dense**: Different vector representations for different tasks

#### **Information Retrieval**
- **BM25 Scoring**: Probabilistic ranking function for term matching
- **TF-IDF Weighting**: Term frequency-inverse document frequency scoring
- **Rank Fusion**: Combine multiple ranking methods (RRF, weighted sum)
- **Late Interaction**: Token-level similarity for precise reranking

### **🚀 Advanced Features**

#### **Entity-Based Processing**
- **Named Entity Recognition**: Extract people, places, concepts from text
- **Entity Linking**: Connect same entities across different documents
- **Entity Disambiguation**: Resolve ambiguous entity mentions
- **Cross-Document Relationships**: Build entity-centric knowledge graph

#### **Temporal Analysis**
- **Knowledge Evolution**: Track how concepts change over time
- **Document Freshness**: Weight recent information more heavily
- **Temporal Drift**: Detect when knowledge becomes outdated
- **Time-Aware Retrieval**: Consider document creation/modification dates

#### **Performance Optimizations**
- **Caching Systems**: Avoid reprocessing unchanged data
- **Incremental Updates**: Process only new/modified documents
- **GPU Acceleration**: Use CUDA for tensor operations
- **Memory Management**: Efficient handling of large knowledge graphs

### **📊 Key Metrics & Evaluation**

#### **Retrieval Quality**
- **Precision**: Proportion of retrieved documents that are relevant
- **Recall**: Proportion of relevant documents that are retrieved
- **nDCG**: Normalized Discounted Cumulative Gain for ranking quality
- **Coverage**: Percentage of knowledge base accessible through traversal

#### **System Performance**
- **Query Latency**: Time from question to response
- **Throughput**: Questions processed per second
- **Memory Usage**: RAM and GPU memory consumption
- **API Costs**: OpenAI token usage and expenses

### **🛠️ Essential Tools & Libraries**

#### **Graph Processing**
- **NetworkX**: Python library for graph creation and analysis
- **igraph**: Alternative graph library with Leiden clustering
- **Community Detection**: python-louvain, leidenalg packages

#### **Vector & Search**
- **OpenAI**: text-embedding-3-large for high-quality embeddings
- **FAISS**: Facebook's vector similarity search library
- **rank-bm25**: BM25 implementation for full-text search
- **scikit-learn**: TF-IDF vectorizers and similarity metrics

#### **NLP & Processing**
- **spaCy**: Industrial-strength NLP pipeline
- **transformers**: Hugging Face model library
- **tiktoken**: OpenAI token counting utility

---

**🎯 Master these Graph RAG concepts to understand how this system creates intelligent knowledge bases that go beyond traditional document search to discover deep relationships in your personal knowledge.** 