# 🧠 Obsidian Graph RAG AI Librarian

**Transform your Obsidian vault into an intelligent, conversational knowledge discovery system.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-4](https://img.shields.io/badge/AI-OpenAI%20GPT--4-green.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Graph RAG](https://img.shields.io/badge/Architecture-Graph%20RAG-purple.svg)](#architecture)

> **Beyond simple Q&A**: A sophisticated knowledge exploration system that understands the connections between your ideas through advanced Graph RAG technology.

---

## 🎯 What Makes This Special

Unlike traditional RAG systems that rely solely on semantic similarity, our **Graph RAG** approach combines:

- 🔍 **Semantic Vector Search** - Find conceptually related content
- 🕸️ **Knowledge Graph Traversal** - Follow explicit connections through wiki-links
- 🧠 **Sequential Thinking** - Multi-step reasoning for complex queries  
- ⚡ **Real-time Processing** - Sub-second responses from your entire vault
- 🎨 **Beautiful Interfaces** - Modern web UI, CLI, and 3D graph visualization

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [💾 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [🎮 Usage](#-usage)
- [🏗️ Architecture](#️-architecture)
- [📖 API Reference](#-api-reference)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## ✨ Features

### 🧠 **Intelligent Knowledge Discovery**
- **Graph RAG Technology**: Combines semantic search with knowledge graph traversal
- **Sequential Thinking**: Multi-step reasoning for complex, nuanced queries
- **Context-Aware Responses**: Understands relationships between your notes
- **Source Citations**: Every answer includes relevant note references

### 🎨 **Multiple Interfaces**
- **🌐 Modern Web Interface**: Beautiful, responsive design with dark/light themes
- **💻 Command Line Interface**: Fast, terminal-based interaction
- **📊 3D Graph Visualization**: Explore your knowledge network interactively
- **🔍 System Diagnostics**: Built-in health monitoring and troubleshooting

### ⚡ **Performance & Scalability**
- **Lightning Fast**: Process 5,000+ documents in minutes, query in milliseconds
- **Incremental Processing**: Only processes changed files, not entire vault
- **Smart Caching**: Persistent storage with automatic cache invalidation
- **Memory Efficient**: Optimized for large knowledge bases

### 🔧 **Developer-Friendly**
- **Modular Architecture**: Clean, extensible codebase
- **Comprehensive Logging**: Detailed system monitoring and debugging
- **Environment Configuration**: Flexible setup via `.env` files
- **Cross-Platform**: Works on Windows, macOS, and Linux

### 🛡️ **Enterprise-Ready**
- **Robust Error Handling**: Graceful failure recovery and user feedback
- **Security Focused**: Local processing with secure API key management
- **Production Ready**: Comprehensive testing and validation systems
- **Scalable Design**: Built to handle growing knowledge bases

## 🚀 Quick Start

Get your AI librarian running in 3 simple steps:

### 1. Clone & Install
```bash
git clone https://github.com/your-username/obsidian-graph-rag.git
cd obsidian-graph-rag
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file:
```bash
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here

# Required: Path to your Obsidian vault
OBSIDIAN_VAULT_PATH=C:\Users\YourName\Documents\Obsidian Vault\Your Vault

# Optional: Customize settings
WEB_PORT=5000
LOG_LEVEL=INFO
```

### 3. Launch Your AI Librarian
```bash
python start_chat.py
```

Choose between the **beautiful web interface** or **command line chat**, and start exploring your knowledge!

## 💾 Installation

### Prerequisites
- **Python 3.8+** (3.10+ recommended)
- **OpenAI API Key** ([Get one here](https://platform.openai.com/api-keys))
- **Obsidian Vault** with markdown files

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/obsidian-graph-rag.git
   cd obsidian-graph-rag
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```bash
   python check_system.py
   ```

### Development Installation

For contributors and developers:

```bash
# Install additional development dependencies
pip install -r requirements_viz.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## ⚙️ Configuration

### Environment Variables (.env file)

Create a `.env` file in the project root with your settings:

```bash
# === REQUIRED SETTINGS ===

# OpenAI API Key - Get from https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here

# Path to your Obsidian vault
OBSIDIAN_VAULT_PATH=/path/to/your/obsidian/vault

# === OPTIONAL SETTINGS ===

# AI Model Configuration
OPENAI_MODEL=gpt-4o                    # Default: gpt-4o
EMBEDDING_MODEL=text-embedding-3-small  # Default: text-embedding-3-small
MAX_TOKENS=2000                        # Response length limit
TEMPERATURE=0.1                        # Creativity level (0.0-1.0)

# Performance Settings
CHUNK_SIZE=500                         # Text chunk size for processing
TOP_K_VECTOR=10                        # Number of semantic search results
TOP_K_GRAPH=5                          # Number of graph traversal results

# Web Interface
WEB_HOST=127.0.0.1                     # Web server host
WEB_PORT=5000                          # Web server port
AUTO_OPEN_BROWSER=true                 # Auto-open browser on start

# System Settings
CACHE_DIR=./cache                      # Cache directory location
LOG_LEVEL=INFO                         # Logging level (DEBUG, INFO, WARNING, ERROR)
```

### Vault Requirements

Your Obsidian vault should contain:
- **Markdown files** (`.md` extension)
- **Wiki-links** (`[[Note Name]]`) for optimal graph connections
- **Tags** (`#tag`) for enhanced categorization
- **YAML frontmatter** (optional) for metadata

## 🎮 Usage

### 🌐 Web Interface (Recommended)

The modern web interface provides the best user experience:

```bash
python start_chat.py
# Choose option 1 for Web Interface
```

**Features:**
- 🎨 Beautiful, modern design with dark/light theme toggle
- 💬 Real-time chat with typing indicators
- 📚 Source citations with expandable note previews
- 📱 Responsive design for desktop and mobile
- ⌨️ Keyboard shortcuts (Enter to send, Shift+Enter for new line)

### 💻 Command Line Interface

For terminal enthusiasts and automation:

```bash
python start_chat.py
# Choose option 2 for CLI Interface
```

**Perfect for:**
- SSH sessions and remote servers
- Automated workflows and scripting
- Low-resource environments
- Terminal-based development

### 📊 3D Graph Visualization

Explore your knowledge network in three dimensions:

```bash
python graph3d_launcher.py
```

**Capabilities:**
- Interactive 3D network visualization
- Node clustering by topic/community
- Real-time graph manipulation
- Export high-resolution images
- Multiple layout algorithms

### 🔍 System Diagnostics

Comprehensive system health monitoring:

```bash
# Basic health check
python check_system.py

# With auto-fix capabilities
python check_system.py --fix-issues

# Generate detailed report
python check_system.py --save-report diagnostic_report.json
```

### Direct Web Interface

Launch the web interface directly:

```bash
# Default settings
python web_chat.py

# Custom port and host
python web_chat.py --port 8080 --host 0.0.0.0

# Debug mode
python web_chat.py --debug
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Obsidian      │    │   Document       │    │   Knowledge     │
│   Vault         │───▶│   Processing     │───▶│   Graph         │
│   (.md files)   │    │   Pipeline       │    │   (NetworkX)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Vector        │    │   Graph RAG      │    │   Sequential    │
│   Embeddings    │◀───│   Retrieval      │───▶│   Thinking      │
│   (OpenAI)      │    │   Engine         │    │   (Multi-step)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌──────────────────┐
                    │   User           │
                    │   Interfaces     │
                    │   (Web/CLI/3D)   │
                    └──────────────────┘
```

### Core Components

#### 1. **Document Processing Pipeline**
- **VaultScanner**: Discovers and monitors markdown files
- **MarkdownParser**: Extracts content, frontmatter, and metadata
- **LinkExtractor**: Identifies wiki-links, tags, and references
- **ChunkProcessor**: Intelligently segments content for optimal retrieval

#### 2. **Knowledge Graph Construction**
- **Graph Builder**: Creates NetworkX graph from document relationships
- **Community Detection**: Identifies thematic clusters using Louvain algorithm
- **Centrality Analysis**: Measures node importance and connectivity
- **Dynamic Updates**: Efficiently processes incremental changes

#### 3. **Hybrid Retrieval System**
- **Semantic Search**: OpenAI embeddings with cosine similarity
- **Graph Traversal**: BFS/DFS through wiki-link connections
- **Result Fusion**: Intelligent combination of semantic and structural relevance
- **Context Expansion**: Multi-hop traversal for comprehensive context

#### 4. **Sequential Thinking Engine**
- **Query Decomposition**: Breaks complex questions into logical steps
- **Iterative Reasoning**: Builds understanding through multiple retrieval rounds
- **Evidence Synthesis**: Combines information from multiple sources
- **Confidence Assessment**: Evaluates answer reliability

#### 5. **Data Persistence Layer**
- **Smart Caching**: File modification tracking for incremental updates
- **Efficient Storage**: Optimized serialization with pickle
- **Cache Validation**: Automatic detection of vault changes
- **Recovery Systems**: Graceful handling of corrupted cache files

### Performance Optimizations

- **Batch Processing**: Efficient bulk operations for embeddings and graph construction
- **Lazy Loading**: On-demand data loading to minimize memory usage
- **Parallel Processing**: Multi-threaded operations where applicable
- **Token Management**: Intelligent chunking to respect API limits
- **Memory Pooling**: Reuse of expensive objects and computations

## 📖 API Reference

### Core Classes

#### `ObsidianGraphRAG`
Main system orchestrator.

```python
from graphrag import ObsidianGraphRAG, GraphRAGConfig

# Initialize with custom config
config = GraphRAGConfig()
config.vault_path = "/path/to/vault"
config.openai_api_key = "your-key"

# Create system
graph_rag = ObsidianGraphRAG(config)
graph_rag.initialize_system()
```

#### `GraphRAGConfig`
Configuration management with environment variable support.

```python
@dataclass
class GraphRAGConfig:
    vault_path: str = field(default_factory=lambda: os.getenv("OBSIDIAN_VAULT_PATH"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    # ... additional configuration options
```

#### `ObsidianChatBot`
Conversational interface for question-answering.

```python
chat_bot = ObsidianChatBot(graph_rag_system)
chat_bot.initialize()

# Ask questions
response = chat_bot.ask_question("What are my main research themes?")
```

#### `WebChatServer`
Flask-based web interface server.

```python
web_server = WebChatServer(config, graph_rag_system)
web_server.run(host="127.0.0.1", port=5000)
```

### Key Methods

#### Document Processing
```python
# Manual vault processing
graph_rag.scan_and_parse_vault()

# Get processing statistics
stats = {
    'documents': len(graph_rag.documents),
    'graph_nodes': graph_rag.knowledge_graph.number_of_nodes(),
    'graph_edges': graph_rag.knowledge_graph.number_of_edges()
}
```

#### Query Processing
```python
# Direct query with Graph RAG
retriever = GraphRAGRetriever(config)
context_docs = retriever.retrieve_context(
    query="research methodology",
    documents=graph_rag.documents,
    knowledge_graph=graph_rag.knowledge_graph,
    embedding_manager=embedding_manager
)
```

#### System Diagnostics
```python
# Health check
persistence_manager = DataPersistenceManager(config)
diagnosis = persistence_manager.diagnose_system_status(vault_path)

# Check for issues
if diagnosis['issues']:
    print("Issues found:", diagnosis['issues'])
    print("Recommendations:", diagnosis['recommendations'])
```

## 🔧 Troubleshooting

### Common Issues

#### 🔑 **OpenAI API Key Issues**

**Problem**: `OpenAI API key not found` or authentication errors.

**Solutions**:
```bash
# Check .env file exists and has correct format
cat .env | grep OPENAI_API_KEY

# Verify API key format (should start with 'sk-')
echo $OPENAI_API_KEY

# Test API key directly
python -c "
import openai
client = openai.OpenAI(api_key='your-key-here')
print('API key is valid!')
"
```

#### 📁 **Vault Path Problems**

**Problem**: `Vault path does not exist` or no files found.

**Solutions**:
```bash
# Check vault path in .env
python check_system.py

# Verify path exists and contains .md files
ls "C:\Path\To\Your\Vault" | grep ".md"

# Use forward slashes or raw strings in .env
OBSIDIAN_VAULT_PATH=C:/Users/Name/Documents/Obsidian Vault/Vault Name
```

#### 🌐 **Web Interface Issues**

**Problem**: Web interface won't start or shows errors.

**Solutions**:
```bash
# Install Flask if missing
pip install flask

# Check port availability
netstat -an | grep :5000

# Use different port
python web_chat.py --port 8080

# Check firewall settings
python web_chat.py --host 0.0.0.0
```

#### 💾 **Cache/Performance Issues**

**Problem**: Slow performance or cache corruption.

**Solutions**:
```bash
# Clear cache and rebuild
python check_system.py --fix-issues

# Manual cache cleanup
rm -rf ./cache
python start_chat.py

# Check disk space and permissions
df -h
ls -la ./cache
```

#### 🔤 **Unicode/Encoding Issues (Windows)**

**Problem**: Character encoding errors in terminal.

**Solutions**:
```powershell
# Set UTF-8 encoding in PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Use Windows Terminal or Command Prompt
# Update PowerShell to latest version
```

### Performance Tuning

For large vaults (1000+ documents):

```bash
# Increase chunk size for better performance
CHUNK_SIZE=1000

# Reduce embedding dimensions
TOP_K_VECTOR=5
TOP_K_GRAPH=3

# Use more aggressive caching
ENABLE_CACHING=true
CACHE_TTL=7200
```

### Getting Help

1. **Run Diagnostics**: `python check_system.py --save-report`
2. **Check Logs**: Review `./logs/graphrag.log`
3. **Enable Debug Mode**: Set `LOG_LEVEL=DEBUG` in `.env`
4. **GitHub Issues**: Report bugs with diagnostic report
5. **Community**: Join our discussions for tips and tricks

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Fork the repository
git clone https://github.com/your-username/obsidian-graph-rag.git
cd obsidian-graph-rag

# Create development environment
python -m venv dev-env
source dev-env/bin/activate  # or dev-env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements_viz.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Contribution Guidelines

1. **Code Style**: Follow PEP 8 and use type hints
2. **Testing**: Add tests for new features
3. **Documentation**: Update docstrings and README
4. **Commits**: Use conventional commit format
5. **Pull Requests**: Include description and test results

### Areas for Contribution

- 🌐 **Additional Language Support**: Non-English vault processing
- 🔌 **Plugin System**: Extensible processing pipeline
- 📱 **Mobile Interface**: React Native or PWA
- ☁️ **Cloud Deployment**: Docker containers and cloud setup
- 🧪 **Advanced Analytics**: Statistical insights and trends
- 🔒 **Security Enhancements**: Advanced privacy controls

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for providing powerful language models and embeddings
- **Obsidian** for creating an amazing knowledge management platform
- **NetworkX** for robust graph processing capabilities
- **Flask** for the excellent web framework
- **The Open Source Community** for inspiration and contributions

---

## 🚀 What's Next?

Transform your Obsidian vault into an intelligent knowledge companion:

1. **[Install now](#-installation)** and get started in minutes
2. **[Join our community](https://github.com/your-username/obsidian-graph-rag/discussions)** for tips and tricks
3. **[Star the repository](https://github.com/your-username/obsidian-graph-rag)** to stay updated
4. **[Report issues](https://github.com/your-username/obsidian-graph-rag/issues)** to help us improve

**Ready to unlock the full potential of your knowledge?** 

[![Get Started](https://img.shields.io/badge/Get%20Started-Now-blue?style=for-the-badge)](https://github.com/your-username/obsidian-graph-rag#-quick-start)

---

<div align="center">

**Built with ❤️ for the Obsidian community**

[🌟 Star](https://github.com/your-username/obsidian-graph-rag) • [🐛 Issues](https://github.com/your-username/obsidian-graph-rag/issues) • [💬 Discussions](https://github.com/your-username/obsidian-graph-rag/discussions) • [📖 Wiki](https://github.com/your-username/obsidian-graph-rag/wiki)

</div> 