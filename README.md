# 🚀 Obsidian RAG-Anything with EmbeddingGemma 308M

**Clean, minimal implementation for processing Obsidian vaults with SOTA chunking and RAG-Anything framework.**

## 🎯 Features

- **✅ SOTA Chunking**: 2K token windows with wikilinks & metadata preservation
- **✅ EmbeddingGemma 308M**: Local, cost-free embeddings (100+ languages)
- **✅ RAG-Anything**: Pure framework with multimodal processing
- **✅ Obsidian Integration**: Wikilinks, tags, frontmatter preserved
- **✅ Conda Environment**: Always runs in `turing0.1`

## 📦 Requirements

- **Conda Environment**: `turing0.1` (required)
- **Python**: 3.9+
- **Vault Path**: `C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro`

## 🚀 Quick Start

### 1. Activate Conda Environment
```bash
conda activate turing0.1
```

### 2. Setup Dependencies
```bash
bash setup_conda_env.sh
```

### 3. Run RAG-Anything
```bash
python run_obsidian_raganything.py
```

## 📁 Clean File Structure

```
├── src/
│   ├── simple_raganything.py    # Main RAG-Anything implementation
│   └── obsidian_chunker.py     # SOTA chunking with wikilinks
├── run_obsidian_raganything.py  # Runner script
├── setup_conda_env.sh          # Setup script
├── requirements.txt            # Minimal dependencies
└── README.md                   # This file
```

## 🔧 Configuration

### Environment Variables (Optional)
```bash
export OBSIDIAN_VAULT_PATH="/path/to/your/vault"  # Default: Your vault
export WORKING_DIR="./rag_storage"                # Default: ./rag_storage
```

### EmbeddingGemma 308M Settings
- **Model**: `google/embeddinggemma-300m`
- **Dimensions**: 768 (truncatable to 512/256/128)
- **Memory**: <200MB with quantization
- **Languages**: 100+ supported
- **Privacy**: Fully offline processing

## 🎯 SOTA Chunking Features

### **2K Token Windows**
- Precise token estimation and chunking
- Maintains document structure
- Preserves context across chunks

### **Wikilinks Preservation**
- Every chunk maintains its connections
- Critical for RAG knowledge graph
- Enables cross-document reasoning

### **Metadata Preservation**
- Frontmatter in every chunk
- Tags and file information
- Creation/modification timestamps

### **File Connections**
- Chunks from same file are linked
- Previous/next chunk tracking
- Document structure awareness

## 🚀 Usage Examples

### Basic Processing
```python
from src.simple_raganything import SimpleRAGAnything

# Initialize (always uses conda turing0.1)
rag = SimpleRAGAnything(vault_path, working_dir)
await rag.initialize()

# Process vault with SOTA chunking
await rag.process_vault()

# Query with preserved connections
result = await rag.query("What are the main topics?")
```

### Multimodal Queries
```python
# Query with multimodal content
result = await rag.query_multimodal("What images and tables are available?")
```

## 📊 Expected Output

```
🚀 PROCESSING OBSIDIAN VAULT WITH SOTA CHUNKING
📦 Step 1: Chunking vault with 2K token windows...
🔄 Preserving wikilinks, metadata, and file connections...

📊 Chunking Complete:
   📄 Files: 150
   📦 Chunks: 300
   🔗 Wikilinks: 1200
   🏷️ Tags: 450

🔄 Step 2: Processing 300 chunks with RAG-Anything...
🔄 Using EmbeddingGemma 308M for embeddings
🔄 Multimodal processing: Images, Tables, Equations
```

## 🛠️ Troubleshooting

### Conda Environment Issues
```bash
# Check current environment
echo $CONDA_DEFAULT_ENV

# Activate correct environment
conda activate turing0.1
```

### Import Errors
```bash
# Reinstall dependencies
bash setup_conda_env.sh
```

## 🎯 Key Benefits

- **Cost-Free**: EmbeddingGemma eliminates API costs
- **Privacy**: Complete offline processing
- **Performance**: <200MB RAM, <15ms latency
- **Multilingual**: 100+ languages support
- **Connections**: Wikilinks preserved for knowledge graph
- **Clean Code**: Minimal, well-documented implementation

## 📝 Notes

- **Always runs in conda environment `turing0.1`**
- **Default vault**: `C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro`
- **2K token chunks** with wikilinks and metadata preserved
- **Pure RAG-Anything** implementation (no LightRAG)
- **EmbeddingGemma 308M** for local, cost-free embeddings

