# G-Indexation: Complete GraphRAG System

A comprehensive GraphRAG (Graph-based Retrieval-Augmented Generation) system that processes documents through 6 distinct stages to create a knowledge graph for intelligent information retrieval and generation.

## 🏗️ System Architecture

The G-Indexation system consists of 4 sequential steps:

1. **Chunking** - Document segmentation into manageable pieces
2. **Entity & Relationship Extraction** - Knowledge extraction from chunks
3. **Knowledge Graph Loading** - Neo4j database population
4. **GraphRAG System Integration** - Complete system assembly

## 📁 File Structure

```
G-Indexation/
├── chunker.py                        # Step 1: Document chunking
├── extractor_01.py                   # Step 2: Entity/relationship extraction
├── load_graphrag_to_neo4j.py         # Step 3: Neo4j knowledge graph loading
├── 02_openie_extraction.py           # Alternative: OpenIE extraction
├── Hermetic_Library/                 # Source documents (.md files)
├── Graph_fragments/                  # Output directory
│   ├── chunks.json                   # Document chunks
│   ├── nodes.jsonl                   # Extracted entities
│   ├── edges.jsonl                   # Extracted relationships
│   └── openie_triples.json           # OpenIE semantic triples
├── Prompts/                          # Prompt templates
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.json` to customize the system:

```json
{
  "source_path": "G-Indexation/Hermetic_Library",
  "sentences_per_chunk": 5,
  "overlap_sentences": 2,
  "output_folder": "G-Indexation/Graph_fragments"
}
```

### 3. Run Complete Pipeline

```bash
# Step 1: Chunk documents
python G-Indexation/chunker.py

# Step 2: Extract entities and relations
python G-Indexation/extractor_01.py

# Step 3: Load into Neo4j (when database is running)
python G-Indexation/load_graphrag_to_neo4j.py

# Alternative: OpenIE extraction
python G-Indexation/02_openie_extraction.py
```

## 📋 Step-by-Step Guide

### Step 1: Chunking (`chunker.py`)

**Purpose**: Segment documents into manageable chunks with overlap for context preservation.

**Features**:
- Sentence-based chunking using NLTK
- Configurable chunk size and overlap
- Markdown file processing
- Globally unique chunk IDs
- Sentence index tracking

**Output**: `chunks.json` with structure:
```json
{
  "chunk_id": "emerald_tablet.md_chunk_000001",
  "source_file": "emerald_tablet.md",
  "start_sentence": 0,
  "end_sentence": 4,
  "text": "Chunk content..."
}
```

### Step 2: Entity & Relationship Extraction (`extractor_01.py`)

**Purpose**: Extract entities and relationships from chunks using spaCy NLP techniques.

**Features**:
- Named Entity Recognition (NER) with fallback models
- Noun chunk extraction
- Co-occurrence relationship extraction
- Modular extractor design for easy swapping
- JSONL output format for efficiency
- Complete traceability to source files

**Output**: 
- `nodes.jsonl` - Extracted entities (34,138 entities)
- `edges.jsonl` - Extracted relationships (208,263 relations)

### Step 3: Knowledge Graph Loading (`load_graphrag_to_neo4j.py`)

**Purpose**: Load extracted entities and relationships into Neo4j graph database.

**Features**:
- Neo4j database connection and management
- Entity node creation with properties
- Relationship edge creation with context
- Database constraints and indexes for performance
- Graph analysis and statistics
- Complete knowledge graph visualization

**Output**: Neo4j graph database with:
- Entity nodes with properties (name, type, source, etc.)
- Relationship edges with context sentences
- Graph statistics and analysis
- Browser interface at http://localhost:7474

### Step 4: GraphRAG System Integration (Future)

**Purpose**: Integrate all components into a complete GraphRAG system.

**Features**:
- Query processing and retrieval
- Contextual response generation
- Interactive demo mode
- System state management

**Output**: Complete GraphRAG system ready for question answering

## 🎮 Interactive Usage

### Command Line Interface

```bash
# Run complete pipeline
python G-Indexation/run_full_graphrag_pipeline.py all

# Run specific step
python G-Indexation/run_full_graphrag_pipeline.py step 2

# Check pipeline status
python G-Indexation/run_full_graphrag_pipeline.py status

# Interactive mode
python G-Indexation/run_full_graphrag_pipeline.py
```

### Individual Step Execution

```bash
# Run each step individually
python G-Indexation/01_chunking.py
python G-Indexation/02_entity_extraction.py
python G-Indexation/03_knowledge_base_construction.py
python G-Indexation/04_community_detection.py
python G-Indexation/05_summarization.py
python G-Indexation/06_graphrag_system.py
```

## 🔧 Configuration Options

### Chunking Configuration (`config.json`)

```json
{
  "source_path": "G-Indexation/Hermetic_Library",
  "sentences_per_chunk": 5,
  "overlap_sentences": 2,
  "output_folder": "G-Indexation/Graph_fragments"
}
```

### Advanced Configuration

Each step can be customized by modifying the respective Python files:

- **Entity Extraction**: Adjust NER models and extraction parameters
- **Community Detection**: Configure detection algorithms and thresholds
- **Summarization**: Set summary length and style preferences
- **GraphRAG System**: Modify retrieval strategies and response generation

## 📊 System Statistics

After running the complete pipeline, you can view comprehensive statistics:

```bash
python G-Indexation/run_full_graphrag_pipeline.py status
```

This will show:
- Number of chunks created
- Entities and relationships extracted
- Knowledge graph nodes and edges
- Communities detected
- Summaries generated
- System performance metrics

## 🎯 Use Cases

### 1. Document Analysis
- Process large document collections
- Extract key information and relationships
- Identify document themes and topics

### 2. Knowledge Discovery
- Discover hidden connections between concepts
- Identify knowledge gaps
- Generate insights from document collections

### 3. Question Answering
- Build intelligent Q&A systems
- Provide contextual responses
- Support complex multi-step reasoning

### 4. Content Summarization
- Generate multi-level summaries
- Create topic-based overviews
- Produce hierarchical content organization

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **File Not Found**: Check that source documents exist in the specified path

3. **Memory Issues**: Reduce chunk size or process documents in batches

4. **Neo4j Connection**: Ensure Neo4j database is running if using database features

### Performance Optimization

- Adjust chunk size based on available memory
- Use parallel processing for large document collections
- Implement caching for intermediate results
- Optimize entity extraction parameters

## 🤝 Contributing

To contribute to the G-Indexation system:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with the `neo4j-graphrag` library
- Uses NLTK for natural language processing
- Inspired by modern GraphRAG architectures
- Designed for hermetic and philosophical text analysis

---

**G-Indexation**: Transforming documents into intelligent knowledge graphs for enhanced information retrieval and generation. 