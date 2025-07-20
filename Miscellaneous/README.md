# Miscellaneous Utilities

This folder contains utility scripts, test files, and documentation for the GraphRAG project.

## 📁 File Descriptions

### 🔍 Knowledge Base Exploration
- **`explore_graphrag_data.py`** - Comprehensive overview of extracted data
- **`view_knowledge_base.py`** - Interactive knowledge base explorer with search
- **`demo_knowledge_base.py`** - Demo showing specific examples from the knowledge base
- **`quick_kb_summary.py`** - Quick summary of key knowledge base statistics

### 🔧 Neo4j Database Tools
- **`quick_neo4j_test.py`** - Quick Neo4j connection test
- **`test_neo4j_connection.py`** - Comprehensive Neo4j connection testing
- **`set_neo4j_password.py`** - Set/reset Neo4j password

### 📚 Documentation
- **`neo4j_setup_guide.md`** - Guide for installing and setting up Neo4j
- **`neo4j_database_setup.md`** - Comprehensive database setup instructions

### 🔬 Analysis and Testing
- **`analyze_extraction.py`** - Analyze extraction results and statistics
- **`check_openie_results.py`** - Check OpenIE extraction results
- **`show_all_triples.py`** - Display all extracted triples

## 🚀 Usage

### Knowledge Base Exploration
```bash
# Quick overview
python quick_kb_summary.py

# Interactive exploration
python view_knowledge_base.py

# Demo examples
python demo_knowledge_base.py

# Comprehensive analysis
python explore_graphrag_data.py
```

### Neo4j Database
```bash
# Test connection
python quick_neo4j_test.py

# Set password
python set_neo4j_password.py

# Load data (run from project root)
python G-Indexation/load_graphrag_to_neo4j.py
```

### Analysis
```bash
# Analyze extraction
python analyze_extraction.py

# Check OpenIE results
python check_openie_results.py

# Show triples
python show_all_triples.py
```

## 📊 Data Files

The scripts reference data files in `../G-Indexation/Graph_fragments/`:
- `nodes.jsonl` - 34,138 extracted entities
- `edges.jsonl` - 208,263 extracted relations
- `chunks.json` - 4,175 text chunks
- `openie_triples.json` - 133 semantic triples

## 🎯 Purpose

These utilities help with:
- **Data exploration** - Understanding what was extracted
- **Database setup** - Neo4j installation and configuration
- **Testing** - Verifying connections and data integrity
- **Analysis** - Deep diving into extraction results
- **Documentation** - Setup guides and instructions

## 🔗 Main Project

The core GraphRAG pipeline is in the `G-Indexation/` folder:
- `chunker.py` - Text chunking
- `extractor_01.py` - Entity and relation extraction
- `load_graphrag_to_neo4j.py` - Neo4j knowledge graph loading
- `Graph_fragments/` - Extracted data files 