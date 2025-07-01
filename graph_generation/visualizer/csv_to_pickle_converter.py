import os
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Set, Optional
from datetime import datetime

# Minimal Document class for compatibility
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
    embedding: Optional[Any] = None

LIBRARY_DIR = Path(__file__).parent.parent / 'Library'
CACHE_DIR = Path(__file__).parent / 'cache' / 'processed_data'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

documents = {}
G = nx.Graph()

for md_file in LIBRARY_DIR.glob('*.md'):
    base = md_file.name
    stem = md_file.stem
    nodes_csv = LIBRARY_DIR / f'{base}_nodes.csv'
    edges_csv = LIBRARY_DIR / f'{base}_edges.csv'
    entities_csv = LIBRARY_DIR / f'{base}_entities.csv'
    if not (nodes_csv.exists() and edges_csv.exists()):
        continue
    # Load nodes
    nodes_df = pd.read_csv(nodes_csv)
    # Load edges
    edges_df = pd.read_csv(edges_csv)
    # Add nodes to graph
    for _, row in nodes_df.iterrows():
        node_id = str(row.get('id', row.get('node', '')))
        G.add_node(node_id, **row.to_dict())
    # Add edges to graph
    for _, row in edges_df.iterrows():
        source = str(row.get('source', ''))
        target = str(row.get('target', ''))
        G.add_edge(source, target, **row.to_dict())
    # Load document content
    with open(md_file, encoding='utf-8') as f:
        content = f.read()
    # Use entities for title if available
    title = stem
    if entities_csv.exists():
        entities_df = pd.read_csv(entities_csv)
        if 'title' in entities_df.columns and not entities_df.empty:
            title = str(entities_df['title'].iloc[0])
    doc_id = stem
    documents[doc_id] = Document(
        id=doc_id,
        path=str(md_file),
        title=title,
        content=content,
        word_count=len(content.split())
    )

# Save graph and documents
with open(CACHE_DIR / 'knowledge_graph.gpickle', 'wb') as f:
    pickle.dump(G, f)
with open(CACHE_DIR / 'documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print(f"Saved graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
print(f"Saved {len(documents)} documents.") 