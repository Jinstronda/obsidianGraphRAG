"""
Explore GraphRAG Data
This script shows you what data has been extracted and stored in your GraphRAG system.
"""

import json
from collections import Counter, defaultdict
import os

def explore_nodes(nodes_file="G-Indexation/Graph_fragments/nodes.jsonl", max_samples=10):
    """Explore the extracted entities/nodes."""
    print("🔍 EXPLORING EXTRACTED ENTITIES")
    print("="*60)
    
    if not os.path.exists(nodes_file):
        print(f"❌ File not found: {nodes_file}")
        return
    
    entity_types = Counter()
    source_files = Counter()
    sample_entities = defaultdict(list)
    
    with open(nodes_file, 'r', encoding='utf-8') as f:
        for line in f:
            entity_data = json.loads(line)
            
            # Count entity types
            entity_types[entity_data['type']] += 1
            
            # Count source files
            source_files[entity_data['source_file']] += 1
            
            # Collect samples
            if len(sample_entities[entity_data['type']]) < max_samples:
                sample_entities[entity_data['type']].append(entity_data['entity'])
    
    print(f"📊 Total entities extracted: {sum(entity_types.values()):,}")
    print(f"📁 Source files processed: {len(source_files)}")
    
    print(f"\n🏷️  Entity types found:")
    for entity_type, count in entity_types.most_common():
        print(f"  {entity_type}: {count:,} entities")
    
    print(f"\n📚 Source files:")
    for source_file, count in source_files.most_common():
        print(f"  {source_file}: {count:,} entities")
    
    print(f"\n📝 Sample entities by type:")
    for entity_type, samples in sample_entities.items():
        print(f"\n  {entity_type}:")
        for sample in samples[:5]:  # Show first 5 samples
            print(f"    • {sample}")
        if len(samples) > 5:
            print(f"    ... and {len(samples) - 5} more")

def explore_edges(edges_file="G-Indexation/Graph_fragments/edges.jsonl", max_samples=10):
    """Explore the extracted relations/edges."""
    print("\n🔗 EXPLORING EXTRACTED RELATIONS")
    print("="*60)
    
    if not os.path.exists(edges_file):
        print(f"❌ File not found: {edges_file}")
        return
    
    source_files = Counter()
    chunk_ids = Counter()
    sample_relations = []
    
    with open(edges_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Only process first 1000 for analysis
                break
                
            edge_data = json.loads(line)
            
            # Count source files
            source_files[edge_data['source_file']] += 1
            
            # Count chunk IDs
            chunk_ids[edge_data['chunk_id']] += 1
            
            # Collect samples
            if len(sample_relations) < max_samples:
                sample_relations.append(edge_data)
    
    # Count total relations
    total_relations = sum(1 for _ in open(edges_file, 'r', encoding='utf-8'))
    
    print(f"📊 Total relations extracted: {total_relations:,}")
    print(f"📁 Source files with relations: {len(source_files)}")
    print(f"📄 Chunks with relations: {len(chunk_ids)}")
    
    print(f"\n📚 Source files by relation count:")
    for source_file, count in source_files.most_common():
        print(f"  {source_file}: {count:,} relations")
    
    print(f"\n📝 Sample relations:")
    for i, relation in enumerate(sample_relations, 1):
        print(f"\n  {i}. {relation['entity1']} ←→ {relation['entity2']}")
        print(f"     Source: {relation['source_file']}")
        print(f"     Chunk: {relation['chunk_id']}")
        print(f"     Sentence: {relation['sentence'][:100]}...")

def explore_chunks(chunks_file="G-Indexation/Graph_fragments/chunks.json"):
    """Explore the original text chunks."""
    print("\n📄 EXPLORING TEXT CHUNKS")
    print("="*60)
    
    if not os.path.exists(chunks_file):
        print(f"❌ File not found: {chunks_file}")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"📊 Total chunks: {len(chunks):,}")
    
    # Analyze chunks by source
    source_chunks = Counter()
    chunk_lengths = []
    
    for chunk in chunks:
        source_chunks[chunk['source_file']] += 1
        chunk_lengths.append(len(chunk['text']))
    
    print(f"\n📚 Chunks by source file:")
    for source_file, count in source_chunks.most_common():
        print(f"  {source_file}: {count:,} chunks")
    
    print(f"\n📏 Chunk statistics:")
    print(f"  Average length: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
    print(f"  Shortest chunk: {min(chunk_lengths)} characters")
    print(f"  Longest chunk: {max(chunk_lengths)} characters")
    
    print(f"\n📝 Sample chunks:")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n  Chunk {i} (from {chunk['source_file']}):")
        print(f"    Text: {chunk['text'][:200]}...")
        print(f"    Length: {len(chunk['text'])} characters")

def explore_triples(triples_file="G-Indexation/Graph_fragments/openie_triples.json"):
    """Explore the OpenIE triples."""
    print("\n🔍 EXPLORING OPENIE TRIPLES")
    print("="*60)
    
    if not os.path.exists(triples_file):
        print(f"❌ File not found: {triples_file}")
        return
    
    with open(triples_file, 'r', encoding='utf-8') as f:
        triples_data = json.load(f)
    
    print(f"📊 Total triples extracted: {len(triples_data):,}")
    
    # Analyze triples
    subjects = Counter()
    predicates = Counter()
    objects = Counter()
    
    for triple in triples_data:
        subjects[triple['subject']] += 1
        predicates[triple['predicate']] += 1
        objects[triple['object']] += 1
    
    print(f"\n🏷️  Top subjects:")
    for subject, count in subjects.most_common(10):
        print(f"  {subject}: {count}")
    
    print(f"\n🔗 Top predicates:")
    for predicate, count in predicates.most_common(10):
        print(f"  {predicate}: {count}")
    
    print(f"\n📝 Sample triples:")
    for i, triple in enumerate(triples_data[:5], 1):
        print(f"\n  {i}. {triple['subject']} → {triple['predicate']} → {triple['object']}")
        print(f"     Confidence: {triple['confidence']:.3f}")
        print(f"     Source: {triple['source_file']}")

def show_file_sizes():
    """Show the size of all data files."""
    print("\n💾 DATA FILE SIZES")
    print("="*60)
    
    files = [
        "G-Indexation/Graph_fragments/chunks.json",
        "G-Indexation/Graph_fragments/nodes.jsonl", 
        "G-Indexation/Graph_fragments/edges.jsonl",
        "G-Indexation/Graph_fragments/openie_triples.json"
    ]
    
    for file_path in files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file_path}: {size_mb:.1f} MB")
        else:
            print(f"  {file_path}: Not found")

def main():
    """Main function to explore all GraphRAG data."""
    print("🔍 GRAPHRAG DATA EXPLORER")
    print("="*60)
    print("This shows you what data has been extracted and stored in your GraphRAG system.")
    print("="*60)
    
    # Show file sizes first
    show_file_sizes()
    
    # Explore each data type
    explore_chunks()
    explore_nodes()
    explore_edges()
    explore_triples()
    
    print("\n" + "="*60)
    print("🎉 DATA EXPLORATION COMPLETE")
    print("="*60)
    print("Your GraphRAG data is ready to be loaded into Neo4j!")
    print("Run: python load_graphrag_to_neo4j.py")

if __name__ == "__main__":
    main() 