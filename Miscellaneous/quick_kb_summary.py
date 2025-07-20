"""
Quick Knowledge Base Summary
Shows the most interesting and important parts of your GraphRAG knowledge base.
"""

import json
from collections import Counter, defaultdict

def quick_summary():
    """Show a quick summary of the most interesting parts of the knowledge base."""
    print("🚀 GRAPHRAG KNOWLEDGE BASE - QUICK SUMMARY")
    print("="*60)
    
    # Load entities
    entities = []
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(json.loads(line))
    
    print(f"📊 TOTAL: {len(entities):,} entities extracted from 11 hermetic texts")
    
    # Most important entity types
    entity_types = Counter(e['type'] for e in entities)
    print(f"\n🏷️  ENTITY TYPES:")
    for entity_type, count in entity_types.most_common():
        print(f"  {entity_type}: {count:,}")
    
    # Most important people
    people = [e for e in entities if e['type'] == 'PERSON']
    people_counts = Counter(e['entity'] for e in people)
    print(f"\n👥 TOP PEOPLE ({len(people)} total):")
    for person, count in people_counts.most_common(15):
        print(f"  {person}: {count} mentions")
    
    # Most important works
    works = [e for e in entities if e['type'] == 'WORK_OF_ART']
    works_counts = Counter(e['entity'] for e in works)
    print(f"\n📚 TOP WORKS ({len(works)} total):")
    for work, count in works_counts.most_common(10):
        print(f"  {work}: {count} mentions")
    
    # Alchemical concepts
    alchemical_keywords = ['gold', 'silver', 'mercury', 'sulfur', 'salt', 'stone', 'philosopher', 'alchemy']
    alchemical_entities = []
    for entity in entities:
        entity_lower = entity['entity'].lower()
        if any(keyword in entity_lower for keyword in alchemical_keywords):
            alchemical_entities.append(entity)
    
    alchemical_counts = Counter(e['entity'] for e in alchemical_entities)
    print(f"\n🔬 TOP ALCHEMICAL CONCEPTS ({len(alchemical_entities)} total):")
    for concept, count in alchemical_counts.most_common(15):
        print(f"  {concept}: {count} mentions")
    
    # Source comparison
    source_counts = Counter(e['source_file'] for e in entities)
    print(f"\n📖 ENTITIES BY SOURCE:")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count:,} entities")
    
    # Load some relations for network analysis
    relations = []
    with open("G-Indexation/Graph_fragments/edges.jsonl", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5000:  # Limit for performance
                break
            relations.append(json.loads(line))
    
    # Most connected entities
    entity_connections = defaultdict(int)
    for relation in relations:
        entity_connections[relation['entity1']] += 1
        entity_connections[relation['entity2']] += 1
    
    print(f"\n🌐 MOST CONNECTED ENTITIES (from {len(relations):,} relations):")
    for entity, connections in sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {entity}: {connections} connections")
    
    # Hermes network
    hermes_connections = set()
    for relation in relations:
        if 'hermes' in relation['entity1'].lower():
            hermes_connections.add(relation['entity2'])
        elif 'hermes' in relation['entity2'].lower():
            hermes_connections.add(relation['entity1'])
    
    print(f"\n🔗 HERMES NETWORK ({len(hermes_connections)} connected entities):")
    for entity in sorted(hermes_connections)[:20]:
        print(f"  • {entity}")
    if len(hermes_connections) > 20:
        print(f"  ... and {len(hermes_connections) - 20} more")
    
    print("\n" + "="*60)
    print("🎉 YOUR KNOWLEDGE BASE IS READY!")
    print("="*60)
    print("You have successfully extracted:")
    print(f"• {len(entities):,} entities from hermetic texts")
    print(f"• {len(relations):,} relationships (sample)")
    print(f"• Rich alchemical and philosophical concepts")
    print(f"• Historical figures and their connections")
    print(f"• Complete traceability to source texts")
    print("\nNext: Load into Neo4j for full graph exploration!")

if __name__ == "__main__":
    quick_summary() 