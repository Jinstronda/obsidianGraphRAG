"""
Demo Knowledge Base Viewer
Quick demo showing specific examples from your GraphRAG knowledge base.
"""

import json
from collections import Counter, defaultdict

def demo_emerald_tablet():
    """Demo showing Emerald Tablet related entities and relations."""
    print("🔮 EMERALD TABLET KNOWLEDGE BASE DEMO")
    print("="*60)
    
    # Load entities
    entities = []
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(json.loads(line))
    
    # Find Emerald Tablet related entities
    emerald_entities = []
    for entity in entities:
        if 'emerald' in entity['entity'].lower() or 'tablet' in entity['entity'].lower():
            emerald_entities.append(entity)
    
    print(f"📊 Found {len(emerald_entities)} Emerald Tablet related entities:")
    
    # Group by type
    by_type = defaultdict(list)
    for entity in emerald_entities:
        by_type[entity['type']].append(entity)
    
    for entity_type, type_entities in by_type.items():
        print(f"\n🏷️  {entity_type}:")
        for entity in type_entities[:5]:
            print(f"  • {entity['entity']} (from {entity['source_file']})")
    
    # Load some relations
    relations = []
    with open("G-Indexation/Graph_fragments/edges.jsonl", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1000:  # Limit for demo
                break
            relations.append(json.loads(line))
    
    # Find Emerald Tablet relations
    emerald_relations = []
    for relation in relations:
        if ('emerald' in relation['entity1'].lower() or 'tablet' in relation['entity1'].lower() or
            'emerald' in relation['entity2'].lower() or 'tablet' in relation['entity2'].lower()):
            emerald_relations.append(relation)
    
    print(f"\n🔗 Found {len(emerald_relations)} Emerald Tablet relations:")
    for i, relation in enumerate(emerald_relations[:10], 1):
        print(f"\n  {i}. {relation['entity1']} ←→ {relation['entity2']}")
        print(f"     Source: {relation['source_file']}")
        print(f"     Context: {relation['sentence'][:100]}...")

def demo_hermetic_concepts():
    """Demo showing hermetic philosophy concepts."""
    print("\n\n🔬 HERMETIC PHILOSOPHY CONCEPTS DEMO")
    print("="*60)
    
    # Load entities
    entities = []
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(json.loads(line))
    
    # Find key hermetic concepts
    hermetic_concepts = []
    keywords = ['hermes', 'alchemy', 'philosopher', 'stone', 'gold', 'silver', 'mercury', 'sulfur', 'salt']
    
    for entity in entities:
        entity_lower = entity['entity'].lower()
        if any(keyword in entity_lower for keyword in keywords):
            hermetic_concepts.append(entity)
    
    print(f"📊 Found {len(hermetic_concepts)} hermetic concept entities:")
    
    # Group by concept type
    concept_groups = {
        'Hermes/Thoth': [],
        'Alchemical Elements': [],
        'Philosophical Concepts': [],
        'Alchemical Processes': []
    }
    
    for entity in hermetic_concepts:
        entity_lower = entity['entity'].lower()
        if 'hermes' in entity_lower or 'thoth' in entity_lower:
            concept_groups['Hermes/Thoth'].append(entity)
        elif any(elem in entity_lower for elem in ['gold', 'silver', 'mercury', 'sulfur', 'salt']):
            concept_groups['Alchemical Elements'].append(entity)
        elif any(proc in entity_lower for proc in ['solve', 'coagula', 'distill', 'purify']):
            concept_groups['Alchemical Processes'].append(entity)
        else:
            concept_groups['Philosophical Concepts'].append(entity)
    
    for group_name, group_entities in concept_groups.items():
        if group_entities:
            print(f"\n🏷️  {group_name}:")
            for entity in group_entities[:5]:
                print(f"  • {entity['entity']} ({entity['type']})")
            if len(group_entities) > 5:
                print(f"  ... and {len(group_entities) - 5} more")

def demo_source_comparison():
    """Demo comparing different source texts."""
    print("\n\n📚 SOURCE TEXT COMPARISON DEMO")
    print("="*60)
    
    # Load entities
    entities = []
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entities.append(json.loads(line))
    
    # Compare key sources
    key_sources = ['emerald_tablet.md', 'hermetic_arcanum.md', 'golden_chain_of_homer.md']
    
    for source in key_sources:
        source_entities = [e for e in entities if e['source_file'] == source]
        
        print(f"\n📖 {source}:")
        print(f"  Total entities: {len(source_entities)}")
        
        # Count by type
        type_counts = Counter(e['type'] for e in source_entities)
        print(f"  Top entity types:")
        for entity_type, count in type_counts.most_common(5):
            print(f"    {entity_type}: {count}")
        
        # Show sample entities
        people = [e for e in source_entities if e['type'] == 'PERSON']
        works = [e for e in source_entities if e['type'] == 'WORK_OF_ART']
        
        if people:
            print(f"  Sample people: {', '.join(e['entity'] for e in people[:3])}")
        if works:
            print(f"  Sample works: {', '.join(e['entity'] for e in works[:3])}")

def demo_entity_network():
    """Demo showing entity network connections."""
    print("\n\n🌐 ENTITY NETWORK DEMO")
    print("="*60)
    
    # Load relations (limited for demo)
    relations = []
    with open("G-Indexation/Graph_fragments/edges.jsonl", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5000:  # Limit for demo
                break
            relations.append(json.loads(line))
    
    # Find most connected entities
    entity_connections = defaultdict(int)
    for relation in relations:
        entity_connections[relation['entity1']] += 1
        entity_connections[relation['entity2']] += 1
    
    print("🏆 Most connected entities:")
    for entity, connections in sorted(entity_connections.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {entity}: {connections} connections")
    
    # Show network for a specific entity
    target_entity = "Hermes"
    print(f"\n🔗 Network for '{target_entity}':")
    
    connected_entities = set()
    for relation in relations:
        if target_entity.lower() in relation['entity1'].lower():
            connected_entities.add(relation['entity2'])
        elif target_entity.lower() in relation['entity2'].lower():
            connected_entities.add(relation['entity1'])
    
    print(f"  Connected to {len(connected_entities)} entities:")
    for entity in sorted(connected_entities)[:15]:
        print(f"    • {entity}")

def main():
    """Run all demos."""
    print("🚀 GRAPHRAG KNOWLEDGE BASE DEMOS")
    print("="*60)
    print("This shows you specific examples from your extracted knowledge base.")
    print("="*60)
    
    demo_emerald_tablet()
    demo_hermetic_concepts()
    demo_source_comparison()
    demo_entity_network()
    
    print("\n" + "="*60)
    print("🎉 DEMO COMPLETE")
    print("="*60)
    print("Your knowledge base contains rich information about:")
    print("• Hermetic philosophy and alchemy")
    print("• Historical figures and texts")
    print("• Alchemical processes and concepts")
    print("• Relationships between ideas and authors")
    print("\nReady to load into Neo4j for full graph exploration!")

if __name__ == "__main__":
    main() 