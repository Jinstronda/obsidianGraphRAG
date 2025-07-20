import json
from collections import defaultdict, Counter

def analyze_jsonl(file_path, max_lines=10):
    """Analyze a JSONL file and show sample entries."""
    entities = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            entities.append(json.loads(line))
    return entities

print("MODULAR EXTRACTOR RESULTS ANALYSIS")
print("=" * 60)

# Analyze nodes (entities)
print("\n📊 ENTITIES ANALYSIS (nodes.jsonl):")
try:
    sample_entities = analyze_jsonl("G-Indexation/Graph_fragments/nodes.jsonl", 5)
    print(f"Sample entities:")
    for i, entity in enumerate(sample_entities, 1):
        print(f"  {i}. {entity['entity']} ({entity['type']}) - {entity['extraction_method']}")
        print(f"     Source: {entity['source_file']} - {entity['chunk_id']}")
except Exception as e:
    print(f"Error reading nodes.jsonl: {e}")

# Analyze edges (relations)
print("\n🔗 RELATIONS ANALYSIS (edges.jsonl):")
try:
    sample_edges = analyze_jsonl("G-Indexation/Graph_fragments/edges.jsonl", 5)
    print(f"Sample edges:")
    for i, edge in enumerate(sample_edges, 1):
        print(f"  {i}. {edge['entity1']} <-> {edge['entity2']}")
        print(f"     Types: [{edge['entity1_type']}] <-> [{edge['entity2_type']}]")
        print(f"     Source: {edge['source_file']} - {edge['chunk_id']}")
        print(f"     Sentence: {edge['sentence'][:100]}...")
except Exception as e:
    print(f"Error reading edges.jsonl: {e}")

# Count entities by type
print("\n📈 ENTITY TYPE BREAKDOWN:")
try:
    entity_types = Counter()
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entity = json.loads(line)
            entity_types[entity['type']] += 1
    
    print("Entity types and counts:")
    for entity_type, count in sorted(entity_types.items()):
        print(f"  {entity_type}: {count}")
    
    print(f"\nTotal unique entity types: {len(entity_types)}")
    
except Exception as e:
    print(f"Error analyzing entity types: {e}")

# Count extraction methods
print("\n🔧 EXTRACTION METHOD BREAKDOWN:")
try:
    extraction_methods = Counter()
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entity = json.loads(line)
            extraction_methods[entity['extraction_method']] += 1
    
    print("Extraction methods:")
    for method, count in extraction_methods.items():
        print(f"  {method}: {count}")
    
except Exception as e:
    print(f"Error analyzing extraction methods: {e}")

# Show some meaningful entities
print("\n🎯 MEANINGFUL ENTITIES EXAMPLES:")
try:
    meaningful_entities = []
    with open("G-Indexation/Graph_fragments/nodes.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entity = json.loads(line)
            # Filter out common words and show meaningful entities
            if (entity['type'] in ['PERSON', 'ORG', 'WORK_OF_ART', 'FAC', 'GPE'] and 
                len(entity['entity']) > 3):
                meaningful_entities.append(entity)
                if len(meaningful_entities) >= 10:
                    break
    
    print("Sample meaningful entities:")
    for i, entity in enumerate(meaningful_entities, 1):
        print(f"  {i}. {entity['entity']} ({entity['type']})")
    
except Exception as e:
    print(f"Error finding meaningful entities: {e}")

print("\n" + "=" * 60)
print("✅ Analysis complete!") 