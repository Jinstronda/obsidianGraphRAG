import json

# Load OpenIE triples
with open('G-Indexation/Graph_fragments/openie_triples.json', 'r', encoding='utf-8') as f:
    triples = json.load(f)

print("ALL TRIPLES FROM 100 CHUNKS")
print("=" * 80)
print(f"Total triples extracted: {len(triples)}")
print("=" * 80)

# Group triples by extraction method
methods = {}
for triple in triples:
    method = triple['extraction_method']
    if method not in methods:
        methods[method] = []
    methods[method].append(triple)

# Show all triples by method
for method, method_triples in methods.items():
    print(f"\n{method.upper()} ({len(method_triples)} triples):")
    print("-" * 60)
    
    for i, triple in enumerate(method_triples, 1):
        # Truncate long objects for readability
        obj = triple['object']
        if len(obj) > 80:
            obj = obj[:77] + "..."
        
        # Add entity types if available
        subject_info = triple['subject']
        object_info = obj
        
        if 'subject_type' in triple and 'object_type' in triple:
            subject_info = f"[{triple['subject_type']}] {triple['subject']}"
            object_info = f"[{triple['object_type']}] {obj}"
        
        print(f"{i:3d}. {subject_info:30s} --{triple['predicate']:15s}--> {object_info}")
        print(f"     Source: {triple['source_chunk']} (confidence: {triple['confidence']})")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Count by method
for method, method_triples in methods.items():
    print(f"{method}: {len(method_triples)} triples")

# Most common predicates
predicates = {}
for triple in triples:
    pred = triple['predicate']
    predicates[pred] = predicates.get(pred, 0) + 1

print(f"\nMost common predicates:")
for pred, count in sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {pred}: {count}")

# Most common entities
entities = {}
for triple in triples:
    subj = triple['subject']
    obj = triple['object']
    entities[subj] = entities.get(subj, 0) + 1
    entities[obj] = entities.get(obj, 0) + 1

print(f"\nMost common entities:")
for entity, count in sorted(entities.items(), key=lambda x: x[1], reverse=True)[:15]:
    print(f"  {entity}: {count}")

print(f"\nTotal unique entities: {len(entities)}")
print(f"Total unique predicates: {len(predicates)}") 