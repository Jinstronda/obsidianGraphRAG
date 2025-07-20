import json

# Load OpenIE triples
with open('G-Indexation/Graph_fragments/openie_triples.json', 'r', encoding='utf-8') as f:
    triples = json.load(f)

print("OPENIE EXTRACTION RESULTS")
print("=" * 50)

print(f"Total triples: {len(triples)}")

# Analysis by extraction method
methods = {}
for triple in triples:
    method = triple['extraction_method']
    methods[method] = methods.get(method, 0) + 1

print("\nExtraction Methods:")
for method, count in methods.items():
    print(f"  {method}: {count}")

# Show sample triples by method
print("\nSample Triples by Method:")
for method in methods.keys():
    method_triples = [t for t in triples if t['extraction_method'] == method]
    print(f"\n{method.upper()} ({len(method_triples)} triples):")
    for i, triple in enumerate(method_triples[:5]):
        print(f"  {i+1}. {triple['subject']} --{triple['predicate']}--> {triple['object']}")

# Show triples with named entities
print("\nTriples with Named Entities:")
ner_triples = [t for t in triples if 'subject_type' in t and 'object_type' in t]
for i, triple in enumerate(ner_triples[:10]):
    print(f"  {i+1}. [{triple['subject_type']}] {triple['subject']} --{triple['predicate']}--> [{triple['object_type']}] {triple['object']}")

# Show copula triples (X is Y)
print("\nCopula Triples (X is Y):")
copula_triples = [t for t in triples if t['extraction_method'] == 'copula_extraction']
for i, triple in enumerate(copula_triples[:10]):
    print(f"  {i+1}. {triple['subject']} --{triple['predicate']}--> {triple['object']}")

# Show possession triples (X has Y)
print("\nPossession Triples (X has Y):")
possession_triples = [t for t in triples if t['extraction_method'] == 'possession_extraction']
for i, triple in enumerate(possession_triples[:10]):
    print(f"  {i+1}. {triple['subject']} --{triple['predicate']}--> {triple['object']}")

# Show dependency-based triples
print("\nDependency-based Triples:")
dep_triples = [t for t in triples if t['extraction_method'] == 'dependency_parsing']
for i, triple in enumerate(dep_triples[:10]):
    print(f"  {i+1}. {triple['subject']} --{triple['predicate']}--> {triple['object']}")

print(f"\nTotal triples analyzed: {len(triples)}") 