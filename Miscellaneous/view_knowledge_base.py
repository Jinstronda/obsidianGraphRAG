"""
Detailed Knowledge Base Viewer
This script allows you to explore specific parts of your GraphRAG knowledge base.
"""

import json
from collections import Counter, defaultdict
import os
import re

class KnowledgeBaseViewer:
    """Interactive viewer for GraphRAG knowledge base."""
    
    def __init__(self):
        self.nodes_file = "G-Indexation/Graph_fragments/nodes.jsonl"
        self.edges_file = "G-Indexation/Graph_fragments/edges.jsonl"
        self.chunks_file = "G-Indexation/Graph_fragments/chunks.json"
        self.triples_file = "G-Indexation/Graph_fragments/openie_triples.json"
        
        # Load data into memory for fast searching
        self.entities = []
        self.relations = []
        self.chunks = []
        self.triples = []
        
        self.load_data()
    
    def load_data(self):
        """Load all data files into memory."""
        print("📥 Loading knowledge base data...")
        
        # Load entities
        if os.path.exists(self.nodes_file):
            with open(self.nodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.entities.append(json.loads(line))
        
        # Load relations (first 10000 for performance)
        if os.path.exists(self.edges_file):
            with open(self.edges_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 10000:  # Limit for performance
                        break
                    self.relations.append(json.loads(line))
        
        # Load chunks
        if os.path.exists(self.chunks_file):
            with open(self.chunks_file, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        
        # Load triples
        if os.path.exists(self.triples_file):
            with open(self.triples_file, 'r', encoding='utf-8') as f:
                self.triples = json.load(f)
        
        print(f"✅ Loaded {len(self.entities)} entities, {len(self.relations)} relations, {len(self.chunks)} chunks, {len(self.triples)} triples")
    
    def search_entities(self, query, entity_type=None, source_file=None):
        """Search for entities by name, type, or source."""
        results = []
        query_lower = query.lower()
        
        for entity in self.entities:
            # Check if entity matches search criteria
            matches_query = query_lower in entity['entity'].lower()
            matches_type = entity_type is None or entity['type'] == entity_type
            matches_source = source_file is None or entity['source_file'] == source_file
            
            if matches_query and matches_type and matches_source:
                results.append(entity)
        
        return results
    
    def find_entity_relations(self, entity_name):
        """Find all relations involving a specific entity."""
        relations = []
        entity_lower = entity_name.lower()
        
        for relation in self.relations:
            if (entity_lower in relation['entity1'].lower() or 
                entity_lower in relation['entity2'].lower()):
                relations.append(relation)
        
        return relations
    
    def show_entity_details(self, entity_name):
        """Show detailed information about a specific entity."""
        print(f"\n🔍 DETAILS FOR: {entity_name}")
        print("="*60)
        
        # Find all instances of this entity
        instances = self.search_entities(entity_name)
        
        if not instances:
            print(f"❌ Entity '{entity_name}' not found")
            return
        
        # Group by type
        by_type = defaultdict(list)
        for instance in instances:
            by_type[instance['type']].append(instance)
        
        print(f"📊 Found {len(instances)} instances across {len(by_type)} types:")
        
        for entity_type, type_instances in by_type.items():
            print(f"\n🏷️  {entity_type} ({len(type_instances)} instances):")
            for instance in type_instances[:5]:  # Show first 5
                print(f"  • {instance['entity']} (from {instance['source_file']})")
            if len(type_instances) > 5:
                print(f"  ... and {len(type_instances) - 5} more")
        
        # Find relations
        relations = self.find_entity_relations(entity_name)
        print(f"\n🔗 Relations ({len(relations)} found):")
        
        # Group relations by other entity
        related_entities = defaultdict(list)
        for relation in relations:
            if entity_name.lower() in relation['entity1'].lower():
                other_entity = relation['entity2']
            else:
                other_entity = relation['entity1']
            related_entities[other_entity].append(relation)
        
        # Show top related entities
        top_related = sorted(related_entities.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        
        for other_entity, entity_relations in top_related:
            print(f"  • {other_entity}: {len(entity_relations)} connections")
            # Show sample sentence
            if entity_relations:
                sample = entity_relations[0]
                print(f"    Example: {sample['sentence'][:100]}...")
    
    def show_source_summary(self, source_file):
        """Show summary of entities and relations from a specific source."""
        print(f"\n📚 SOURCE SUMMARY: {source_file}")
        print("="*60)
        
        # Count entities by type
        entity_types = Counter()
        source_entities = []
        
        for entity in self.entities:
            if entity['source_file'] == source_file:
                entity_types[entity['type']] += 1
                source_entities.append(entity)
        
        print(f"📊 Total entities: {len(source_entities)}")
        print(f"🏷️  Entity types:")
        for entity_type, count in entity_types.most_common():
            print(f"  {entity_type}: {count}")
        
        # Count relations
        source_relations = [r for r in self.relations if r['source_file'] == source_file]
        print(f"\n🔗 Relations: {len(source_relations)}")
        
        # Show sample entities
        print(f"\n📝 Sample entities:")
        for entity_type in ['PERSON', 'WORK_OF_ART', 'NOUN']:
            type_entities = [e for e in source_entities if e['type'] == entity_type]
            if type_entities:
                print(f"\n  {entity_type}:")
                for entity in type_entities[:5]:
                    print(f"    • {entity['entity']}")
    
    def show_top_entities(self, entity_type=None, limit=20):
        """Show most frequent entities."""
        print(f"\n🏆 TOP ENTITIES{f' ({entity_type})' if entity_type else ''}")
        print("="*60)
        
        # Count entities
        entity_counts = Counter()
        for entity in self.entities:
            if entity_type is None or entity['type'] == entity_type:
                entity_counts[entity['entity']] += 1
        
        print(f"📊 Showing top {limit} entities:")
        for entity, count in entity_counts.most_common(limit):
            print(f"  {entity}: {count} occurrences")
    
    def show_entity_network(self, entity_name, depth=1):
        """Show network of entities connected to a specific entity."""
        print(f"\n🌐 NETWORK FOR: {entity_name}")
        print("="*60)
        
        # Find direct relations
        direct_relations = self.find_entity_relations(entity_name)
        
        if not direct_relations:
            print(f"❌ No relations found for '{entity_name}'")
            return
        
        # Get connected entities
        connected_entities = set()
        for relation in direct_relations:
            if entity_name.lower() in relation['entity1'].lower():
                connected_entities.add(relation['entity2'])
            else:
                connected_entities.add(relation['entity1'])
        
        print(f"🔗 Direct connections ({len(connected_entities)} entities):")
        for connected_entity in sorted(connected_entities)[:20]:  # Show first 20
            print(f"  • {connected_entity}")
        
        if len(connected_entities) > 20:
            print(f"  ... and {len(connected_entities) - 20} more")
    
    def interactive_menu(self):
        """Interactive menu for exploring the knowledge base."""
        while True:
            print("\n" + "="*60)
            print("🔍 GRAPHRAG KNOWLEDGE BASE EXPLORER")
            print("="*60)
            print("1. Search for entities")
            print("2. Show entity details")
            print("3. Show source summary")
            print("4. Show top entities")
            print("5. Show entity network")
            print("6. Show all sources")
            print("7. Show entity types")
            print("8. Exit")
            print("="*60)
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == '1':
                query = input("Enter search term: ").strip()
                entity_type = input("Entity type (optional, press Enter to skip): ").strip() or None
                results = self.search_entities(query, entity_type)
                print(f"\n📊 Found {len(results)} entities:")
                for entity in results[:10]:
                    print(f"  • {entity['entity']} ({entity['type']}) - {entity['source_file']}")
                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more")
            
            elif choice == '2':
                entity_name = input("Enter entity name: ").strip()
                self.show_entity_details(entity_name)
            
            elif choice == '3':
                source_file = input("Enter source file name: ").strip()
                self.show_source_summary(source_file)
            
            elif choice == '4':
                entity_type = input("Entity type (optional, press Enter for all): ").strip() or None
                self.show_top_entities(entity_type)
            
            elif choice == '5':
                entity_name = input("Enter entity name: ").strip()
                self.show_entity_network(entity_name)
            
            elif choice == '6':
                sources = set(entity['source_file'] for entity in self.entities)
                print(f"\n📚 All sources ({len(sources)}):")
                for source in sorted(sources):
                    count = len([e for e in self.entities if e['source_file'] == source])
                    print(f"  • {source}: {count} entities")
            
            elif choice == '7':
                entity_types = Counter(entity['type'] for entity in self.entities)
                print(f"\n🏷️  Entity types:")
                for entity_type, count in entity_types.most_common():
                    print(f"  {entity_type}: {count}")
            
            elif choice == '8':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-8.")

def main():
    """Main function."""
    print("🚀 GraphRAG Knowledge Base Viewer")
    print("="*60)
    
    viewer = KnowledgeBaseViewer()
    viewer.interactive_menu()

if __name__ == "__main__":
    main() 