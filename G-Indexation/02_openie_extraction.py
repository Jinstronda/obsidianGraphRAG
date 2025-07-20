"""
Step 2: Open Information Extraction (OpenIE)
This module implements OpenIE-style triple extraction for GraphRAG using spaCy and custom rules.
"""

import json
import spacy
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict, Counter
import re

class OpenIEExtractor:
    def __init__(self, chunks_path: str = "G-Indexation/Graph_fragments/chunks.json"):
        """Initialize the OpenIE extraction process."""
        self.chunks_path = chunks_path
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("❌ spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
        # Triple storage
        self.triples = []
        self.entity_counter = Counter()
        self.predicate_counter = Counter()
        
        # Common verb patterns for relationships
        self.relationship_verbs = {
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'has', 'have', 'had', 'contains', 'includes',
            'creates', 'creates', 'makes', 'produces',
            'discovers', 'finds', 'identifies',
            'writes', 'authors', 'composes',
            'translates', 'interprets', 'explains',
            'studies', 'researches', 'investigates',
            'believes', 'thinks', 'considers',
            'says', 'states', 'mentions', 'describes'
        }
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_triples_from_text(self, text: str, chunk_id: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract triples (subject, predicate, object) from text using OpenIE approach."""
        doc = self.nlp(text)
        triples = []
        
        # Extract triples from sentences
        for sent in doc.sents:
            sent_triples = self.extract_triples_from_sentence(sent, chunk_id, source_file)
            triples.extend(sent_triples)
        
        return triples
    
    def extract_triples_from_sentence(self, sent, chunk_id: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract triples from a single sentence."""
        triples = []
        
        # Find the root verb (predicate)
        root_verb = None
        for token in sent:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                root_verb = token
                break
        
        if not root_verb:
            return triples
        
        # Find subject and object
        subject = None
        obj = None
        
        for token in sent:
            if token.dep_ == 'nsubj' and token.head == root_verb:
                subject = self.get_entity_span(token)
            elif token.dep_ == 'dobj' and token.head == root_verb:
                obj = self.get_entity_span(token)
            elif token.dep_ == 'pobj' and token.head.dep_ == 'prep' and token.head.head == root_verb:
                obj = self.get_entity_span(token)
        
        # Create triple if we have subject and object
        if subject and obj:
            triple = {
                'id': f"{chunk_id}_triple_{len(triples)}",
                'subject': subject,
                'predicate': root_verb.text,
                'object': obj,
                'source_chunk': chunk_id,
                'source_file': source_file,
                'confidence': 0.8,
                'extraction_method': 'dependency_parsing'
            }
            triples.append(triple)
            
            # Update counters
            self.entity_counter[subject] += 1
            self.entity_counter[obj] += 1
            self.predicate_counter[root_verb.text] += 1
        
        # Extract copula triples (X is Y)
        copula_triples = self.extract_copula_triples(sent, chunk_id, source_file)
        triples.extend(copula_triples)
        
        # Extract possession triples (X has Y)
        possession_triples = self.extract_possession_triples(sent, chunk_id, source_file)
        triples.extend(possession_triples)
        
        return triples
    
    def get_entity_span(self, token) -> str:
        """Get the full entity span starting from a token."""
        # Find the head of the noun phrase
        head = token
        while head.head.pos_ in ['NOUN', 'PROPN', 'ADJ'] and head.head.dep_ in ['compound', 'amod']:
            head = head.head
        
        # Get the full span
        span_start = head.left_edge.i
        span_end = head.right_edge.i + 1
        
        return ' '.join([token.text for token in token.doc[span_start:span_end]])
    
    def extract_copula_triples(self, sent, chunk_id: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract copula triples (X is Y)."""
        triples = []
        
        for token in sent:
            if token.dep_ == 'ROOT' and token.lemma_ in ['be', 'become', 'remain', 'stay']:
                subject = None
                complement = None
                
                for child in token.children:
                    if child.dep_ == 'nsubj':
                        subject = self.get_entity_span(child)
                    elif child.dep_ in ['attr', 'acomp']:
                        complement = self.get_entity_span(child)
                
                if subject and complement:
                    triple = {
                        'id': f"{chunk_id}_copula_{len(triples)}",
                        'subject': subject,
                        'predicate': token.text,
                        'object': complement,
                        'source_chunk': chunk_id,
                        'source_file': source_file,
                        'confidence': 0.9,
                        'extraction_method': 'copula_extraction'
                    }
                    triples.append(triple)
                    
                    self.entity_counter[subject] += 1
                    self.entity_counter[complement] += 1
                    self.predicate_counter[token.text] += 1
        
        return triples
    
    def extract_possession_triples(self, sent, chunk_id: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract possession triples (X has Y)."""
        triples = []
        
        for token in sent:
            if token.dep_ == 'ROOT' and token.lemma_ in ['have', 'contain', 'include', 'possess']:
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ == 'nsubj':
                        subject = self.get_entity_span(child)
                    elif child.dep_ == 'dobj':
                        obj = self.get_entity_span(child)
                
                if subject and obj:
                    triple = {
                        'id': f"{chunk_id}_possession_{len(triples)}",
                        'subject': subject,
                        'predicate': token.text,
                        'object': obj,
                        'source_chunk': chunk_id,
                        'source_file': source_file,
                        'confidence': 0.85,
                        'extraction_method': 'possession_extraction'
                    }
                    triples.append(triple)
                    
                    self.entity_counter[subject] += 1
                    self.entity_counter[obj] += 1
                    self.predicate_counter[token.text] += 1
        
        return triples
    
    def extract_named_entity_triples(self, text: str, chunk_id: str, source_file: str) -> List[Dict[str, Any]]:
        """Extract triples involving named entities."""
        doc = self.nlp(text)
        triples = []
        
        # Find named entities
        entities = list(doc.ents)
        
        # Create triples between entities that appear in the same sentence
        for sent in doc.sents:
            sent_entities = [ent for ent in entities if ent.sent == sent]
            
            if len(sent_entities) >= 2:
                # Find the main verb in the sentence
                main_verb = None
                for token in sent:
                    if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                        main_verb = token.text
                        break
                
                if main_verb:
                    # Create triples between entities
                    for i, ent1 in enumerate(sent_entities):
                        for ent2 in sent_entities[i+1:]:
                            triple = {
                                'id': f"{chunk_id}_ner_{len(triples)}",
                                'subject': ent1.text,
                                'predicate': main_verb,
                                'object': ent2.text,
                                'source_chunk': chunk_id,
                                'source_file': source_file,
                                'confidence': 0.7,
                                'extraction_method': 'named_entity_extraction',
                                'subject_type': ent1.label_,
                                'object_type': ent2.label_
                            }
                            triples.append(triple)
                            
                            self.entity_counter[ent1.text] += 1
                            self.entity_counter[ent2.text] += 1
                            self.predicate_counter[main_verb] += 1
        
        return triples
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks to extract triples."""
        print(f"Processing {len(chunks)} chunks with OpenIE...")
        print("-" * 50)
        
        all_triples = []
        
        for i, chunk in enumerate(chunks):
            # Extract dependency-based triples
            triples = self.extract_triples_from_text(
                chunk['text'], 
                chunk['chunk_id'], 
                chunk['source_file']
            )
            
            # Extract named entity triples
            ner_triples = self.extract_named_entity_triples(
                chunk['text'],
                chunk['chunk_id'],
                chunk['source_file']
            )
            
            all_triples.extend(triples)
            all_triples.extend(ner_triples)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1} chunks...")
        
        print("-" * 50)
        print(f"Extraction complete!")
        print(f"Total triples: {len(all_triples)}")
        
        return all_triples
    
    def analyze_results(self, triples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the extraction results."""
        # Triple analysis
        extraction_methods = Counter()
        predicates = Counter()
        
        for triple in triples:
            extraction_methods[triple['extraction_method']] += 1
            predicates[triple['predicate']] += 1
        
        # Most frequent entities
        top_entities = self.entity_counter.most_common(10)
        
        # Most frequent predicates
        top_predicates = self.predicate_counter.most_common(10)
        
        return {
            'total_triples': len(triples),
            'extraction_methods': dict(extraction_methods),
            'predicates': dict(predicates),
            'top_entities': top_entities,
            'top_predicates': top_predicates
        }
    
    def save_results(self, triples: List[Dict[str, Any]]):
        """Save the extracted triples."""
        # Save triples
        triples_path = Path("G-Indexation/Graph_fragments/openie_triples.json")
        triples_path.parent.mkdir(parents=True, exist_ok=True)
        with open(triples_path, 'w', encoding='utf-8') as f:
            json.dump(triples, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {triples_path}")
    
    def run(self, max_chunks: int = 100):
        """Run the OpenIE extraction."""
        print("🚀 Starting Open Information Extraction")
        print("=" * 60)
        
        # Load chunks
        chunks = self.load_chunks()
        
        # Limit to max_chunks for testing
        test_chunks = chunks[:max_chunks]
        print(f"Testing with first {len(test_chunks)} chunks out of {len(chunks)} total")
        
        # Process chunks
        triples = self.process_chunks(test_chunks)
        
        # Analyze results
        analysis = self.analyze_results(triples)
        
        # Save results
        self.save_results(triples)
        
        # Print analysis
        print("\n📊 EXTRACTION ANALYSIS:")
        print(f"Total triples: {analysis['total_triples']}")
        
        print("\nExtraction Methods:")
        for method, count in analysis['extraction_methods'].items():
            print(f"  {method}: {count}")
        
        print("\nTop 10 Predicates:")
        for predicate, count in analysis['top_predicates']:
            print(f"  {predicate}: {count}")
        
        print("\nTop 10 Entities:")
        for entity, count in analysis['top_entities']:
            print(f"  {entity}: {count}")
        
        print("\nSample Triples:")
        for i, triple in enumerate(triples[:10]):
            print(f"  {i+1:2d}. {triple['subject']} --{triple['predicate']}--> {triple['object']}")
        
        print("\n✅ OpenIE extraction completed!")

def main():
    """Main function to run the OpenIE extraction."""
    extractor = OpenIEExtractor()
    extractor.run(max_chunks=100)  # Test with 100 chunks

if __name__ == "__main__":
    main() 