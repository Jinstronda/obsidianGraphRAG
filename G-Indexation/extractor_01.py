"""
Modular Entity and Relation Extractor for GraphRAG
This script provides a foundation for extracting entities and relations from text chunks.
Designed to be easily extensible with GLiNER, GLiREL, or LLM-based extractors.
"""

import json
import spacy
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseExtractor:
    """Base class for entity and relation extraction."""
    
    def __init__(self, chunks_path: str = "G-Indexation/Graph_fragments/chunks.json"):
        """Initialize the extractor with chunks path."""
        self.chunks_path = chunks_path
        self.entities = []
        self.edges = []
        self.entity_counter = defaultdict(int)
        self.edge_counter = defaultdict(int)
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from JSON file."""
        try:
            with open(self.chunks_path, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from {self.chunks_path}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            raise
    
    def extract_entities(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from a chunk. To be implemented by subclasses."""
        raise NotImplementedError
    
    def extract_relations(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relations from a chunk. To be implemented by subclasses."""
        raise NotImplementedError
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process all chunks and return entities and edges."""
        raise NotImplementedError
    
    def save_results(self, entities: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """Save entities and edges to JSONL files."""
        # Save entities
        entities_path = Path("G-Indexation/Graph_fragments/nodes.jsonl")
        entities_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(entities_path, 'w', encoding='utf-8') as f:
            for entity in entities:
                f.write(json.dumps(entity, ensure_ascii=False) + '\n')
        
        # Save edges
        edges_path = Path("G-Indexation/Graph_fragments/edges.jsonl")
        with open(edges_path, 'w', encoding='utf-8') as f:
            for edge in edges:
                f.write(json.dumps(edge, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(entities)} entities to {entities_path}")
        logger.info(f"Saved {len(edges)} edges to {edges_path}")
    
    def print_summary(self, entities: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        """Print extraction summary."""
        print("\n" + "="*60)
        print("EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total entities extracted: {len(entities)}")
        print(f"Total edges extracted: {len(edges)}")
        
        # Entity type breakdown
        entity_types = defaultdict(int)
        for entity in entities:
            entity_types[entity['type']] += 1
        
        print(f"\nEntity types:")
        for entity_type, count in sorted(entity_types.items()):
            print(f"  {entity_type}: {count}")
        
        # Most common entities
        print(f"\nTop 10 entities:")
        for entity, count in sorted(self.entity_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {entity}: {count}")
        
        print("="*60)

class SpacyExtractor(BaseExtractor):
    """spaCy-based entity and relation extractor."""
    
    def __init__(self, chunks_path: str = "G-Indexation/Graph_fragments/chunks.json"):
        """Initialize spaCy extractor with best available model."""
        super().__init__(chunks_path)
        self.nlp = self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load the best available spaCy model."""
        models = ['en_core_web_trf', 'en_core_web_lg', 'en_core_web_sm']
        
        for model in models:
            try:
                logger.info(f"Attempting to load spaCy model: {model}")
                nlp = spacy.load(model)
                logger.info(f"✅ Successfully loaded spaCy model: {model}")
                return nlp
            except OSError:
                logger.warning(f"❌ Model {model} not found, trying next...")
                continue
        
        raise RuntimeError("No spaCy models available. Please install one of: en_core_web_trf, en_core_web_lg, en_core_web_sm")
    
    def extract_entities(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from a chunk using spaCy."""
        entities = []
        doc = self.nlp(chunk['text'])
        
        # Extract named entities
        for ent in doc.ents:
            entity = {
                'entity': ent.text,
                'type': ent.label_,
                'chunk_id': chunk['chunk_id'],
                'source_file': chunk['source_file'],
                'start_char': ent.start_char,
                'end_char': ent.end_char,
                'extraction_method': 'ner'
            }
            entities.append(entity)
            self.entity_counter[ent.text] += 1
        
        # Extract noun chunks (avoiding duplicates with NER)
        ner_texts = {ent.text.lower() for ent in doc.ents}
        
        for nc in doc.noun_chunks:
            # Skip if this noun chunk is already covered by NER
            if nc.text.lower() in ner_texts:
                continue
            
            # Filter out very short or common noun chunks
            if len(nc.text.split()) <= 3 and len(nc.text) > 2:
                entity = {
                    'entity': nc.text,
                    'type': 'NOUN',
                    'chunk_id': chunk['chunk_id'],
                    'source_file': chunk['source_file'],
                    'start_char': nc.start_char,
                    'end_char': nc.end_char,
                    'extraction_method': 'noun_chunk'
                }
                entities.append(entity)
                self.entity_counter[nc.text] += 1
        
        return entities
    
    def extract_relations(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract relations from a chunk using spaCy."""
        edges = []
        doc = self.nlp(chunk['text'])
        
        # Get all entities in this chunk
        chunk_entities = self.extract_entities(chunk)
        entity_texts = {entity['entity'] for entity in chunk_entities}
        
        # Process each sentence
        for sent in doc.sents:
            # Find entities that appear in this sentence
            sent_entities = []
            for entity in chunk_entities:
                if (entity['start_char'] >= sent.start_char and 
                    entity['end_char'] <= sent.end_char):
                    sent_entities.append(entity)
            
            # Create edges between all pairs of entities in the sentence
            for i, entity1 in enumerate(sent_entities):
                for entity2 in sent_entities[i+1:]:
                    # Create undirected edge (sort entities alphabetically for consistency)
                    if entity1['entity'] < entity2['entity']:
                        e1, e2 = entity1, entity2
                    else:
                        e1, e2 = entity2, entity1
                    
                    edge = {
                        'entity1': e1['entity'],
                        'entity2': e2['entity'],
                        'chunk_id': chunk['chunk_id'],
                        'source_file': chunk['source_file'],
                        'sentence': sent.text.strip(),
                        'entity1_type': e1['type'],
                        'entity2_type': e2['type'],
                        'extraction_method': 'cooccurrence'
                    }
                    edges.append(edge)
                    
                    # Count edge occurrences
                    edge_key = f"{e1['entity']} <-> {e2['entity']}"
                    self.edge_counter[edge_key] += 1
        
        return edges
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process all chunks and extract entities and relations."""
        logger.info(f"Processing {len(chunks)} chunks with spaCy...")
        
        all_entities = []
        all_edges = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Extract entities
                entities = self.extract_entities(chunk)
                all_entities.extend(entities)
                
                # Extract relations
                edges = self.extract_relations(chunk)
                all_edges.extend(edges)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i+1}/{len(chunks)} chunks...")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk['chunk_id']}: {e}")
                continue
        
        logger.info(f"Extraction complete! Found {len(all_entities)} entities and {len(all_edges)} edges")
        return all_entities, all_edges

class ExtractorFactory:
    """Factory for creating different types of extractors."""
    
    @staticmethod
    def create_extractor(extractor_type: str = "spacy", **kwargs) -> BaseExtractor:
        """Create an extractor of the specified type."""
        if extractor_type.lower() == "spacy":
            return SpacyExtractor(**kwargs)
        elif extractor_type.lower() == "gliner":
            # Future: return GLiNERExtractor(**kwargs)
            raise NotImplementedError("GLiNER extractor not yet implemented")
        elif extractor_type.lower() == "glirel":
            # Future: return GLiRELExtractor(**kwargs)
            raise NotImplementedError("GLiREL extractor not yet implemented")
        elif extractor_type.lower() == "llm":
            # Future: return LLMExtractor(**kwargs)
            raise NotImplementedError("LLM extractor not yet implemented")
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")

def main():
    """Main function to run the extraction pipeline."""
    print("🚀 Starting Entity and Relation Extraction")
    print("="*60)
    
    try:
        # Create extractor (easily swappable)
        extractor = ExtractorFactory.create_extractor("spacy")
        
        # Load chunks
        chunks = extractor.load_chunks()
        
        # Process chunks
        entities, edges = extractor.process_chunks(chunks)
        
        # Save results
        extractor.save_results(entities, edges)
        
        # Print summary
        extractor.print_summary(entities, edges)
        
        print("\n✅ Extraction completed successfully!")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

if __name__ == "__main__":
    main() 