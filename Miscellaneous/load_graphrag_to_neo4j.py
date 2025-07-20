"""
Load GraphRAG Data into Neo4j
This script loads the extracted entities and relations from our modular extractor into Neo4j.
"""

import json
from neo4j import GraphDatabase
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAGLoader:
    """Load GraphRAG data into Neo4j database."""
    
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="88888888"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.entity_counter = defaultdict(int)
        self.relation_counter = defaultdict(int)
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def test_connection(self):
        """Test the Neo4j connection."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("✅ Neo4j connection successful!")
                    return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def clear_database(self):
        """Clear existing GraphRAG data from database."""
        try:
            with self.driver.session() as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("🗑️  Cleared existing data from database")
        except Exception as e:
            logger.error(f"❌ Failed to clear database: {e}")
    
    def create_constraints_and_indexes(self):
        """Create constraints and indexes for better performance."""
        try:
            with self.driver.session() as session:
                # Create constraints for unique entities
                session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
                
                # Create indexes for common properties
                session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)")
                session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)")
                session.run("CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)")
                
                logger.info("📊 Created constraints and indexes")
        except Exception as e:
            logger.warning(f"⚠️  Some constraints/indexes may already exist: {e}")
    
    def load_entities(self, nodes_file="../G-Indexation/Graph_fragments/nodes.jsonl"):
        """Load entities from JSONL file into Neo4j."""
        logger.info(f"📥 Loading entities from {nodes_file}")
        
        try:
            with self.driver.session() as session:
                with open(nodes_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i % 1000 == 0:
                            logger.info(f"Processed {i} entities...")
                        
                        entity_data = json.loads(line)
                        
                        # Create entity node
                        query = """
                        CREATE (e:Entity {
                            id: $id,
                            name: $name,
                            type: $type,
                            chunk_id: $chunk_id,
                            source_file: $source_file,
                            extraction_method: $extraction_method
                        })
                        """
                        
                        session.run(query, {
                            'id': f"{entity_data['entity']}_{entity_data['chunk_id']}",
                            'name': entity_data['entity'],
                            'type': entity_data['type'],
                            'chunk_id': entity_data['chunk_id'],
                            'source_file': entity_data['source_file'],
                            'extraction_method': entity_data['extraction_method']
                        })
                        
                        self.entity_counter[entity_data['type']] += 1
                
                logger.info(f"✅ Loaded {sum(self.entity_counter.values())} entities")
                
        except Exception as e:
            logger.error(f"❌ Failed to load entities: {e}")
            raise
    
    def load_relations(self, edges_file="../G-Indexation/Graph_fragments/edges.jsonl"):
        """Load relations from JSONL file into Neo4j."""
        logger.info(f"🔗 Loading relations from {edges_file}")
        
        try:
            with self.driver.session() as session:
                with open(edges_file, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i % 5000 == 0:
                            logger.info(f"Processed {i} relations...")
                        
                        edge_data = json.loads(line)
                        
                        # Create relationship between entities
                        query = """
                        MATCH (e1:Entity {name: $entity1})
                        MATCH (e2:Entity {name: $entity2})
                        CREATE (e1)-[r:RELATES_TO {
                            type: $relation_type,
                            chunk_id: $chunk_id,
                            source_file: $source_file,
                            sentence: $sentence,
                            extraction_method: $extraction_method
                        }]->(e2)
                        """
                        
                        session.run(query, {
                            'entity1': edge_data['entity1'],
                            'entity2': edge_data['entity2'],
                            'relation_type': 'cooccurrence',
                            'chunk_id': edge_data['chunk_id'],
                            'source_file': edge_data['source_file'],
                            'sentence': edge_data['sentence'],
                            'extraction_method': edge_data['extraction_method']
                        })
                        
                        self.relation_counter['cooccurrence'] += 1
                
                logger.info(f"✅ Loaded {sum(self.relation_counter.values())} relations")
                
        except Exception as e:
            logger.error(f"❌ Failed to load relations: {e}")
            raise
    
    def create_entity_summary(self):
        """Create summary nodes for entity types."""
        try:
            with self.driver.session() as session:
                for entity_type, count in self.entity_counter.items():
                    query = """
                    CREATE (s:Summary {
                        type: 'entity_type',
                        name: $entity_type,
                        count: $count
                    })
                    """
                    session.run(query, {'entity_type': entity_type, 'count': count})
                
                logger.info("📊 Created entity type summaries")
        except Exception as e:
            logger.error(f"❌ Failed to create summaries: {e}")
    
    def run_analysis_queries(self):
        """Run analysis queries on the loaded data."""
        logger.info("🔍 Running analysis queries...")
        
        try:
            with self.driver.session() as session:
                # Count total nodes and relationships
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()["node_count"]
                
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                
                # Get top entities by type
                result = session.run("""
                    MATCH (n:Entity)
                    RETURN n.type as type, count(n) as count
                    ORDER BY count DESC
                    LIMIT 10
                """)
                
                top_types = [(record["type"], record["count"]) for record in result]
                
                # Get most connected entities
                result = session.run("""
                    MATCH (n:Entity)-[r:RELATES_TO]-()
                    RETURN n.name as name, count(r) as connections
                    ORDER BY connections DESC
                    LIMIT 10
                """)
                
                top_entities = [(record["name"], record["connections"]) for record in result]
                
                print("\n" + "="*60)
                print("GRAPH ANALYSIS RESULTS")
                print("="*60)
                print(f"Total nodes: {node_count}")
                print(f"Total relationships: {rel_count}")
                
                print(f"\nTop entity types:")
                for entity_type, count in top_types:
                    print(f"  {entity_type}: {count}")
                
                print(f"\nMost connected entities:")
                for entity, connections in top_entities:
                    print(f"  {entity}: {connections} connections")
                
                print("="*60)
                
        except Exception as e:
            logger.error(f"❌ Failed to run analysis: {e}")
    
    def load_all_data(self, clear_existing=True):
        """Load all GraphRAG data into Neo4j."""
        logger.info("🚀 Starting GraphRAG data load into Neo4j")
        
        # Test connection first
        if not self.test_connection():
            logger.error("❌ Cannot connect to Neo4j. Please start the database.")
            return False
        
        try:
            # Clear existing data if requested
            if clear_existing:
                self.clear_database()
            
            # Create constraints and indexes
            self.create_constraints_and_indexes()
            
            # Load entities
            self.load_entities()
            
            # Load relations
            self.load_relations()
            
            # Create summaries
            self.create_entity_summary()
            
            # Run analysis
            self.run_analysis_queries()
            
            logger.info("🎉 GraphRAG data successfully loaded into Neo4j!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False

def main():
    """Main function to load GraphRAG data into Neo4j."""
    print("🚀 GraphRAG to Neo4j Data Loader")
    print("="*60)
    
    # Initialize loader
    loader = GraphRAGLoader()
    
    try:
        # Load all data
        success = loader.load_all_data()
        
        if success:
            print("\n✅ GraphRAG knowledge graph successfully created in Neo4j!")
            print("🔗 Access your graph at: http://localhost:7474")
            print("📊 Username: neo4j")
            print("🔑 Password: 88888888")
        else:
            print("\n❌ Failed to load data. Check the logs above.")
            
    finally:
        loader.close()

if __name__ == "__main__":
    main() 