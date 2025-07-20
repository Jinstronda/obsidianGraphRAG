"""
Quick Neo4j Connection Test with Known Password
"""

from neo4j import GraphDatabase

def test_connection():
    """Test Neo4j connection with the password we just set."""
    
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "88888888"  # The password we just set
    
    print("🔍 Quick Neo4j Connection Test")
    print("=" * 40)
    print(f"URI: {uri}")
    print(f"Username: {username}")
    print(f"Password: {password}")
    print()
    
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print("✅ Connection successful!")
                print("🎉 Neo4j is running and accessible!")
                return True
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Possible issues:")
        print("1. Neo4j database is not started")
        print("2. Database is not running on port 7687")
        print("3. Authentication failed")
        print("\n💡 Try opening Neo4j Desktop and starting your database")
        return False
    
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    test_connection() 