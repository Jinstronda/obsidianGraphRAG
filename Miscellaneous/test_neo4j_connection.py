"""
Neo4j Connection Test Script
Use this to test your Neo4j database connection and help with setup.
"""

import sys
import os

def test_neo4j_connection():
    """Test Neo4j connection and provide setup guidance."""
    
    print("🔍 Neo4j Connection Test")
    print("=" * 50)
    
    # Check if neo4j package is installed
    try:
        from neo4j import GraphDatabase
        print("✅ neo4j Python package is installed")
    except ImportError:
        print("❌ neo4j Python package not found")
        print("Installing neo4j package...")
        os.system("pip install neo4j")
        try:
            from neo4j import GraphDatabase
            print("✅ neo4j Python package installed successfully")
        except ImportError:
            print("❌ Failed to install neo4j package")
            return False
    
    # Connection parameters
    uri = "bolt://localhost:7687"
    username = "neo4j"
    
    print(f"\n🔗 Testing connection to: {uri}")
    print(f"Username: {username}")
    
    # Get password from user
    password = input("Enter your Neo4j password: ")
    
    try:
        # Test connection
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record and record["test"] == 1:
                print("✅ Connection successful!")
                
                # Get Neo4j version
                version_result = session.run("CALL dbms.components() YIELD name, versions, edition")
                for record in version_result:
                    if record["name"] == "Neo4j Kernel":
                        print(f"✅ Neo4j version: {record['versions'][0]} ({record['edition']})")
                        break
                
                # Test basic operations
                print("\n🧪 Testing basic operations...")
                
                # Create a test node
                session.run("CREATE (n:Test {name: 'GraphRAG Test'})")
                print("✅ Created test node")
                
                # Query the test node
                result = session.run("MATCH (n:Test) RETURN n.name as name")
                record = result.single()
                if record:
                    print(f"✅ Retrieved test node: {record['name']}")
                
                # Clean up test node
                session.run("MATCH (n:Test) DELETE n")
                print("✅ Cleaned up test node")
                
                print("\n🎉 Neo4j is ready for GraphRAG!")
                print("You can now load your extracted entities and relations.")
                
                return True
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Neo4j Desktop is running")
        print("2. Check that your database is started (green status)")
        print("3. Verify the password is correct")
        print("4. Check if port 7687 is available")
        print("5. Try opening http://localhost:7474 in your browser")
        return False
    
    finally:
        if 'driver' in locals():
            driver.close()

def setup_neo4j_guide():
    """Provide setup guidance."""
    print("\n📋 Neo4j Setup Guide:")
    print("=" * 50)
    print("1. Open Neo4j Desktop from Start Menu")
    print("2. Click 'New' → 'Create a Local Graph'")
    print("3. Choose Neo4j 5.x")
    print("4. Name: 'GraphRAG_DB'")
    print("5. Set a password (remember it!)")
    print("6. Click 'Start' on your database")
    print("7. Wait for green status")
    print("8. Run this script again to test connection")

if __name__ == "__main__":
    print("🚀 Neo4j Setup and Connection Test")
    print("=" * 60)
    
    # Check if user wants setup guide
    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        setup_neo4j_guide()
    else:
        # Test connection
        success = test_neo4j_connection()
        
        if not success:
            print("\n💡 Need help setting up Neo4j?")
            print("Run: python test_neo4j_connection.py --setup") 