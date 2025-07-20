# Neo4j Setup Guide for GraphRAG

## Installation Complete ✅
Neo4j Desktop has been successfully installed via winget.

## Setup Steps:

### 1. Launch Neo4j Desktop
- Open from Windows Start Menu
- Or run: `start "Neo4j Desktop"`

### 2. Create Database
1. Click "New" → "Create a Local Graph"
2. Choose Neo4j 5.x
3. Name: "GraphRAG_DB"
4. Set password (remember this!)

### 3. Start Database
1. Click "Start" on your database
2. Wait for green status
3. Click "Open with Neo4j Browser"

### 4. Set Password via Command Line (Alternative)
If you need to set password via command line:

```bash
# Find Neo4j installation
cd "C:\Users\%USERNAME%\AppData\Local\Neo4j Desktop\Application\neo4j-desktop-1.6.2\resources\app.asar.unpacked\bin\neo4j"

# Set password
neo4j-admin set-initial-password your_password
```

### 5. Test Connection
Once running, you can test with:
- Browser: http://localhost:7474
- Username: neo4j
- Password: (what you set)

### 6. GraphRAG Integration
After setup, we'll connect our extracted entities and relations to Neo4j!

## Default Ports:
- HTTP: 7474 (Browser)
- Bolt: 7687 (Applications)

## Troubleshooting:
- If port conflicts: Change ports in database settings
- If password issues: Use neo4j-admin reset-password
- If startup fails: Check logs in Neo4j Desktop 