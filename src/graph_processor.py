import json
import os
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphProcessor:
    """
    Process RAG-Anything vector database into a 3D graph structure
    Uses PCA for dimensionality reduction and K-Means for clustering
    """
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.vdb_chunks_path = self.storage_dir / "vdb_chunks.json"
        self.output_path = self.storage_dir / "brain_graph_data.json"
        
    def load_data(self) -> List[Dict]:
        """Load chunks and embeddings from VDB"""
        if not self.vdb_chunks_path.exists():
            logger.error(f"VDB chunks file not found: {self.vdb_chunks_path}")
            return []
            
        try:
            with open(self.vdb_chunks_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures based on inspection
            if isinstance(data, dict) and 'data' in data:
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                logger.error("Unknown VDB chunks format")
                return []
        except Exception as e:
            logger.error(f"Error loading VDB chunks: {e}")
            return []

    def process_graph(self, n_clusters: int = 8) -> Dict[str, Any]:
        """
        Process embeddings into 3D coordinates
        
        Args:
            n_clusters: Number of clusters for coloring
            
        Returns:
            Dictionary containing nodes and metadata
        """
        try:
            # Lazy import to avoid startup overhead if not used
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn not installed. Cannot process graph.")
            return {"error": "scikit-learn not installed"}

        chunks = self.load_data()
        if not chunks:
            return {"nodes": [], "error": "No data found"}
            
        # Extract embeddings and metadata
        embeddings = []
        valid_chunks = []
        
        for chunk in chunks:
            if 'vector' in chunk and chunk['vector']:
                vec = chunk['vector']
                if isinstance(vec, str):
                    try:
                        import base64
                        # Try float32 first as it's most common for embeddings
                        vec_arr = np.frombuffer(base64.b64decode(vec), dtype=np.float32)
                        vec = vec_arr.tolist()
                    except Exception as e:
                        logger.warning(f"Failed to decode vector for chunk {chunk.get('__id__')}: {e}")
                        continue
                
                # Ensure it's a list
                if hasattr(vec, 'tolist'):
                    vec = vec.tolist()
                
                # Validate length (assuming 768 or 1536, but just checking consistency)
                if embeddings and len(vec) != len(embeddings[0]):
                    logger.warning(f"Vector length mismatch: expected {len(embeddings[0])}, got {len(vec)} for chunk {chunk.get('__id__')}")
                    continue
                    
                embeddings.append(vec)
                valid_chunks.append(chunk)
                
        if not embeddings:
            return {"nodes": [], "error": "No embeddings found"}
            
        try:
            X = np.array(embeddings)
            # Handle NaN/Inf
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            logger.error(f"Failed to create numpy array from embeddings: {e}")
            return {"nodes": [], "error": str(e)}
        
        # 1. Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. PCA for 3D layout (x, y, z)
        pca = PCA(n_components=3)
        coords = pca.fit_transform(X_scaled)
        
        # 3. Clustering for colors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # 4. Build graph structure
        nodes = []
        for i, chunk in enumerate(valid_chunks):
            # Normalize coordinates to a reasonable range (e.g., -100 to 100)
            # This helps with 3d-force-graph scaling
            x, y, z = coords[i] * 20 
            
            node = {
                "id": chunk.get('__id__') or f"chunk_{i}",
                "file_path": chunk.get('file_path', 'Unknown'),
                "content": chunk.get('content', '')[:200] + "...", # Preview only
                "group": int(clusters[i]),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "val": 1 # Size
            }
            nodes.append(node)
            
        result = {
            "nodes": nodes,
            "links": [],
            "clusters": n_clusters
        }
        
        # Cache result
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f)
        except Exception as e:
            logger.warning(f"Could not cache graph data: {e}")
            
        return result

if __name__ == "__main__":
    # Test run
    processor = GraphProcessor(r"c:\Users\joaop\Documents\Hobbies\Obsidian Rag\rag_storage")
    result = processor.process_graph()
    print(f"Processed {len(result.get('nodes', []))} nodes")
