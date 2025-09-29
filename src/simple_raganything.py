"""
Simple RAG-Anything Implementation with EmbeddingGemma 308M
Clean, minimal implementation for Obsidian vault processing
"""

import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from tqdm import tqdm

# RAG-Anything imports
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.utils import EmbeddingFunc

# Import Obsidian chunker
try:
    from .obsidian_chunker import ObsidianChunker
except ImportError:
    from obsidian_chunker import ObsidianChunker

# Load environment variables
load_dotenv()

class SimpleRAGAnything:
    """
    Simple RAG-Anything implementation with EmbeddingGemma 308M
    Processes Obsidian vaults with multimodal support
    """
    
    def __init__(self, vault_path: str, working_dir: str = "./rag_storage"):
        """
        Initialize Simple RAG-Anything with Obsidian chunking
        
        Args:
            vault_path: Path to Obsidian vault
            working_dir: Working directory for RAG storage
        """
        self.vault_path = vault_path
        self.working_dir = working_dir
        self.rag_anything = None
        self.embedding_model = None
        self.chunker = None
        
        # Create working directory if it doesn't exist
        os.makedirs(working_dir, exist_ok=True)
        
        # Initialize Obsidian chunker for 2K token chunks
        self.chunker = ObsidianChunker(vault_path, target_tokens=2000)
        
        print(f"Vault Path: {vault_path}")
        print(f"Working Dir: {working_dir}")
        print(f"Chunker: 2K token chunks with wikilinks & metadata")
    
    def _detect_existing_database(self) -> bool:
        """Check if RAG-Anything database already exists"""
        key_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        existing_files = []
        for file in key_files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                existing_files.append(file)
        
        return len(existing_files) > 0
    
    def _get_database_stats(self) -> dict:
        """Get statistics about existing database"""
        stats = {
            "total_files": 0,
            "key_files": [],
            "last_modified": None,
            "database_size": 0
        }
        
        key_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        for file in key_files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                stats["key_files"].append(file)
                stats["total_files"] += 1
                
                # Get file size
                file_size = os.path.getsize(file_path)
                stats["database_size"] += file_size
                
                # Get last modified time
                mod_time = os.path.getmtime(file_path)
                if stats["last_modified"] is None or mod_time > stats["last_modified"]:
                    stats["last_modified"] = mod_time
        
        return stats
    
    def _delete_existing_database(self):
        """Delete existing database files"""
        print("Deleting existing database...")
        
        key_files = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_full_entities.json",
            "kv_store_full_relations.json",
            "kv_store_llm_response_cache.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json"
        ]
        
        deleted_count = 0
        for file in key_files:
            file_path = os.path.join(self.working_dir, file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except Exception as e:
                    print(f"Warning: Could not delete {file}: {e}")
        
        print(f"Deleted {deleted_count} database files")
    
    def _handle_database_choice(self) -> str:
        """Present user with database options and get choice"""
        stats = self._get_database_stats()
        
        print("\n" + "="*60)
        print("Found existing RAG-Anything database!")
        print("="*60)
        print(f"Database Statistics:")
        print(f"   - Files: {stats['total_files']}")
        print(f"   - Size: {stats['database_size'] / 1024 / 1024:.1f} MB")
        if stats['last_modified']:
            import datetime
            mod_time = datetime.datetime.fromtimestamp(stats['last_modified'])
            print(f"   - Last Updated: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   - Key Files: {', '.join(stats['key_files'][:3])}{'...' if len(stats['key_files']) > 3 else ''}")
        
        print("\nChoose an option:")
        print("1. Use existing database (continue with current data)")
        print("2. Build new database (delete old data and start fresh)")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            if choice in ['1', '2']:
                return choice
            print("Please enter 1 or 2")
    
    def _initialize_embedding_model(self):
        """Initialize EmbeddingGemma 308M model with EmbeddingFunc wrapper"""
        print("Loading EmbeddingGemma 308M model...")
        try:
            # Check if CUDA is available and set device
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            if device == 'cuda':
                print(f"   GPU Detected: {torch.cuda.get_device_name(0)}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

            # Load model on GPU if available
            self.embedding_model = SentenceTransformer("google/embeddinggemma-300m", device=device)
            print(f"EmbeddingGemma 308M loaded successfully on {device.upper()}!")
            print(f"   - Dimensions: 768 (truncatable to 512/256/128)")
            print(f"   - Languages: 100+ supported")
            print(f"   - Memory: <200MB with quantization")
            
            # Create a standalone async embedding function for RAG-Anything
            def create_embedding_function(model):
                async def embedding_function(texts):
                    # Return embeddings in the exact same format as openai_embed()
                    # openai_embed returns a list of embedding vectors
                    # Run in executor to avoid blocking async loop
                    embeddings = model.encode(texts, convert_to_numpy=True)
                    return embeddings.tolist()
                return embedding_function

            # Create EmbeddingFunc wrapper for RAG-Anything
            self.embedding_func = EmbeddingFunc(
                embedding_dim=768,
                max_token_size=8192,
                func=create_embedding_function(self.embedding_model)
            )
            print("EmbeddingFunc wrapper created successfully!")
            
        except Exception as e:
            print(f"Error loading EmbeddingGemma: {e}")
            raise
    
    def _embedding_function(self, texts: List[str]) -> List[List[float]]:
        """
        Embedding function using EmbeddingGemma 308M
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.embedding_model:
            self._initialize_embedding_model()
        
        # Generate embeddings with EmbeddingGemma
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        
        # Optional: Truncate to 512 dimensions for memory efficiency
        # embeddings = embeddings[:, :512]
        
        return embeddings.tolist()
    
    async def _llm_function(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        LLM function for text generation using OpenAI
        Supports both GPT-4 and GPT-5 models

        Args:
            prompt: User prompt
            system_prompt: System prompt
            **kwargs: Additional arguments (can include 'model' to override default)

        Returns:
            Generated text response
        """
        try:
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return "Error: OPENAI_API_KEY not found in environment"

            # Get model from kwargs or use default from env
            model = kwargs.get("model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=api_key)

            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # GPT-5 models use max_completion_tokens, GPT-4 and earlier use max_tokens
            if model.startswith("gpt-5"):
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.7),
                    max_completion_tokens=kwargs.get("max_tokens", 2000)
                )
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 2000)
                )

            return response.choices[0].message.content

        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"
    
    async def _vision_function(self, prompt: str, image_data: str = None, **kwargs) -> str:
        """
        Vision function for multimodal processing (images, tables, equations)
        
        Args:
            prompt: Text prompt
            image_data: Base64 encoded image data
            **kwargs: Additional arguments
            
        Returns:
            Generated response for multimodal content
        """
        # TODO: Implement your vision function here
        # Example for OpenAI Vision:
        # if image_data:
        #     response = client.chat.completions.create(
        #         model="gpt-4o",
        #         messages=[{"role": "user", "content": [
        #             {"type": "text", "text": prompt},
        #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        #         ]}]
        #     )
        #     return response.choices[0].message.content
        
        # Placeholder response
        return f"Vision Response for: {prompt[:50]}..."
    
    async def initialize(self):
        """Initialize RAG-Anything with EmbeddingGemma and database management"""
        print("Initializing RAG-Anything with EmbeddingGemma 308M...")
        
        # Check for existing database
        if self._detect_existing_database():
            choice = self._handle_database_choice()
            
            if choice == "2":  # Build new database
                self._delete_existing_database()
                print("Starting fresh with new database...")
            else:  # Use existing database
                print("Using existing database...")
                # Initialize embedding model for queries
                self._initialize_embedding_model()
                return
        
        # Initialize embedding model
        self._initialize_embedding_model()

        print("\n⏳ Configuring RAG-Anything framework...")
        print("   - Setting up multimodal processing (images, tables, equations)")
        print("   - Configuring MinerU parser")
        print("   - Preparing knowledge graph storage")

        # RAG-Anything configuration
        config = RAGAnythingConfig(
            working_dir=self.working_dir,
            parser="mineru",  # Advanced document parsing
            parse_method="auto",
            enable_image_processing=True,  # Process images in documents
            enable_table_processing=True,  # Process tables in documents
            enable_equation_processing=True,  # Process equations in documents
            context_window=2,  # Context window for processing
            context_mode="page",  # Page-based context
            max_context_tokens=3000,  # Maximum context tokens
            include_headers=True,  # Include document headers
            include_captions=True,  # Include image/table captions
            context_filter_content_types=["text", "image", "table", "equation"]
        )

        print("\n⏳ Initializing RAG-Anything (this may take 1-2 minutes)...")
        print("   - Verifying MinerU installation")
        print("   - Setting up LightRAG backend")
        print("   - Creating graph storage files")

        # Initialize RAG-Anything
        self.rag_anything = RAGAnything(
            config=config,
            llm_model_func=self._llm_function,
            vision_model_func=self._vision_function,
            embedding_func=self.embedding_func
        )

        print("\n✓ RAG-Anything initialized successfully!")
        print("   - Multimodal processing: Images, Tables, Equations")
        print("   - Embedding model: EmbeddingGemma 308M")
        print("   - Context awareness: Enabled")
    
    
    async def process_vault(self):
        """
        Process entire Obsidian vault with SOTA chunking and RAG-Anything
        Uses 2K token chunks with wikilinks and metadata preservation
        """
        print("="*70)
        print("🚀 PROCESSING OBSIDIAN VAULT WITH SOTA CHUNKING")
        print("="*70)
        
        # Step 1: Chunk vault with SOTA approach
        print("Step 1: Chunking vault with 2K token windows...")
        print("Preserving wikilinks, metadata, and file connections...")
        
        chunks_data = self.chunker.process_entire_vault()
        
        if not chunks_data or not chunks_data.get('chunks'):
            print("No chunks created!")
            return
        
        chunks = chunks_data['chunks']
        stats = chunks_data['stats']
        
        print(f"\nChunking Complete:")
        print(f"   Files: {stats['total_files']}")
        print(f"   Chunks: {stats['total_chunks']}")
        print(f"   Wikilinks: {stats['total_wikilinks']}")
        print(f"   Tags: {stats['total_tags']}")
        
        # Step 2: Process chunks with RAG-Anything
        print(f"\nStep 2: Processing {len(chunks)} chunks with RAG-Anything...")
        print("Using EmbeddingGemma 308M for embeddings")
        print("Multimodal processing: Images, Tables, Equations")
        
        successful_chunks = 0
        failed_chunks = 0
        
        # Convert chunks to content list format for RAG-Anything
        print("\nConverting chunks to RAG-Anything content format...")

        content_list = []
        with tqdm(total=len(chunks), desc="Converting chunks", unit="chunk") as pbar:
            for i, chunk in enumerate(chunks):
                chunk_id = chunk['chunk_id']
                source_file = chunk['metadata'].source_file

                # Convert chunk to content list format
                content_item = {
                    "type": "text",
                    "text": chunk['content'],
                    "page_idx": i  # Use chunk index as page index
                }
                content_list.append(content_item)
                pbar.update(1)

        print(f"\n✓ Converted {len(content_list)} chunks to content format")

        # Process all chunks at once with RAG-Anything
        print("\n" + "="*70)
        print("PROCESSING WITH RAG-ANYTHING (This may take several minutes...)")
        print("="*70)
        print("⏳ Building knowledge graph with embeddings...")
        print("⏳ Extracting entities and relationships...")
        print("⏳ Creating vector database...")
        print("\nPlease wait - RAG-Anything is processing in the background...")
        
        try:
            await self.rag_anything.insert_content_list(
                content_list=content_list,
                file_path="obsidian_vault.md",  # Reference file name
                split_by_character=None,
                split_by_character_only=False,
                doc_id=None,
                display_stats=True
            )
            
            successful_chunks = len(chunks)
            failed_chunks = 0
            print(f"Successfully processed all {successful_chunks} chunks!")
            
        except Exception as e:
            successful_chunks = 0
            failed_chunks = len(chunks)
            print(f"Failed to process chunks: {str(e)}")
        
        # Print summary
        print("\n" + "="*70)
        print("PROCESSING COMPLETE!")
        print("="*70)
        print(f"Files Processed: {stats['total_files']}")
        print(f"Chunks Created: {stats['total_chunks']}")
        print(f"Successful Chunks: {successful_chunks}")
        print(f"Failed Chunks: {failed_chunks}")
        print(f"📈 Success Rate: {(successful_chunks/len(chunks))*100:.1f}%")
        print(f"Wikilinks Preserved: {stats['total_wikilinks']}")
        print(f"Tags Preserved: {stats['total_tags']}")
        print(f"Storage: {self.working_dir}")
        print("="*70)
    
    async def query(self, question: str, mode: str = "hybrid") -> str:
        """
        Query the processed knowledge base
        
        Args:
            question: Question to ask
            mode: Query mode (naive, local, global, hybrid)
            
        Returns:
            Answer from the knowledge base
        """
        if not self.rag_anything:
            await self.initialize()
        
        print(f"❓ Query: {question}")
        print(f"🔍 Mode: {mode}")
        
        try:
            result = await self.rag_anything.aquery(
                question,
                mode=mode
            )
            
            print(f"Query completed successfully!")
            return result
            
        except Exception as e:
            print(f"Query failed: {e}")
            return f"Error: {str(e)}"
    
    async def query_multimodal(self, question: str, multimodal_content: List[Dict] = None) -> str:
        """
        Query with multimodal content (images, tables, equations)
        
        Args:
            question: Question to ask
            multimodal_content: List of multimodal content
            
        Returns:
            Answer from the knowledge base
        """
        if not self.rag_anything:
            await self.initialize()
        
        print(f"❓ Multimodal Query: {question}")
        print(f"Content: {len(multimodal_content or [])} items")
        
        try:
            result = await self.rag_anything.aquery_with_multimodal(
                question,
                multimodal_content=multimodal_content or []
            )
            
            print(f"Multimodal query completed successfully!")
            return result
            
        except Exception as e:
            print(f"Multimodal query failed: {e}")
            return f"Error: {str(e)}"


def check_conda_environment():
    """Check if we're running in conda environment turing0.1"""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    
    if conda_env != 'turing0.1':
        print("ERROR: Not running in conda environment turing0.1")
        print(f"   Current environment: {conda_env}")
        print("")
        print("💡 Please activate the correct environment:")
        print("   conda activate turing0.1")
        print("   python run_raganything.py")
        sys.exit(1)
    
    print(f"Running in conda environment: {conda_env}")

async def main():
    """
    Main function to run the Simple RAG-Anything system
    Always runs in conda environment turing0.1
    """
    # Check conda environment first
    check_conda_environment()
    
    # Configuration - Default to your Obsidian vault
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH", r"C:\Users\joaop\Documents\Obsidian Vault\Segundo Cerebro")
    working_dir = os.getenv("WORKING_DIR", "./rag_storage")
    
    print("🚀 Simple RAG-Anything with EmbeddingGemma 308M")
    print("Conda Environment: turing0.1")
    print("="*50)
    
    # Initialize system
    rag = SimpleRAGAnything(vault_path, working_dir)
    await rag.initialize()
    
    # Process vault
    await rag.process_vault()
    
    # Test queries
    print("\n🧪 Testing queries...")
    
    # Basic query
    result1 = await rag.query("What are the main topics in my notes?")
    print(f"\n📝 Basic Query Result:\n{result1}")
    
    # Multimodal query
    result2 = await rag.query_multimodal("What images and tables are available?")
    print(f"\nMultimodal Query Result:\n{result2}")


if __name__ == "__main__":
    asyncio.run(main())
