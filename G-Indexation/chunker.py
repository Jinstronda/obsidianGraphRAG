import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize

class DocumentChunker:
    def __init__(self):
        """Initialize the document chunker with configuration from config.json file."""
        # Load configuration from JSON file
        config_path = Path('config.json')
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.source_path = config.get('source_path', 'G-Indexation/Hermetic_Library')
            self.sentences_per_chunk = config.get('sentences_per_chunk', 5)
            self.overlap_sentences = config.get('overlap_sentences', 2)
            self.output_folder = config.get('output_folder', 'chunks')
        else:
            # Default values if config file doesn't exist
            self.source_path = 'G-Indexation/Hermetic_Library'
            self.sentences_per_chunk = 5
            self.overlap_sentences = 2
            self.output_folder = 'chunks'
        
        # Create output directory if it doesn't exist
        Path(self.output_folder).mkdir(exist_ok=True)
        
        # Initialize chunk counter
        self.chunk_id = 1
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        # Clean the text first
        text = self.clean_text(text)
        
        # Use NLTK sentence tokenizer
        sentences = sent_tokenize(text)
        
        # Filter out empty sentences and clean each sentence
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def create_chunks_with_overlap(self, sentences: List[str], source_file: str) -> List[Dict[str, Any]]:
        """Create chunks from sentences with specified overlap."""
        chunks = []
        
        if len(sentences) <= self.sentences_per_chunk:
            # If text is shorter than chunk size, create single chunk
            chunk_text = ' '.join(sentences)
            chunks.append({
                'chunk_id': f"{source_file}_chunk_{self.chunk_id:06d}",
                'source_file': source_file,
                'start_sentence': 0,
                'end_sentence': len(sentences) - 1,
                'text': chunk_text
            })
            self.chunk_id += 1
            return chunks
        
        # Calculate step size (how many sentences to move forward)
        step_size = self.sentences_per_chunk - self.overlap_sentences
        
        # Create chunks with overlap
        for i in range(0, len(sentences), step_size):
            end_idx = min(i + self.sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[i:end_idx]
            
            # Skip if chunk is too small (except for the last chunk)
            if len(chunk_sentences) < self.sentences_per_chunk and i > 0:
                break
                
            chunk_text = ' '.join(chunk_sentences)
            
            chunks.append({
                'chunk_id': f"{source_file}_chunk_{self.chunk_id:06d}",
                'source_file': source_file,
                'start_sentence': i,
                'end_sentence': end_idx - 1,
                'text': chunk_text
            })
            self.chunk_id += 1
            
            # If we've reached the end, break
            if end_idx >= len(sentences):
                break
        
        return chunks
    
    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file and return its chunks."""
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into sentences
            sentences = self.split_into_sentences(content)
            
            if not sentences:
                print(f"Warning: No sentences found in {file_path.name}")
                return []
            
            # Create chunks
            chunks = self.create_chunks_with_overlap(sentences, file_path.name)
            
            print(f"Processed {file_path.name}: {len(sentences)} sentences -> {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            return []
    
    def save_chunks_to_single_file(self, all_chunks: List[Dict[str, Any]]) -> None:
        """Save all chunks to a single JSON file."""
        filename = "chunks.json"
        filepath = Path(self.output_folder) / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    
    def process_all_files(self) -> None:
        """Process all text files in the source directory."""
        source_dir = Path(self.source_path)
        
        if not source_dir.exists():
            print(f"Error: Source directory '{self.source_path}' does not exist.")
            return
        
        # Find all markdown files (.md only)
        text_files = list(source_dir.glob('*.md'))
        
        if not text_files:
            print(f"No markdown files found in {self.source_path}")
            return
        
        print(f"Found {len(text_files)} markdown files to process")
        print(f"Configuration: {self.sentences_per_chunk} sentences per chunk, {self.overlap_sentences} sentences overlap")
        print("-" * 50)
        
        all_chunks = []
        
        # Process each file
        for file_path in sorted(text_files):
            chunks = self.process_file(file_path)
            all_chunks.extend(chunks)
        
        # Save all chunks to a single file
        self.save_chunks_to_single_file(all_chunks)
        
        print("-" * 50)
        print(f"Processing complete! Created {len(all_chunks)} chunks in '{self.output_folder}/chunks.json'")

def main():
    """Main function to run the document chunker."""
    chunker = DocumentChunker()
    chunker.process_all_files()

if __name__ == "__main__":
    main() 