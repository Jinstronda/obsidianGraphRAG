# GraphRAG Document Chunker

This project contains a Python script that chunks text documents by sentences for use in GraphRAG applications.

## Features

- **Sentence-based chunking**: Splits documents into chunks based on sentence boundaries using NLTK
- **Configurable chunk size**: Set the number of sentences per chunk
- **Overlap support**: Configure sentence overlap between chunks for better context preservation
- **Markdown files**: Processes `.md` files only
- **JSON output**: Each chunk is saved as a JSON file with metadata
- **Configuration file**: Uses `config.json` for easy parameter adjustment

## Setup

1. **Create virtual environment** (already done):
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install dependencies** (already done):
   ```bash
   pip install python-dotenv nltk
   ```

3. **Download NLTK data** (already done):
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

## Configuration

Edit the `config.json` file to customize the chunking parameters:

```json
{
  "source_path": "G-Indexation/Hermetic_Library",
  "sentences_per_chunk": 5,
  "overlap_sentences": 2,
  "output_folder": "chunks"
}
```

### Parameters

- **source_path**: Directory containing the text files to process
- **sentences_per_chunk**: Number of sentences in each chunk
- **overlap_sentences**: Number of sentences that overlap between consecutive chunks
- **output_folder**: Directory where chunk JSON files will be saved

## Usage

Run the chunking script:

```bash
python G-Indexation/chunker.py
```

## Output

The script creates a single JSON file containing all chunks. The output file is located at `G-Indexation/Graph_fragments/chunks.json` and contains an array of chunk objects, each with the following structure:

```json
[
  {
    "chunk_id": "emerald_tablet.md_chunk_000001",
    "source_file": "emerald_tablet.md",
    "start_sentence": 0,
    "end_sentence": 4,
    "text": "The actual content of the chunk..."
  },
  {
    "chunk_id": "emerald_tablet.md_chunk_000002",
    "source_file": "emerald_tablet.md",
    "start_sentence": 3,
    "end_sentence": 7,
    "text": "The actual content of the next chunk..."
  }
]
```

### Output Fields

- **chunk_id**: Globally unique identifier combining source file and chunk number (e.g., "emerald_tablet.md_chunk_000001")
- **source_file**: Name of the original document file
- **start_sentence**: Index of the first sentence in the chunk (0-based)
- **end_sentence**: Index of the last sentence in the chunk (0-based)
- **text**: The actual text content of the chunk

## Example

For the current configuration:
- **Source**: `G-Indexation/Hermetic_Library` (22 files)
- **Chunk size**: 5 sentences per chunk
- **Overlap**: 2 sentences between chunks
- **Output**: 2,764 chunks created in `G-Indexation/Graph_fragments/chunks.json`

## File Structure

```
GraphRAG_reincarnated/
├── G-Indexation/
│   ├── chunker.py              # Main chunking script
│   ├── Hermetic_Library/       # Source text files
│   └── Graph_fragments/        # Output directory
│       └── chunks.json         # Single JSON file with all chunks
├── config.json                # Configuration file
├── venv/                      # Virtual environment
└── README.md                 # This file
```

## Customization

To process different files or change chunking parameters:

1. Update the `config.json` file with your desired settings
2. Place your text files in the specified source directory
3. Run the script again

The script will automatically create the output directory if it doesn't exist and process all `.md` files in the source directory. 