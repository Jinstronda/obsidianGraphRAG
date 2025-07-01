import os
import asyncio
import pandas as pd
from graph_generation.index.text_splitting.text_splitting import TokenTextSplitter
from graph_generation.index.operations.extract_graph.extract_graph import extract_graph
from graph_generation.index.operations.build_noun_graph.build_noun_graph import build_noun_graph
from graph_generation.index.operations.cluster_graph import cluster_graph
from graph_generation.index.operations.summarize_communities.summarize_communities import summarize_communities
from graph_generation.config.enums import AsyncType
import networkx as nx
from graph_generation.index.operations.build_noun_graph.np_extractors.regex_extractor import RegexENNounPhraseExtractor
# TODO: Import or define PipelineCache, WorkflowCallbacks, and any config needed

# Set OpenAI API key as environment variable for fnllm
os.environ["OPENAI_API_KEY"] = "put-your-OpenAI_API-key-here"

# Placeholder cache and callbacks (replace with real implementations as needed)
class DummyCache:
    async def get(self, key): return None
    async def set(self, key, value, metadata=None): pass
    def child(self, name): return self

class DummyCallbacks:
    def progress(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): print("Error:", args, kwargs)

async def process_markdown_files(md_folder):
    for filename in os.listdir(md_folder):
        if filename.endswith('.md'):
            file_path = os.path.join(md_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            # Step 1: Split into chunks
            splitter = TokenTextSplitter()
            chunks = splitter.split_text(raw_text)
            print(f"Processed {filename}: {len(chunks)} chunks")
            # Save chunks
            pd.DataFrame({'chunk': chunks}).to_csv(f"{file_path}_chunks.csv", index=False)

            # Step 2: Entity & Relationship Extraction
            # Convert chunks to DataFrame for extraction
            text_units = pd.DataFrame({'id': range(len(chunks)), 'text': chunks})
            strategy = {
                "llm": {
                    "provider": "openai",
                    "type": "openai_chat",  # Required by LanguageModelConfig
                    "model": "gpt-3.5-turbo",
                    "api_key": os.environ["OPENAI_API_KEY"],
                }
            }
            entities, relationships = await extract_graph(
                text_units=text_units,
                callbacks=DummyCallbacks(),
                cache=DummyCache(),
                text_column='text',
                id_column='id',
                strategy=strategy,
                async_mode=AsyncType.AsyncIO,  # Use valid async mode
            )
            print(f"Extracted {len(entities)} entities and {len(relationships)} relationships from {filename}")
            print(f"Entity DataFrame columns: {entities.columns.tolist()}")
            if 'title' not in entities.columns:
                print("First few rows of entities DataFrame:")
                print(entities.head())
            entities.to_csv(f"{file_path}_entities.csv", index=False)
            relationships.to_csv(f"{file_path}_relationships.csv", index=False)

            # Step 3: Build Knowledge Graph (noun graph)
            # Use a real noun phrase extractor
            text_analyzer = RegexENNounPhraseExtractor(
                exclude_nouns=[],
                max_word_length=15,
                word_delimiter=" "
            )
            # Print noun phrases from the first chunk for demonstration
            print("Noun phrases from first chunk:", text_analyzer.extract(chunks[0]))
            nodes, edges = await build_noun_graph(
                text_unit_df=text_units,
                text_analyzer=text_analyzer,
                normalize_edge_weights=True,
                num_threads=4,
                cache=DummyCache(),
            )
            print(f"Built graph with {len(nodes)} nodes and {len(edges)} edges for {filename}")
            nodes.to_csv(f"{file_path}_nodes.csv", index=False)
            edges.to_csv(f"{file_path}_edges.csv", index=False)

            # Step 4: Community Detection
            # For demonstration, use networkx to build a graph from edges
            G = nx.Graph()
            for _, row in edges.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])
            communities = cluster_graph(G, max_cluster_size=10, use_lcc=True)
            print(f"Detected {len(communities)} communities in {filename}")
            pd.DataFrame(communities, columns=['level', 'community', 'parent', 'nodes']).to_csv(f"{file_path}_communities.csv", index=False)

            # Step 5: Community Summarization (placeholder, as this is complex)
            # TODO: Integrate real summarization logic and configs
            # summaries = await summarize_communities(...)
            print(f"[TODO] Summarization for {filename} not yet implemented.")
            # Step 6: (Optional) Query Answering
            print(f"[TODO] Query answering for {filename} not yet implemented.")

if __name__ == "__main__":
    asyncio.run(process_markdown_files("graph_generation/Library")) 