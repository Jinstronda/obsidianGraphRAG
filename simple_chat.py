#!/usr/bin/env python3
"""
Simple Obsidian Graph RAG Chat Interface

This is a simplified version for testing the chat functionality
with a subset of your notes.
"""

import os
import sys
import logging
from pathlib import Path
import networkx as nx
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simple document class
class SimpleDocument:
    def __init__(self, id, title, content, path):
        self.id = id
        self.title = title  
        self.content = content
        self.path = path
        self.wikilinks = set()

# Simple Graph RAG system
class SimpleGraphRAG:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.documents = {}
        self.embeddings = {}
        self.graph = nx.Graph()
        
    def add_sample_documents(self):
        """Add a few sample documents for testing."""
        
        # Sample documents (you can replace with real content)
        sample_docs = [
            {
                "id": "doc1",
                "title": "Artificial Intelligence",
                "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines. It includes machine learning, neural networks, and deep learning. AI has applications in various fields including natural language processing, computer vision, and robotics.",
                "path": "ai.md"
            },
            {
                "id": "doc2", 
                "title": "Machine Learning",
                "content": "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
                "path": "ml.md"
            },
            {
                "id": "doc3",
                "title": "Neural Networks", 
                "content": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that can learn complex patterns in data. Deep neural networks with many layers are used in deep learning applications.",
                "path": "nn.md"
            }
        ]
        
        for doc_data in sample_docs:
            doc = SimpleDocument(
                doc_data["id"],
                doc_data["title"],
                doc_data["content"],
                doc_data["path"]
            )
            self.documents[doc.id] = doc
            
        # Add simple graph connections
        self.graph.add_edge("doc1", "doc2")  # AI -> ML
        self.graph.add_edge("doc1", "doc3")  # AI -> NN
        self.graph.add_edge("doc2", "doc3")  # ML -> NN
        
    def generate_embeddings(self):
        """Generate embeddings for documents."""
        print("🔄 Generating embeddings...")
        
        for doc_id, doc in self.documents.items():
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=f"{doc.title}\n\n{doc.content}"
                )
                self.embeddings[doc_id] = np.array(response.data[0].embedding)
            except Exception as e:
                print(f"❌ Error generating embedding for {doc.title}: {e}")
                
        print(f"✅ Generated {len(self.embeddings)} embeddings")
        
    def search_documents(self, query, top_k=2):
        """Search for relevant documents."""
        if not self.embeddings:
            return list(self.documents.values())[:top_k]
            
        try:
            # Generate query embedding
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            query_embedding = np.array(response.data[0].embedding)
            
            # Calculate similarities
            similarities = []
            for doc_id, doc_embedding in self.embeddings.items():
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append((doc_id, similarity))
            
            # Sort and get top documents
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for doc_id, score in similarities[:top_k]:
                results.append(self.documents[doc_id])
                
            # Add connected documents via graph
            for doc_id, score in similarities[:1]:  # Take top result
                if doc_id in self.graph:
                    neighbors = list(self.graph.neighbors(doc_id))
                    for neighbor_id in neighbors[:1]:  # Add one connected doc
                        if neighbor_id in self.documents:
                            neighbor_doc = self.documents[neighbor_id]
                            if neighbor_doc not in results:
                                results.append(neighbor_doc)
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return list(self.documents.values())[:top_k]
    
    def answer_question(self, question):
        """Answer a question using retrieved documents."""
        print(f"🔍 Searching for: {question}")
        
        # Retrieve relevant documents
        relevant_docs = self.search_documents(question)
        
        # Format context
        context = ""
        for doc in relevant_docs:
            context += f"## {doc.title}\n\n{doc.content}\n\n"
            
        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an AI assistant that answers questions based on the provided context from a knowledge base. Be helpful and cite the relevant sources."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating answer: {e}"

def main():
    """Main chat interface."""
    
    print("🤖 Simple Obsidian Graph RAG Chat")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ API key required")
        return
    
    # Initialize system
    print("🔧 Initializing simple Graph RAG system...")
    rag = SimpleGraphRAG(api_key)
    rag.add_sample_documents()
    rag.generate_embeddings()
    
    print(f"✅ Loaded {len(rag.documents)} sample documents")
    print("📝 Sample topics: AI, Machine Learning, Neural Networks")
    print("\n" + "="*40)
    print("Ask me anything about these topics!")
    print("Type 'quit' to exit.")
    print("="*40 + "\n")
    
    # Chat loop
    while True:
        try:
            question = input("📝 You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
                
            if not question:
                continue
                
            print("\n🤔 Thinking...")
            answer = rag.answer_question(question)
            print(f"\n🤖 AI: {answer}\n")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Chat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 