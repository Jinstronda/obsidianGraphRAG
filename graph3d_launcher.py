#!/usr/bin/env python3
"""
3D Graph Launcher for Obsidian Graph RAG
========================================

Unified launcher for the Obsidian Graph RAG system with 3D visualization
capabilities using Plotly instead of Three.js for better reliability.

This launcher provides multiple interface options:
1. 3D Graph Visualizer (Plotly-based)
2. AI Librarian Chat
3. Combined interface

The Plotly-based approach eliminates browser compatibility issues and
provides better performance and user experience.

Author: Assistant
License: MIT
"""

import asyncio
import sys
import os
import logging
import webbrowser
import tempfile
from pathlib import Path
from typing import Optional

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import our modules
from graphrag import ObsidianGraphRAG, GraphRAGConfig
from plotly_graph_visualizer import PlotlyGraphVisualizer, create_obsidian_graph_visualization


class Graph3DLauncher:
    """
    Main launcher class for the 3D Graph RAG system.
    
    This class orchestrates the initialization and launching of different
    interface options for the Obsidian Graph RAG system.
    """
    
    def __init__(self):
        """Initialize the launcher."""
        self.logger = logging.getLogger(__name__)
        self.graph_rag = None
        self.visualizer = None
        
        # Configuration
        self.config = GraphRAGConfig()
        # Use environment variable or prompt user for vault path
        # self.config.vault_path will be set from environment or user input
        
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.
        
        Returns:
            True if all dependencies are available
        """
        try:
            print("🔍 Checking dependencies...")
            
            # Check core dependencies
            import networkx
            import plotly
            import numpy
            import pandas
            
            # Check optional dependencies
            try:
                import openai
                openai_available = True
            except ImportError:
                openai_available = False
            
            print("✓ All core dependencies available")
            if not openai_available:
                print("⚠️  OpenAI not available - AI features will be limited")
            
            return True
            
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("\nPlease install required packages:")
            print("pip install plotly networkx numpy pandas community")
            print("pip install openai  # Optional, for AI features")
            return False
    
    def setup_api_key(self) -> None:
        """Setup OpenAI API key if not already configured."""
        if not self.config.openai_api_key:
            print("⚠️  OpenAI API key not found in environment.")
            api_key = input("Please enter your OpenAI API key (or press Enter to skip AI features): ").strip()
            if api_key:
                self.config.openai_api_key = api_key
            else:
                print("⚠️  Continuing without OpenAI API key. AI features will be limited.")
    
    def initialize_system(self) -> bool:
        """
        Initialize the Graph RAG system.
        
        Returns:
            True if initialization successful
        """
        try:
            print(f"\n🔧 Initializing system for vault: {self.config.vault_path}")
            self.graph_rag = ObsidianGraphRAG(self.config)
            print("✓ System initialized successfully")
            
            print("\n📚 Initializing Data (with caching support)")
            print("-" * 40)
            print("💡 Note: If this is your first run with incremental processing,")
            print("   a one-time cache upgrade will be performed.")
            
            # Initialize system with persistence support
            self.graph_rag.initialize_system()
            
            # Display results
            print(f"\n📊 Processing Results:")
            print(f"   Documents processed: {len(self.graph_rag.documents)}")
            print(f"   Graph nodes: {self.graph_rag.knowledge_graph.number_of_nodes()}")
            print(f"   Graph edges: {self.graph_rag.knowledge_graph.number_of_edges()}")
            
            # Show sample of processed documents
            if self.graph_rag.documents:
                print(f"\n📝 Sample processed documents:")
                for i, (doc_id, doc) in enumerate(list(self.graph_rag.documents.items())[:5]):
                    print(f"   {i+1}. {doc.title} ({doc.word_count} words, {len(doc.wikilinks)} links)")
            
            print("\n✓ System initialization completed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            print(f"❌ Initialization failed: {e}")
            return False
    
    def launch_3d_visualizer(self, layout: str = 'spring_3d') -> bool:
        """
        Launch the 3D graph visualizer using Plotly.
        
        Args:
            layout: Layout algorithm to use
            
        Returns:
            True if successful
        """
        try:
            print("\n🌐 Starting 3D Graph Visualizer...")
            print("🚀 Launching 3D Graph Visualizer...")
            print("=" * 50)
            
            # Create visualizer
            self.visualizer = PlotlyGraphVisualizer(self.config)
            
            # Create visualization
            fig = self.visualizer.visualize_graph(
                knowledge_graph=self.graph_rag.knowledge_graph,
                documents=self.graph_rag.documents,
                layout_algorithm=layout,
                title=f"Obsidian Knowledge Graph - {len(self.graph_rag.documents)} Documents",
                auto_open=True
            )
            
            print("✓ 3D Visualizer started successfully!")
            print(f"📊 Visualization Features:")
            print("  • Interactive 3D graph navigation")
            print("  • Multiple layout algorithms")
            print("  • Community detection and coloring")
            print("  • Node search and filtering")
            print("  • Export to HTML")
            print("  • Performance optimization for large graphs")
            
            print(f"\n🎮 Controls:")
            print("  • Mouse: Rotate view")
            print("  • Scroll: Zoom in/out") 
            print("  • Click nodes: View details")
            print("  • Hover: See node information")
            print("  • Layout dropdown: Change visualization style")
            
            print(f"\n⚡ Available Layout Algorithms:")
            algorithms = {
                'spring_3d': 'Physics-based layout with attractive/repulsive forces',
                'circular_3d': 'Multi-layer circular arrangement',
                'spherical_3d': 'Nodes arranged on sphere surface',
                'hierarchical_3d': 'Vertical layers based on node importance',
                'community_layers_3d': 'Separate Z-layers for communities'
            }
            
            for alg, desc in algorithms.items():
                marker = "→" if alg == layout else " "
                print(f"  {marker} {alg.replace('_', ' ').title()}: {desc}")
            
            print(f"\n🌍 Visualization opened in browser automatically")
            print("💡 The visualization is a standalone HTML file that works offline")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error launching 3D visualizer: {e}")
            print(f"❌ 3D Visualizer failed: {e}")
            return False
    
    def launch_ai_chat(self) -> bool:
        """
        Launch the AI chat interface.
        
        Returns:
            True if successful
        """
        try:
            if not self.config.openai_api_key:
                print("❌ AI Chat requires OpenAI API key")
                return False
            
            print("\n🤖 Starting AI Librarian Chat Interface...")
            print("This may take a moment to generate embeddings for your notes...")
            
            self.graph_rag.start_chat_session()
            return True
            
        except Exception as e:
            self.logger.error(f"Error launching AI chat: {e}")
            print(f"❌ AI Chat failed: {e}")
            return False
    
    def show_interface_menu(self) -> int:
        """
        Show interface selection menu.
        
        Returns:
            Selected option number
        """
        print("\n🎛️ Choose Your Interface:")
        print("1. 🌐 3D Graph Visualizer (Interactive 3D exploration)")
        print("2. 🤖 AI Librarian Chat (Text-based Q&A)")
        print("3. 🔧 Both (Start with 3D, then chat)")
        print("4. 🎨 3D Visualizer (Custom settings)")
        
        while True:
            try:
                choice = input("\nSelect option (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return int(choice)
                else:
                    print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                sys.exit(0)
    
    def get_custom_layout(self) -> str:
        """
        Get custom layout selection from user.
        
        Returns:
            Selected layout algorithm
        """
        layouts = {
            '1': 'spring_3d',
            '2': 'circular_3d', 
            '3': 'spherical_3d',
            '4': 'hierarchical_3d',
            '5': 'community_layers_3d'
        }
        
        print("\n🎨 Select Layout Algorithm:")
        print("1. Spring 3D (Physics-based, recommended)")
        print("2. Circular 3D (Multi-layer circles)")
        print("3. Spherical 3D (Nodes on sphere surface)")
        print("4. Hierarchical 3D (Vertical importance layers)")
        print("5. Community Layers 3D (Communities on separate layers)")
        
        while True:
            try:
                choice = input("\nSelect layout (1-5): ").strip()
                if choice in layouts:
                    return layouts[choice]
                else:
                    print("Invalid choice. Please enter 1-5.")
            except KeyboardInterrupt:
                return 'spring_3d'  # Default fallback


async def main():
    """Main async entry point."""
    
    print("🌐 Obsidian Graph RAG 3D Visualizer")
    print("=" * 60)
    
    try:
        # Initialize launcher
        launcher = Graph3DLauncher()
        
        # Check dependencies
        if not launcher.check_dependencies():
            sys.exit(1)
        
        # Setup API key
        launcher.setup_api_key()
        
        # Initialize system
        if not launcher.initialize_system():
            sys.exit(1)
        
        # Show interface menu
        choice = launcher.show_interface_menu()
        
        if choice == 1:
            # 3D Visualizer only
            launcher.launch_3d_visualizer()
            
        elif choice == 2:
            # AI Chat only
            if launcher.config.openai_api_key:
                launcher.launch_ai_chat()
            else:
                print("❌ AI Chat requires OpenAI API key")
                
        elif choice == 3:
            # Both interfaces
            launcher.launch_3d_visualizer()
            
            if launcher.config.openai_api_key:
                input("\n📝 Press Enter to continue to AI Chat...")
                launcher.launch_ai_chat()
            else:
                print("\n⚠️ Skipping AI Chat (no API key provided)")
                
        elif choice == 4:
            # Custom 3D settings
            layout = launcher.get_custom_layout()
            launcher.launch_3d_visualizer(layout)
        
        print("\n✓ Session completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    """Entry point for the launcher."""
    asyncio.run(main()) 