#!/usr/bin/env python3
"""
Obsidian Graph RAG Web Chat Interface
=====================================

A beautiful, modern web interface for chatting with your Obsidian knowledge base.
Features a clean, responsive design similar to modern AI assistants.

Features:
- Modern, aesthetic design with dark/light themes
- Real-time chat interface
- Source citations and references
- Responsive layout
- Self-contained (no external dependencies)
- Auto-launches in browser

Usage:
    python web_chat.py
    python web_chat.py --port 5000

Author: Assistant
"""

import os
import sys
import json
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
import asyncio

# Flask imports
try:
    from flask import Flask, render_template_string, request, jsonify, send_from_directory
except ImportError:
    print("❌ Flask not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    from flask import Flask, render_template_string, request, jsonify, send_from_directory

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from graphrag import GraphRAGConfig, ObsidianGraphRAG, ObsidianChatBot


class WebChatServer:
    """
    Web-based chat server for Obsidian Graph RAG system.
    
    Provides a modern, aesthetic web interface for interacting with
    the knowledge base through a browser.
    """
    
    def __init__(self, config: GraphRAGConfig, graph_rag_system: ObsidianGraphRAG):
        """
        Initialize the web chat server.
        
        Args:
            config: Graph RAG configuration
            graph_rag_system: Initialized Graph RAG system
        """
        self.config = config
        self.graph_rag = graph_rag_system
        self.chat_history = []
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.secret_key = 'obsidian-graph-rag-secret-key'
        
        # Initialize chat bot
        self.chat_bot = ObsidianChatBot(graph_rag_system)
        self.chat_bot.initialize()
        
        # Chat settings
        self.current_model = self.config.llm_model
        self.custom_prompt = self.config.custom_system_prompt if hasattr(self.config, 'custom_system_prompt') else None
        
        # Setup routes
        self._setup_routes()
        
        print("🌐 Web chat server initialized successfully")
        print(f"🤖 AI Model: {self.current_model}")
        if self.custom_prompt:
            print(f"📝 Custom Prompt: {self.custom_prompt[:100]}..." if len(self.custom_prompt) > 100 else f"📝 Custom Prompt: {self.custom_prompt}")
    
    def _setup_routes(self):
        """Setup Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            """Main chat interface."""
            return render_template_string(HTML_TEMPLATE, 
                                        vault_name=Path(self.config.vault_path).name,
                                        document_count=len(self.graph_rag.documents),
                                        graph_edges=self.graph_rag.knowledge_graph.number_of_edges())
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """Handle chat messages."""
            try:
                data = request.json
                message = data.get('message', '').strip()
                
                if not message:
                    return jsonify({'error': 'Empty message'}), 400
                
                # Add user message to history
                user_msg = {
                    'type': 'user',
                    'content': message,
                    'timestamp': datetime.now().isoformat()
                }
                self.chat_history.append(user_msg)
                
                # Get AI response
                if self.custom_prompt:
                    # Temporarily set custom prompt for this request
                    original_prompt = getattr(self.chat_bot.retriever.config, 'custom_system_prompt', None)
                    self.chat_bot.retriever.config.custom_system_prompt = self.custom_prompt
                
                # Set current model
                original_model = self.chat_bot.retriever.config.llm_model
                self.chat_bot.retriever.config.llm_model = self.current_model
                    
                response = asyncio.run(self.chat_bot.ask_question(message))
                
                # Restore original settings
                self.chat_bot.retriever.config.llm_model = original_model
                
                if self.custom_prompt:
                    # Restore original prompt
                    if original_prompt:
                        self.chat_bot.retriever.config.custom_system_prompt = original_prompt
                    else:
                        delattr(self.chat_bot.retriever.config, 'custom_system_prompt')
                
                # Add AI response to history with restored source extraction
                sources = asyncio.run(self._extract_sources(message))
                ai_msg = {
                    'type': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat(),
                    'sources': sources
                }
                self.chat_history.append(ai_msg)
                
                return jsonify({
                    'response': response,
                    'sources': sources,
                    'timestamp': ai_msg['timestamp']
                })
                
            except Exception as e:
                print(f"❌ Chat error: {e}")
                return jsonify({'error': f'Chat error: {str(e)}'}), 500
        
        @self.app.route('/api/history')
        def get_history():
            """Get chat history."""
            return jsonify({'history': self.chat_history})
        
        @self.app.route('/api/clear', methods=['POST'])
        def clear_history():
            """Clear chat history."""
            self.chat_history.clear()
            return jsonify({'status': 'cleared'})
        
        @self.app.route('/api/status')
        def get_status():
            """Get system status."""
            return jsonify({
                'status': 'ready',
                'vault_path': self.config.vault_path,
                'document_count': len(self.graph_rag.documents),
                'graph_edges': self.graph_rag.knowledge_graph.number_of_edges(),
                'has_openai': bool(self.config.openai_api_key),
                'model': self.current_model,
                'custom_prompt': self.custom_prompt
            })
        
        @self.app.route('/api/model', methods=['GET', 'POST'])
        def model_settings():
            """Get or update model settings."""
            if request.method == 'GET':
                return jsonify({
                    'model': self.current_model,
                    'custom_prompt': self.custom_prompt,
                    'available_models': [
                        'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-4-turbo',
                        'gpt-3.5-turbo', 'gpt-3.5-turbo-16k'
                    ]
                })
            
            # POST - Update settings
            try:
                data = request.json
                if 'model' in data:
                    self.current_model = data['model']
                    self.config.llm_model = data['model']
                
                if 'custom_prompt' in data:
                    self.custom_prompt = data['custom_prompt']
                    self.config.custom_system_prompt = data['custom_prompt']
                
                return jsonify({
                    'status': 'updated',
                    'model': self.current_model,
                    'custom_prompt': self.custom_prompt
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    async def _extract_sources(self, query: str) -> List[Dict[str, str]]:
        """Extract source documents for a query."""
        try:
            # Use the retriever to get context documents
            from graphrag import GraphRAGRetriever, EmbeddingManager
            
            retriever = GraphRAGRetriever(self.config)
            embedding_manager = EmbeddingManager(self.config, self.graph_rag.client, self.graph_rag)
            
            context_docs = await retriever.retrieve_context(
                query,
                self.graph_rag.documents,
                self.graph_rag.knowledge_graph,
                embedding_manager
            )
            
            sources = []
            for doc in context_docs[:5]:  # Limit to top 5 sources
                sources.append({
                    'title': doc.title,
                    'path': doc.path,
                    'word_count': doc.word_count,
                    'excerpt': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                })
            
            return sources
            
        except Exception as e:
            print(f"⚠️  Could not extract sources: {e}")
            return []
    
    def run(self, host='127.0.0.1', port=5000, debug=False, auto_open=True):
        """
        Run the web server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
            auto_open: Automatically open browser
        """
        def open_browser_robust():
            """Robust browser opening with multiple strategies."""
            import time
            import subprocess
            import platform
            
            url = f'http://{host}:{port}'
            time.sleep(2)  # Give server time to start
            
            print(f"🌐 Opening browser to: {url}")
            
            try:
                # Strategy 1: Python webbrowser module
                webbrowser.open(url)
                print("✅ Browser opened successfully")
                return
            except Exception as e:
                print(f"⚠️  Browser open method 1 failed: {e}")
            
            try:
                # Strategy 2: OS-specific commands
                system = platform.system()
                if system == "Windows":
                    subprocess.run(['start', url], shell=True, check=True)
                elif system == "Darwin":  # macOS
                    subprocess.run(['open', url], check=True)
                elif system == "Linux":
                    subprocess.run(['xdg-open', url], check=True)
                print("✅ Browser opened via OS command")
                return
            except Exception as e:
                print(f"⚠️  Browser open method 2 failed: {e}")
            
            # Strategy 3: Direct browser execution
            try:
                browsers = [
                    'chrome', 'firefox', 'msedge', 'safari',
                    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                    'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
                    'C:\\Program Files\\Mozilla Firefox\\firefox.exe',
                    'C:\\Program Files (x86)\\Mozilla Firefox\\firefox.exe'
                ]
                
                for browser in browsers:
                    try:
                        subprocess.run([browser, url], check=True)
                        print(f"✅ Browser opened: {browser}")
                        return
                    except:
                        continue
                        
            except Exception as e:
                print(f"⚠️  All browser opening methods failed: {e}")
            
            print(f"🔗 Please manually open: {url}")
        
        if auto_open:
            # Start browser opening in background thread
            threading.Thread(target=open_browser_robust, daemon=True).start()
        
        print(f"\n🚀 Starting Obsidian AI Librarian Web Interface...")
        print(f"📍 URL: http://{host}:{port}")
        print(f"📚 Vault: {Path(self.config.vault_path).name}")
        print(f"📊 {len(self.graph_rag.documents):,} documents loaded")
        print(f"🤖 AI Model: {self.current_model}")
        if auto_open:
            print(f"🌐 Opening in your browser...")
        print(f"\n💡 Manual access: http://{host}:{port}")
        print("⚙️  Press Ctrl+C to stop the server")
        print("🎛️  Access settings in the web interface to customize prompts")
        
        try:
            # Force immediate server start
            self.app.run(host=host, port=port, debug=debug, use_reloader=False, threaded=True)
        except KeyboardInterrupt:
            print("\n👋 Web server stopped")
        except Exception as e:
            print(f"❌ Server error: {e}")
            print(f"🔗 Try manually opening: http://{host}:{port}")


# HTML Template with inline CSS and JavaScript for self-contained deployment
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Obsidian AI Librarian</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-color: #2563eb;
            --primary-dark: #1d4ed8;
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
            --success-color: #059669;
            --error-color: #dc2626;
            --user-bg: #2563eb;
            --assistant-bg: #374151;
            --input-bg: #1e293b;
        }

        [data-theme="light"] {
            --bg-color: #ffffff;
            --surface-color: #f8fafc;
            --text-color: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --user-bg: #2563eb;
            --assistant-bg: #f1f5f9;
            --input-bg: #ffffff;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
            height: 100vh;
            overflow: hidden;
            transition: all 0.3s ease;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }

        .header {
            background: var(--surface-color);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, var(--primary-color), #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .vault-info {
            color: var(--text-muted);
            font-size: 0.875rem;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .theme-toggle {
            background: none;
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.5rem;
            border-radius: 0.5rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .theme-toggle:hover {
            background: var(--border-color);
        }

        .settings-button {
            background: none;
            border: none;
            color: var(--text-color);
            padding: 0.5rem;
            border-radius: 0.5rem;
            font-size: 1.2rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .settings-button:hover {
            background: var(--border-color);
        }

        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(5px);
        }

        .modal-content {
            background-color: var(--surface-color);
            margin: 5% auto;
            padding: 2rem;
            border: 1px solid var(--border-color);
            border-radius: 1rem;
            width: 90%;
            max-width: 600px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .modal-title {
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--text-color);
        }

        .close {
            color: var(--text-muted);
            font-size: 2rem;
            font-weight: bold;
            cursor: pointer;
            border: none;
            background: none;
            padding: 0;
            line-height: 1;
        }

        .close:hover {
            color: var(--text-color);
        }

        .form-group {
            margin-bottom: 1.5rem;
        }

        .form-label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: var(--text-color);
        }

        .form-input, .form-select, .form-textarea {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            background: var(--input-bg);
            color: var(--text-color);
            font-size: 1rem;
            transition: border-color 0.2s ease;
        }

        .form-input:focus, .form-select:focus, .form-textarea:focus {
            outline: none;
            border-color: var(--primary-color);
        }

        .form-textarea {
            min-height: 120px;
            resize: vertical;
            font-family: inherit;
        }

        .form-buttons {
            display: flex;
            gap: 1rem;
            justify-content: flex-end;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 0.5rem;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-dark);
        }

        .btn-secondary {
            background: var(--border-color);
            color: var(--text-color);
        }

        .btn-secondary:hover {
            background: var(--text-muted);
        }

        .current-model {
            display: inline-block;
            background: var(--primary-color);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 1rem;
            font-size: 0.875rem;
            font-weight: 600;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--success-color);
            color: white;
            border-radius: 1rem;
            font-size: 0.875rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #ffffff;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .message {
            display: flex;
            gap: 1rem;
            animation: slideIn 0.3s ease;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            justify-content: flex-end;
        }

        .message-content {
            max-width: 70%;
            padding: 1rem 1.5rem;
            border-radius: 1.5rem;
            position: relative;
        }

        .message.user .message-content {
            background: var(--user-bg);
            color: white;
            border-bottom-right-radius: 0.5rem;
        }

        .message.assistant .message-content {
            background: var(--assistant-bg);
            color: var(--text-color);
            border-bottom-left-radius: 0.5rem;
            border: 1px solid var(--border-color);
        }

        .message-avatar {
            width: 2.5rem;
            height: 2.5rem;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: var(--user-bg);
            color: white;
            order: 2;
        }

        .message.assistant .message-avatar {
            background: var(--primary-color);
            color: white;
        }

        .message-text {
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .message-time {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        .sources {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .sources-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .source-item {
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            font-size: 0.875rem;
        }

        .source-title {
            font-weight: 600;
            color: var(--primary-color);
        }

        .source-excerpt {
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        .input-area {
            background: var(--surface-color);
            border-top: 1px solid var(--border-color);
            padding: 1.5rem 2rem;
        }

        .input-container {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
            max-width: 1000px;
            margin: 0 auto;
        }

        .input-field {
            flex: 1;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 1.5rem;
            padding: 1rem 1.5rem;
            color: var(--text-color);
            font-size: 1rem;
            resize: none;
            max-height: 120px;
            min-height: 50px;
            transition: all 0.2s ease;
        }

        .input-field:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .send-button {
            background: var(--primary-color);
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.2s ease;
            flex-shrink: 0;
        }

        .send-button:hover:not(:disabled) {
            background: var(--primary-dark);
            transform: scale(1.05);
        }

        .send-button:disabled {
            background: var(--text-muted);
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-muted);
            font-style: italic;
        }

        .loading-dots {
            display: flex;
            gap: 2px;
        }

        .loading-dot {
            width: 4px;
            height: 4px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: loadingPulse 1.4s infinite;
        }

        .loading-dot:nth-child(2) { animation-delay: 0.2s; }
        .loading-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes loadingPulse {
            0%, 80%, 100% { opacity: 0.3; }
            40% { opacity: 1; }
        }

        .empty-state {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem;
        }

        .empty-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        .empty-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .empty-subtitle {
            color: var(--text-muted);
            margin-bottom: 2rem;
        }

        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: center;
        }

        .suggestion {
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            padding: 0.75rem 1.5rem;
            border-radius: 1rem;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.875rem;
        }

        .suggestion:hover {
            background: var(--primary-color);
            color: white;
            transform: translateY(-2px);
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .header {
                padding: 1rem;
            }

            .messages {
                padding: 1rem;
            }

            .message-content {
                max-width: 85%;
            }

            .input-area {
                padding: 1rem;
            }

            .header-right {
                gap: 0.5rem;
            }

            .vault-info {
                display: none;
            }
        }
    </style>
</head>
<body data-theme="dark">
    <div class="container">
        <header class="header">
            <div class="header-left">
                <div class="logo">🧠 Obsidian AI</div>
                <div class="vault-info">
                    {{ vault_name }} • {{ "{:,}".format(document_count) }} documents • {{ "{:,}".format(graph_edges) }} connections
                </div>
            </div>
            <div class="header-right">
                <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
                    <span id="theme-icon">🌙</span>
                </button>
                <button class="settings-button" onclick="openSettings()" title="AI Settings">
                    ⚙️
                </button>
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span>Ready</span>
                </div>
            </div>
        </header>

        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="empty-state">
                    <div class="empty-icon">💬</div>
                    <div class="empty-title">Welcome to your AI Librarian!</div>
                    <div class="empty-subtitle">Ask me anything about your Obsidian vault. I can help you find connections, summarize topics, and explore your knowledge.</div>
                    <div class="suggestions">
                        <div class="suggestion" onclick="sendSuggestion('What are the main themes in my notes?')">Main themes</div>
                        <div class="suggestion" onclick="sendSuggestion('Show me recent notes about productivity')">Recent notes</div>
                        <div class="suggestion" onclick="sendSuggestion('Find connections between my ideas')">Find connections</div>
                        <div class="suggestion" onclick="sendSuggestion('Summarize my thoughts on leadership')">Summarize topics</div>
                    </div>
                </div>
            </div>

            <div class="input-area">
                <div class="input-container">
                    <textarea 
                        id="messageInput" 
                        class="input-field" 
                        placeholder="Ask me about your notes..."
                        rows="1"></textarea>
                    <button id="sendButton" class="send-button" onclick="sendMessage()">
                        <span id="send-icon">➤</span>
                    </button>
                </div>
            </div>
        </div>
    </div>

    <!-- Settings Modal -->
    <div id="settingsModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">🤖 AI Settings</h2>
                <button class="close" onclick="closeSettings()">&times;</button>
            </div>
            
            <form id="settingsForm">
                <div class="form-group">
                    <label class="form-label">AI Model <span id="currentModel" class="current-model">Loading...</span></label>
                    <select id="modelSelect" class="form-select">
                        <option value="gpt-4o">GPT-4o (Recommended)</option>
                        <option value="gpt-4o-mini">GPT-4o Mini (Fast & Economical)</option>
                        <option value="gpt-4">GPT-4 (Most Capable)</option>
                        <option value="gpt-4-turbo">GPT-4 Turbo (Balanced)</option>
                        <option value="gpt-3.5-turbo">GPT-3.5 Turbo (Economic)</option>
                        <option value="gpt-3.5-turbo-16k">GPT-3.5 Turbo 16K (Long Context)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label class="form-label">Custom System Prompt (Optional)</label>
                    <textarea id="customPrompt" class="form-textarea" 
                              placeholder="Enter a custom system prompt to modify the AI's behavior. Leave empty for default behavior.
                              
Example: 'You are a helpful research assistant specializing in academic papers. Always provide detailed citations and focus on evidence-based responses.'"></textarea>
                    <small style="color: var(--text-muted); font-size: 0.875rem; margin-top: 0.5rem; display: block;">
                        💡 The system prompt defines how the AI behaves. Use this to make it more specialized for your use case.
                    </small>
                </div>
                
                <div class="form-buttons">
                    <button type="button" class="btn btn-secondary" onclick="resetSettings()">Reset to Default</button>
                    <button type="button" class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        let isLoading = false;
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });

        // Send message on Enter (but not Shift+Enter)
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        function toggleTheme() {
            const body = document.body;
            const themeIcon = document.getElementById('theme-icon');
            
            if (body.getAttribute('data-theme') === 'dark') {
                body.setAttribute('data-theme', 'light');
                themeIcon.textContent = '☀️';
            } else {
                body.setAttribute('data-theme', 'dark');
                themeIcon.textContent = '🌙';
            }
        }

        function sendSuggestion(text) {
            messageInput.value = text;
            sendMessage();
        }

        function addMessage(type, content, timestamp, sources = null) {
            const emptyState = messagesContainer.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = type === 'user' ? 'U' : '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            const messageText = document.createElement('div');
            messageText.className = 'message-text';
            messageText.textContent = content;
            
            const messageTime = document.createElement('div');
            messageTime.className = 'message-time';
            messageTime.textContent = new Date(timestamp).toLocaleTimeString();
            
            messageContent.appendChild(messageText);
            messageContent.appendChild(messageTime);
            
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                
                const sourcesTitle = document.createElement('div');
                sourcesTitle.className = 'sources-title';
                sourcesTitle.textContent = `Sources (${sources.length})`;
                sourcesDiv.appendChild(sourcesTitle);
                
                sources.forEach(source => {
                    const sourceItem = document.createElement('div');
                    sourceItem.className = 'source-item';
                    
                    const sourceTitle = document.createElement('div');
                    sourceTitle.className = 'source-title';
                    sourceTitle.textContent = source.title;
                    
                    const sourceExcerpt = document.createElement('div');
                    sourceExcerpt.className = 'source-excerpt';
                    sourceExcerpt.textContent = source.excerpt;
                    
                    sourceItem.appendChild(sourceTitle);
                    sourceItem.appendChild(sourceExcerpt);
                    sourcesDiv.appendChild(sourceItem);
                });
                
                messageContent.appendChild(sourcesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function addLoadingMessage() {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.id = 'loading-message';
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = '🤖';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            const loading = document.createElement('div');
            loading.className = 'loading';
            loading.innerHTML = `
                <span>Thinking</span>
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            `;
            
            messageContent.appendChild(loading);
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            messagesContainer.appendChild(messageDiv);
            
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }

        function removeLoadingMessage() {
            const loadingMessage = document.getElementById('loading-message');
            if (loadingMessage) {
                loadingMessage.remove();
            }
        }

        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isLoading) return;

            isLoading = true;
            sendButton.disabled = true;
            document.getElementById('send-icon').textContent = '⏳';
            
            // Add user message
            addMessage('user', message, new Date().toISOString());
            
            // Clear input
            messageInput.value = '';
            messageInput.style.height = 'auto';
            
            // Add loading message
            addLoadingMessage();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                removeLoadingMessage();
                
                if (response.ok) {
                    addMessage('assistant', data.response, data.timestamp, data.sources);
                } else {
                    addMessage('assistant', `Error: ${data.error}`, new Date().toISOString());
                }
                
            } catch (error) {
                removeLoadingMessage();
                addMessage('assistant', `Network error: ${error.message}`, new Date().toISOString());
            }
            
            isLoading = false;
            sendButton.disabled = false;
            document.getElementById('send-icon').textContent = '➤';
            messageInput.focus();
        }

        // Focus input on load
        window.addEventListener('load', () => {
            messageInput.focus();
            loadSettings();
        });

        // Settings Modal Functions
        function openSettings() {
            document.getElementById('settingsModal').style.display = 'block';
            loadSettings();
        }

        function closeSettings() {
            document.getElementById('settingsModal').style.display = 'none';
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('settingsModal');
            if (event.target === modal) {
                modal.style.display = 'none';
            }
        }

        async function loadSettings() {
            try {
                const response = await fetch('/api/model');
                const data = await response.json();
                
                document.getElementById('currentModel').textContent = data.model;
                document.getElementById('modelSelect').value = data.model;
                document.getElementById('customPrompt').value = data.custom_prompt || '';
                
            } catch (error) {
                console.error('Error loading settings:', error);
            }
        }

        async function saveSettings() {
            const model = document.getElementById('modelSelect').value;
            const customPrompt = document.getElementById('customPrompt').value.trim();
            
            try {
                const response = await fetch('/api/model', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model: model,
                        custom_prompt: customPrompt
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    document.getElementById('currentModel').textContent = data.model;
                    closeSettings();
                    
                    // Show success message
                    addMessage('assistant', `✅ Settings updated successfully!\n🤖 Model: ${data.model}\n📝 Custom prompt: ${data.custom_prompt ? 'Enabled' : 'Default'}`, new Date().toISOString());
                } else {
                    alert('Error saving settings: ' + data.error);
                }
                
            } catch (error) {
                console.error('Error saving settings:', error);
                alert('Network error saving settings');
            }
        }

        function resetSettings() {
            if (confirm('Reset to default settings? This will remove your custom prompt.')) {
                document.getElementById('modelSelect').value = 'gpt-4o-mini';
                document.getElementById('customPrompt').value = '';
                saveSettings();
            }
        }
    </script>
</body>
</html>
'''


def main():
    """Main function to start the web chat interface."""
    # Load environment variables FIRST before any config creation
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment variables loaded from .env")
    except ImportError:
        print("⚠️  dotenv not installed, trying without .env file")
    
    parser = argparse.ArgumentParser(description="Obsidian Graph RAG Web Chat Interface")
    parser.add_argument('--host', type=str, default=None, help="Host to bind to (overrides .env)")
    parser.add_argument('--port', type=int, default=None, help="Port to bind to (overrides .env)")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--no-browser', action='store_true', help="Don't auto-open browser")
    parser.add_argument('--vault-path', type=str, default=None, help="Path to Obsidian vault (overrides .env)")
    
    args = parser.parse_args()
    
    try:
        # Clean up any existing Python processes to prevent server conflicts
        print("🧹 Cleaning up existing servers...")
        import subprocess
        import platform
        
        try:
            if platform.system() == "Windows":
                # More targeted cleanup - only kill processes on the specific port
                import psutil
                import os
                current_pid = os.getpid()
                
                for proc in psutil.process_iter(['pid', 'name', 'connections']):
                    try:
                        # Skip the current process
                        if proc.info['pid'] == current_pid:
                            continue
                            
                        # Check if it's a Python process using our port
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            for conn in proc.info['connections'] or []:
                                if hasattr(conn, 'laddr') and conn.laddr and conn.laddr.port == config.web_port:
                                    proc.terminate()
                                    print(f"✅ Terminated process {proc.info['pid']} using port {config.web_port}")
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
                        
                print("✅ Cleaned up existing web chat processes on port")
            else:
                # For Unix-like systems, kill processes on the specific port
                subprocess.run(['pkill', '-f', 'web_chat.py'], 
                             capture_output=True, check=False)
                print("✅ Cleaned up existing web chat processes")
        except ImportError:
            # If psutil not available, skip cleanup
            print("⚠️  psutil not available, skipping process cleanup")
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup warning: {cleanup_error}")
        
        print("🚀 Initializing Obsidian Graph RAG Web Interface...")
        
        # Setup configuration (loads from .env)
        config = GraphRAGConfig()
        
        # Override with command line arguments if provided
        if args.vault_path:
            config.vault_path = args.vault_path
        if args.host:
            config.web_host = args.host
        if args.port:
            config.web_port = args.port
        if args.no_browser:
            config.auto_open_browser = False
        
        # Check for OpenAI API key
        if not config.openai_api_key:
            print("\n🔑 OpenAI API Key Required")
            print("To use AI features, you need an OpenAI API key.")
            print("💡 Add it to your .env file: OPENAI_API_KEY=your-key-here")
            print("Or get one from: https://platform.openai.com/api-keys")
            
            api_key = input("Enter your OpenAI API key (or press Enter to exit): ").strip()
            
            if not api_key:
                print("❌ API key required for web chat interface")
                sys.exit(1)
            
            config.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("✓ Found OpenAI API key in .env file")
        
        # Initialize Graph RAG system
        print("📚 Loading knowledge base...")
        graph_rag = ObsidianGraphRAG(config)
        graph_rag.initialize_system()
        
        # Create and start web server
        web_server = WebChatServer(config, graph_rag)
        web_server.run(
            host=config.web_host,
            port=config.web_port,
            debug=args.debug,
            auto_open=config.auto_open_browser
        )
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 