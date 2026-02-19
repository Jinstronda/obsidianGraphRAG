"""
FastAPI Web UI for Obsidian RAG System
Provides a ChatGPT-style interface for querying your Obsidian vault
"""

import sys
import os
import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add parent directory to path to import src modules
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.simple_raganything import SimpleRAGAnything
from src.agentic_rag import AgenticRAG

# Initialize FastAPI app
app = FastAPI(title="Obsidian RAG Chat")

# Global RAG instances
rag_system = None
agentic_rag = None

# Lock to prevent concurrent sync and query operations
# This prevents data corruption when sync runs while queries are in progress
rag_lock = asyncio.Lock()

# Resolve static files path (absolute)
STATIC_DIR = (PROJECT_ROOT / "ui" / "static").resolve()
print(f"[DEBUG] STATIC_DIR resolved to: {STATIC_DIR}")
print(f"[DEBUG] index.html exists: {(STATIC_DIR / 'index.html').exists()}")


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on server startup"""
    global rag_system, agentic_rag

    print("=" * 60)
    print("Starting Obsidian RAG Chat UI")
    print("=" * 60)

    # Configuration from environment or defaults
    vault_path = os.getenv("OBSIDIAN_VAULT_PATH")
    if not vault_path:
        raise RuntimeError("OBSIDIAN_VAULT_PATH not set. Set it in .env or export it.")
    working_dir = os.getenv("RAG_WORKING_DIR", "./rag_storage")

    print(f"Vault: {vault_path}")
    print(f"Storage: {working_dir}")
    print()

    # Initialize RAG system (non-interactive mode for web server)
    try:
        rag_system = SimpleRAGAnything(
            vault_path=vault_path,
            working_dir=working_dir,
            non_interactive=True  # Skip interactive prompts in web server mode
        )

        print("Initializing RAG system...")
        await rag_system.initialize()

        # Initialize AgenticRAG wrapper for CRAG mode
        agentic_rag = AgenticRAG(rag_system)
        print("AgenticRAG (CRAG) initialized with Gemini 3 Flash")

        print()
        print("RAG system ready!")
        print("=" * 60)

    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        print("=" * 60)
        raise


@app.get("/")
async def get_chat_page():
    """Serve the main chat interface"""
    html_path = STATIC_DIR / "index.html"
    print(f"[DEBUG] Serving index.html from: {html_path}")
    if not html_path.exists():
        print(f"[ERROR] index.html NOT FOUND at {html_path}")
        return HTMLResponse(content=f"File not found: {html_path}", status_code=404)
    return FileResponse(html_path, media_type="text/html")


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat with streaming responses

    Protocol:
    - Client sends: {"message": "user question", "agentic": false}
    - Server streams: {"type": "chunk", "content": "..."} for each chunk
    - Server sends: {"type": "step", "content": {...}} for agentic mode steps
    - Server sends: {"type": "done"} when complete
    - Server sends: {"type": "error", "content": "..."} on error
    """
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "").strip()
            query_mode = message_data.get("mode", "hybrid")
            enable_agentic = message_data.get("agentic", False)
            conversation_history = message_data.get("history", [])

            if not user_message:
                await websocket.send_json({
                    "type": "error",
                    "content": "Empty message received"
                })
                continue

            # Log query to terminal
            print(f"\nUser: {user_message}")
            print(f"Mode: {query_mode}, Agentic: {enable_agentic}, History: {len(conversation_history)} msgs")

            try:
                # Acquire lock to prevent queries during sync
                async with rag_lock:
                    if enable_agentic and agentic_rag:
                        # Use CRAG agentic mode with conversation history
                        print("Using CRAG Agent with Gemini 3 Flash...")
                        async for event in agentic_rag.query_streaming(user_message, history=conversation_history):
                            event_type = event.get("type")

                            if event_type == "step":
                                # Send agent step to UI
                                await websocket.send_json({
                                    "type": "step",
                                    "content": event["content"]
                                })
                                step = event["content"]
                                print(f"  [{step['step']}] {step['tool']}({step['args']})")

                            elif event_type == "token_update":
                                # Send token usage update to UI
                                await websocket.send_json({
                                    "type": "token_update",
                                    "content": event["content"]
                                })
                                tokens = event["content"]["tokens"]
                                percent = event["content"]["percent"]
                                print(f"  [Tokens] {tokens:,} ({percent:.1f}% of limit)")

                            elif event_type == "context_compacted":
                                # Notify UI that context was compacted
                                await websocket.send_json({
                                    "type": "context_compacted",
                                    "content": event["content"]
                                })
                                count = event["content"]["compaction_count"]
                                print(f"  [Context] Compacted (#{count})")

                            elif event_type == "thinking":
                                # Optional: send thinking to UI
                                pass

                            elif event_type == "answer":
                                # Stream the final answer word by word
                                answer = event["content"]["answer"]
                                confidence = event["content"]["confidence"]
                                sources = event["content"].get("sources_used", 0)
                                print(f"Answer (confidence: {confidence}, sources: {sources}):")
                                print(answer[:200] + "..." if len(answer) > 200 else answer)

                                # Send full answer at once
                                await websocket.send_json({
                                    "type": "chunk",
                                    "content": answer
                                })

                                # Send metadata
                                await websocket.send_json({
                                    "type": "metadata",
                                    "content": {
                                        "confidence": confidence,
                                        "sources_used": sources
                                    }
                                })

                            elif event_type == "done":
                                await websocket.send_json({"type": "done"})

                    else:
                        # Standard RAG query
                        print("Using standard RAG...")
                        response = await rag_system.query(user_message, mode=query_mode)
                        print(response[:200] + "..." if len(response) > 200 else response)

                        # Send full response at once
                        await websocket.send_json({
                            "type": "chunk",
                            "content": response
                        })

                        await websocket.send_json({"type": "done"})

            except Exception as e:
                error_msg = f"Query failed: {str(e)}"
                print(f"Error: {error_msg}")
                import traceback
                traceback.print_exc()

                await websocket.send_json({
                    "type": "error",
                    "content": error_msg
                })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy",
        "rag_initialized": rag_system is not None,
        "gpu_available": False,
    }

    try:
        import psutil
        health_data["memory_usage_mb"] = psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        pass

    try:
        import torch
        health_data["gpu_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            health_data["gpu_name"] = torch.cuda.get_device_name(0)
            health_data["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(0) / 1024 / 1024
            health_data["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(0) / 1024 / 1024
    except ImportError:
        pass

    return health_data


@app.get("/api/config")
async def get_config():
    """Get current RAG system configuration"""
    if not rag_system:
        return {"error": "RAG system not initialized"}

    return {
        "vault_path": rag_system.vault_path,
        "working_dir": rag_system.working_dir,
        "using_existing_db": rag_system.using_existing_db,
        "non_interactive": rag_system.non_interactive,
        "agentic_available": agentic_rag is not None,
        "models": {
            "embedding": "EmbeddingGemma 308M",
            "llm": "Gemini 3 Flash",
            "agentic_llm": "Gemini 3 Flash",
            "reranker": "BGE Reranker v2-m3"
        }
    }


@app.post("/api/sync")
async def sync_vault():
    """Incrementally sync vault changes"""
    if not rag_system:
        return {"error": "RAG system not initialized"}

    # Acquire lock to prevent sync during queries (and prevent concurrent syncs)
    async with rag_lock:
        try:
            # Import here to avoid circular dependencies
            from src.vault_monitor import VaultMonitor, IncrementalVaultUpdater

            # Initialize monitor
            tracking_file = os.path.join(rag_system.working_dir, "vault_tracking.json")
            monitor = VaultMonitor(rag_system.vault_path, tracking_file)

            # Scan for changes in a thread so the event loop isn't blocked
            # (os.walk + MD5 hashing is I/O-heavy)
            new_files, modified_files, deleted_files = await asyncio.to_thread(
                monitor.scan_vault
            )

            # Create updater and sync
            updater = IncrementalVaultUpdater(rag_system, monitor)
            await updater.sync_vault()

            return {
                "status": "success",
                "changes": {
                    "new": len(new_files),
                    "modified": len(modified_files),
                    "deleted": len(deleted_files)
                },
                "total_tracked": len(monitor.file_registry)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


@app.get("/api/vault/status")
async def vault_status():
    """Get vault sync status"""
    if not rag_system:
        return {"error": "RAG system not initialized"}
    
    try:
        from src.vault_monitor import VaultMonitor

        tracking_file = os.path.join(rag_system.working_dir, "vault_tracking.json")
        monitor = VaultMonitor(rag_system.vault_path, tracking_file)

        # Scan in thread so it doesn't block the event loop
        new_files, modified_files, deleted_files = await asyncio.to_thread(
            monitor.scan_vault
        )
        
        return {
            "status": "success",
            "vault_path": rag_system.vault_path,
            "total_tracked": len(monitor.file_registry),
            "pending_changes": {
                "new": len(new_files),
                "modified": len(modified_files),
                "deleted": len(deleted_files)
            },
            "needs_sync": len(new_files) + len(modified_files) + len(deleted_files) > 0
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# Mount static files LAST - after all routes to avoid catch-all interference
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
