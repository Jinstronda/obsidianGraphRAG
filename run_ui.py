"""
Launcher script for Obsidian RAG Chat UI
Starts the FastAPI server with proper configuration
"""

import os
import sys
import socket


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available port found in range {start_port}-{start_port + max_attempts}")


if __name__ == "__main__":
    import uvicorn

    port = find_available_port(8000)

    print("\n" + "=" * 60)
    print("Starting Obsidian RAG Chat UI Server")
    print("=" * 60)
    print(f"URL: http://localhost:{port}")
    print("Press CTRL+C to stop the server")
    print("=" * 60 + "\n")

    dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

    if dev_mode:
        print("Development mode: Auto-reload enabled")
        print("(Set DEV_MODE=false to disable)\n")

    uvicorn.run(
        "ui.app:app",
        host="0.0.0.0",
        port=port,
        reload=dev_mode,
        log_level="info"
    )
