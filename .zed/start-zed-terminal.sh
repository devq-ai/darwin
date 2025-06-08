#!/bin/bash

# Zed Terminal Startup Script for DevQ.ai
# This script sets up the terminal environment for development

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set environment variables
export DEVQAI_ROOT="$SCRIPT_DIR"
export PYTHONPATH="$DEVQAI_ROOT:$PYTHONPATH"
export MCP_SERVERS_PATH="$DEVQAI_ROOT/mcp/mcp-servers"
export PTOLEMIES_PATH="$DEVQAI_ROOT/ptolemies"

# Database configuration
export SURREALDB_URL="ws://localhost:8000/rpc"
export SURREALDB_USERNAME="root"
export SURREALDB_PASSWORD="root"
export SURREALDB_NAMESPACE="ptolemies"
export SURREALDB_DATABASE="knowledge"

# Dart AI Configuration
export DART_TOKEN="dsa_1a21dba13961ac8abbe58ea7f9cb7d5621148dc2f3c79a9d346ef40430795e8f"

# Load environment variables from .env if it exists
if [ -f "$DEVQAI_ROOT/.env" ]; then
    set -a
    source "$DEVQAI_ROOT/.env"
    set +a
fi

# Start zsh with proper configuration
exec /bin/zsh -l -c "
    # Source main zshrc
    source ~/.zshrc 2>/dev/null || true

    # Source project-specific configuration
    source '$DEVQAI_ROOT/.zshrc.devqai' 2>/dev/null || true

    # Start interactive zsh
    exec zsh
"
