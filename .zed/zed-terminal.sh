#!/bin/bash

# Zed Terminal Startup Script for DevQ.ai
# This script ensures proper .zshrc loading in Zed terminal

# Get the directory where this script is located (should be devqai root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up environment for DevQ.ai project
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

# Change to project directory
cd "$DEVQAI_ROOT"

# Create a temporary .zshrc for this session that includes everything
TEMP_ZSHRC=$(mktemp)
cat > "$TEMP_ZSHRC" << 'EOF'
# Source the main .zshrc first
if [ -f "$HOME/.zshrc" ]; then
    source "$HOME/.zshrc"
fi

# Then source the DevQ.ai project configuration
if [ -f "$DEVQAI_ROOT/.zshrc.devqai" ]; then
    source "$DEVQAI_ROOT/.zshrc.devqai"
fi

# Clean up temp file on exit
trap 'rm -f "$TEMP_ZSHRC"' EXIT
EOF

# Start zsh with the temporary configuration
exec /bin/zsh -c "export ZDOTDIR=$(dirname $TEMP_ZSHRC) && export TEMP_ZSHRC=$TEMP_ZSHRC && source $TEMP_ZSHRC && exec zsh"
