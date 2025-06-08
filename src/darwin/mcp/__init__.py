"""
Darwin MCP (Model Context Protocol) Module

This module provides MCP server and client implementations for the Darwin genetic algorithm platform.
It enables integration with MCP-compatible systems and provides tools for creating, running, and
monitoring genetic algorithm optimizations through the MCP protocol.

Components:
- MCPServer: FastAPI-based MCP server implementation
- DarwinMCPClient: Client for interacting with MCP servers
- Protocol models and utilities
"""

from .client import (
    DarwinMCPClient,
    MCPClientError,
    create_simple_optimizer,
    run_simple_optimization,
)
from .server import MCPError, MCPRequest, MCPResponse, MCPServer, app

__all__ = [
    # Server components
    "MCPServer",
    "MCPRequest",
    "MCPResponse",
    "MCPError",
    "app",
    # Client components
    "DarwinMCPClient",
    "MCPClientError",
    "create_simple_optimizer",
    "run_simple_optimization",
]

__version__ = "1.0.0"
