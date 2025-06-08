"""
Darwin Dashboard Utilities Package

This package contains utility modules and classes that support the Darwin
dashboard components. These utilities handle communication with the backend
API, WebSocket connections, data processing, and other shared functionality.

Modules:
- api_client: HTTP client for Darwin API communication
- websocket_manager: WebSocket client for real-time updates
- data_processing: Data transformation and analysis utilities
- ui_helpers: Common UI helper functions and widgets
- validation: Data validation and schema checking
- export: Data export and file handling utilities

Key Classes:
- DarwinAPIClient: Async HTTP client for REST API communication
- WebSocketManager: WebSocket connection and message handling
- DataProcessor: Data transformation and analysis tools
- UIHelpers: Common UI components and styling utilities

The utilities are designed to be:
- Reusable across different dashboard components
- Async-first for optimal performance
- Well-documented and type-annotated
- Robust with proper error handling
- Testable with clear interfaces

Usage:
    from darwin.dashboard.utils import DarwinAPIClient, WebSocketManager

    # Create API client
    api_client = DarwinAPIClient("http://localhost:8000")

    # Create WebSocket manager
    ws_manager = WebSocketManager("ws://localhost:8000/ws/optimization/progress")

    # Use in dashboard components
    await api_client.get_optimization_results("optimizer_123")
    await ws_manager.connect()
"""

from .api_client import APIError, DarwinAPIClient, close_api_client, get_api_client
from .websocket_manager import (
    ConnectionStatus,
    MessageType,
    WebSocketManager,
    WebSocketMessage,
)

__version__ = "1.0.0"

__all__ = [
    # API client
    "DarwinAPIClient",
    "APIError",
    "get_api_client",
    "close_api_client",
    # WebSocket manager
    "WebSocketManager",
    "WebSocketMessage",
    "ConnectionStatus",
    "MessageType",
]
