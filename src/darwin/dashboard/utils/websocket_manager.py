"""
WebSocket Manager for Darwin Dashboard

This module provides WebSocket client functionality for receiving real-time updates
from the Darwin API server. It handles connection management, message parsing,
automatic reconnection, and error handling.

Features:
- Automatic connection and reconnection with exponential backoff
- Message queue for reliable delivery
- Type-safe message handling
- Connection status monitoring
- Graceful error handling and logging
- Support for multiple message types
"""

import asyncio
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import logfire
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """WebSocket connection status enumeration."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class MessageType(Enum):
    """WebSocket message type enumeration."""

    OPTIMIZATION_PROGRESS = "optimization_progress"
    OPTIMIZATION_COMPLETE = "optimization_complete"
    OPTIMIZATION_ERROR = "optimization_error"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class WebSocketMessage:
    """Structured WebSocket message wrapper."""

    def __init__(
        self, type: str, data: Dict[str, Any], timestamp: Optional[datetime] = None
    ):
        self.type = type
        self.data = data
        self.timestamp = timestamp or datetime.utcnow()

    @classmethod
    def from_json(cls, json_str: str) -> "WebSocketMessage":
        """Create message from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(
                type=data.get("type", "unknown"),
                data=data.get("data", {}),
                timestamp=datetime.fromisoformat(
                    data.get("timestamp", datetime.utcnow().isoformat())
                ),
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
            return cls(type="error", data={"error": str(e), "raw_message": json_str})

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
        }


class WebSocketManager:
    """
    WebSocket client manager for Darwin dashboard real-time updates.

    Handles connection lifecycle, message processing, and error recovery
    with automatic reconnection and exponential backoff.
    """

    def __init__(
        self,
        websocket_url: str,
        max_reconnect_attempts: int = 10,
        initial_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize WebSocket manager.

        Args:
            websocket_url: WebSocket server URL
            max_reconnect_attempts: Maximum reconnection attempts
            initial_reconnect_delay: Initial delay between reconnection attempts
            max_reconnect_delay: Maximum delay between reconnection attempts
            heartbeat_interval: Heartbeat ping interval in seconds
        """
        self.websocket_url = websocket_url
        self.max_reconnect_attempts = max_reconnect_attempts
        self.initial_reconnect_delay = initial_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self.heartbeat_interval = heartbeat_interval

        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.status = ConnectionStatus.DISCONNECTED
        self.reconnect_attempts = 0
        self.last_heartbeat = None

        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

        # Background tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._message_processor_task: Optional[asyncio.Task] = None

        # Event flags
        self._shutdown_event = asyncio.Event()
        self._connected_event = asyncio.Event()

    async def connect(self) -> bool:
        """
        Connect to the WebSocket server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.status = ConnectionStatus.CONNECTING

            logfire.info(
                "Connecting to WebSocket server",
                url=self.websocket_url,
                attempt=self.reconnect_attempts + 1,
            )

            # Connect to WebSocket
            self.websocket = await websockets.connect(
                self.websocket_url,
                ping_interval=self.heartbeat_interval,
                ping_timeout=10,
                close_timeout=5,
                max_size=1024 * 1024,  # 1MB message size limit
                compression=None,
            )

            # Update connection status
            self.status = ConnectionStatus.CONNECTED
            self.reconnect_attempts = 0
            self.last_heartbeat = datetime.utcnow()
            self._connected_event.set()

            # Start background tasks
            await self._start_background_tasks()

            logfire.info("WebSocket connection established successfully")
            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            logfire.error(
                "Failed to connect to WebSocket server",
                error=str(e),
                url=self.websocket_url,
            )
            return False

    async def disconnect(self):
        """Gracefully disconnect from WebSocket server."""
        try:
            self._shutdown_event.set()

            # Stop background tasks
            await self._stop_background_tasks()

            # Close WebSocket connection
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()

            self.status = ConnectionStatus.DISCONNECTED
            self._connected_event.clear()

            logfire.info("WebSocket connection closed")

        except Exception as e:
            logger.error(f"Error during WebSocket disconnect: {e}")

    async def send_message(self, message: Dict[str, Any]) -> bool:
        """
        Send a message to the WebSocket server.

        Args:
            message: Message data to send

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if not self.is_connected():
                logger.warning("Cannot send message: WebSocket not connected")
                return False

            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.utcnow().isoformat()

            # Send message
            await self.websocket.send(json.dumps(message))

            logfire.debug(
                "WebSocket message sent",
                message_type=message.get("type", "unknown"),
                message_size=len(json.dumps(message)),
            )

            return True

        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False

    async def listen(self) -> AsyncGenerator[WebSocketMessage, None]:
        """
        Listen for incoming WebSocket messages.

        Yields:
            WebSocketMessage objects as they arrive
        """
        while not self._shutdown_event.is_set():
            try:
                # Wait for connection
                if not self.is_connected():
                    await self._wait_for_connection()
                    if self._shutdown_event.is_set():
                        break

                # Receive message with timeout
                try:
                    raw_message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=1.0,  # 1 second timeout to check shutdown event
                    )

                    # Parse and yield message
                    message = WebSocketMessage.from_json(raw_message)

                    # Update heartbeat timestamp for heartbeat messages
                    if message.type == MessageType.HEARTBEAT.value:
                        self.last_heartbeat = datetime.utcnow()

                    # Log message received
                    logfire.debug(
                        "WebSocket message received",
                        message_type=message.type,
                        data_keys=list(message.data.keys())
                        if isinstance(message.data, dict)
                        else [],
                    )

                    yield message

                except asyncio.TimeoutError:
                    # Timeout is expected, continue listening
                    continue

                except ConnectionClosed:
                    logger.warning("WebSocket connection closed by server")
                    await self._handle_disconnection()

                except WebSocketException as e:
                    logger.error(f"WebSocket error: {e}")
                    await self._handle_disconnection()

            except Exception as e:
                logger.error(f"Unexpected error in WebSocket listener: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return (
            self.status == ConnectionStatus.CONNECTED
            and self.websocket is not None
            and not self.websocket.closed
        )

    def add_message_handler(
        self, message_type: str, handler: Callable[[WebSocketMessage], None]
    ):
        """
        Add a message handler for specific message types.

        Args:
            message_type: Type of message to handle
            handler: Callable to handle the message
        """
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []

        self.message_handlers[message_type].append(handler)
        logger.info(f"Added message handler for type: {message_type}")

    def remove_message_handler(
        self, message_type: str, handler: Callable[[WebSocketMessage], None]
    ):
        """Remove a message handler."""
        if message_type in self.message_handlers:
            try:
                self.message_handlers[message_type].remove(handler)
                logger.info(f"Removed message handler for type: {message_type}")
            except ValueError:
                logger.warning(f"Handler not found for message type: {message_type}")

    async def _start_background_tasks(self):
        """Start background tasks for heartbeat and message processing."""
        # Start heartbeat task
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start message processor task
        if self._message_processor_task is None or self._message_processor_task.done():
            self._message_processor_task = asyncio.create_task(self._process_messages())

    async def _stop_background_tasks(self):
        """Stop all background tasks."""
        tasks = [
            self._heartbeat_task,
            self._reconnect_task,
            self._message_processor_task,
        ]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _heartbeat_loop(self):
        """Background task to send periodic heartbeat messages."""
        try:
            while not self._shutdown_event.is_set():
                if self.is_connected():
                    # Send heartbeat
                    heartbeat_msg = {
                        "type": MessageType.HEARTBEAT.value,
                        "data": {"timestamp": datetime.utcnow().isoformat()},
                    }

                    await self.send_message(heartbeat_msg)

                    # Check for stale connection
                    if (
                        self.last_heartbeat
                        and (datetime.utcnow() - self.last_heartbeat).total_seconds()
                        > self.heartbeat_interval * 3
                    ):
                        logger.warning("Heartbeat timeout detected, reconnecting...")
                        await self._handle_disconnection()

                await asyncio.sleep(self.heartbeat_interval)

        except asyncio.CancelledError:
            logger.info("Heartbeat loop cancelled")
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")

    async def _process_messages(self):
        """Background task to process messages through registered handlers."""
        try:
            async for message in self.listen():
                # Call registered handlers for this message type
                handlers = self.message_handlers.get(message.type, [])

                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        logger.error(f"Error in message handler: {e}")

        except asyncio.CancelledError:
            logger.info("Message processor cancelled")
        except Exception as e:
            logger.error(f"Error in message processor: {e}")

    async def _handle_disconnection(self):
        """Handle WebSocket disconnection and attempt reconnection."""
        if self.status == ConnectionStatus.RECONNECTING:
            return  # Already handling disconnection

        self.status = ConnectionStatus.RECONNECTING
        self._connected_event.clear()

        # Close existing connection
        if self.websocket and not self.websocket.closed:
            try:
                await self.websocket.close()
            except:
                pass

        # Start reconnection task
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Background task to handle automatic reconnection."""
        try:
            while (
                not self._shutdown_event.is_set()
                and self.reconnect_attempts < self.max_reconnect_attempts
                and not self.is_connected()
            ):
                self.reconnect_attempts += 1

                # Calculate delay with exponential backoff
                delay = min(
                    self.initial_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
                    self.max_reconnect_delay,
                )

                logfire.info(
                    "Attempting WebSocket reconnection",
                    attempt=self.reconnect_attempts,
                    max_attempts=self.max_reconnect_attempts,
                    delay=delay,
                )

                # Wait before reconnecting
                await asyncio.sleep(delay)

                # Attempt reconnection
                if await self.connect():
                    logfire.info("WebSocket reconnection successful")
                    return

            # Max attempts reached
            if self.reconnect_attempts >= self.max_reconnect_attempts:
                self.status = ConnectionStatus.ERROR
                logfire.error("Max reconnection attempts reached, giving up")

        except asyncio.CancelledError:
            logger.info("Reconnection loop cancelled")
        except Exception as e:
            logger.error(f"Error in reconnection loop: {e}")
            self.status = ConnectionStatus.ERROR

    async def _wait_for_connection(self, timeout: float = 30.0):
        """Wait for WebSocket connection to be established."""
        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for WebSocket connection")

    def get_connection_info(self) -> Dict[str, Any]:
        """Get current connection information."""
        return {
            "url": self.websocket_url,
            "status": self.status.value,
            "connected": self.is_connected(),
            "reconnect_attempts": self.reconnect_attempts,
            "last_heartbeat": self.last_heartbeat.isoformat()
            if self.last_heartbeat
            else None,
            "message_handlers": {
                msg_type: len(handlers)
                for msg_type, handlers in self.message_handlers.items()
            },
        }
