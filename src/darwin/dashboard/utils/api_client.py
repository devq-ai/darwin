"""
Darwin API Client for Dashboard Integration

This module provides a comprehensive API client for communicating with the Darwin
FastAPI backend from the Panel dashboard. It handles all REST API operations,
authentication, error handling, and data serialization.

Features:
- Async HTTP client with connection pooling
- Automatic retry logic with exponential backoff
- Request/response logging and monitoring
- Type-safe data models integration
- Error handling and user-friendly error messages
- Health checking and connection monitoring
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import logfire

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception for API errors."""

    def __init__(
        self, message: str, status_code: int = None, response_data: dict = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class DarwinAPIClient:
    """
    Async HTTP client for Darwin API communication.

    Provides methods for all Darwin API endpoints with proper error handling,
    logging, and type conversion.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the Darwin API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={
                "User-Agent": "Darwin-Dashboard/1.0.0",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        # Track connection status
        self.is_connected = False
        self.last_health_check = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()
            logger.info("API client closed")

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic and error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            **kwargs: Additional httpx request parameters

        Returns:
            Response data as dictionary

        Raises:
            APIError: If request fails after all retries
        """
        url = (
            f"{self.base_url}/api/v1{endpoint}"
            if not endpoint.startswith("/api/")
            else f"{self.base_url}{endpoint}"
        )

        for attempt in range(self.max_retries + 1):
            try:
                # Log request
                logfire.info(
                    f"API request: {method} {url}",
                    method=method,
                    url=url,
                    attempt=attempt + 1,
                    data=data,
                    params=params,
                )

                # Make request
                response = await self.client.request(
                    method=method, url=url, json=data, params=params, **kwargs
                )

                # Log response
                logfire.info(
                    f"API response: {response.status_code}",
                    status_code=response.status_code,
                    response_time=response.elapsed.total_seconds(),
                )

                # Handle response
                if response.status_code == 200:
                    self.is_connected = True
                    return response.json()
                elif response.status_code == 204:
                    self.is_connected = True
                    return {}
                elif response.status_code in [404, 422]:
                    # Don't retry for client errors
                    error_data = {}
                    try:
                        error_data = response.json()
                    except:
                        pass

                    raise APIError(
                        f"Client error: {response.status_code} - {error_data.get('detail', 'Unknown error')}",
                        status_code=response.status_code,
                        response_data=error_data,
                    )
                else:
                    # Server error - may retry
                    if attempt < self.max_retries:
                        delay = self.retry_delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}), retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        error_data = {}
                        try:
                            error_data = response.json()
                        except:
                            pass

                        raise APIError(
                            f"Server error: {response.status_code} - {error_data.get('detail', 'Unknown error')}",
                            status_code=response.status_code,
                            response_data=error_data,
                        )

            except httpx.RequestError as e:
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Connection error (attempt {attempt + 1}), retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    self.is_connected = False
                    raise APIError(f"Connection error: {str(e)}")

            except Exception as e:
                logger.error(f"Unexpected error in API request: {e}")
                self.is_connected = False
                raise APIError(f"Unexpected error: {str(e)}")

        # Should never reach here
        raise APIError("Maximum retries exceeded")

    # Health and Status
    async def check_health(self) -> Optional[Dict[str, Any]]:
        """Check API health status."""
        try:
            response = await self._make_request("GET", "/health")
            self.last_health_check = datetime.utcnow()
            return response
        except APIError:
            self.is_connected = False
            return None

    async def get_system_metrics(self) -> Optional[Dict[str, Any]]:
        """Get system performance metrics."""
        try:
            return await self._make_request("GET", "/metrics/system")
        except APIError:
            return None

    # Optimizer Management
    async def create_optimizer(
        self, problem_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new genetic algorithm optimizer.

        Args:
            problem_data: Problem definition and configuration data

        Returns:
            Optimizer creation response with optimizer_id
        """
        try:
            return await self._make_request("POST", "/optimizers", data=problem_data)
        except APIError as e:
            logger.error(f"Failed to create optimizer: {e}")
            return None

    async def get_optimizer(self, optimizer_id: str) -> Optional[Dict[str, Any]]:
        """Get optimizer details by ID."""
        try:
            return await self._make_request("GET", f"/optimizers/{optimizer_id}")
        except APIError as e:
            logger.error(f"Failed to get optimizer {optimizer_id}: {e}")
            return None

    async def list_optimizers(
        self, skip: int = 0, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List all optimizers with pagination."""
        try:
            response = await self._make_request(
                "GET", "/optimizers", params={"skip": skip, "limit": limit}
            )
            return response.get("optimizers", [])
        except APIError as e:
            logger.error(f"Failed to list optimizers: {e}")
            return []

    async def delete_optimizer(self, optimizer_id: str) -> bool:
        """Delete an optimizer."""
        try:
            await self._make_request("DELETE", f"/optimizers/{optimizer_id}")
            return True
        except APIError as e:
            logger.error(f"Failed to delete optimizer {optimizer_id}: {e}")
            return False

    # Optimization Runs
    async def start_optimization(
        self, optimizer_id: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Start an optimization run."""
        try:
            data = {"optimizer_id": optimizer_id}
            if config:
                data.update(config)

            return await self._make_request(
                "POST", f"/optimizers/{optimizer_id}/run", data=data
            )
        except APIError as e:
            logger.error(f"Failed to start optimization {optimizer_id}: {e}")
            return None

    async def stop_optimization(self, optimizer_id: str) -> bool:
        """Stop a running optimization."""
        try:
            await self._make_request("POST", f"/optimizers/{optimizer_id}/stop")
            return True
        except APIError as e:
            logger.error(f"Failed to stop optimization {optimizer_id}: {e}")
            return False

    async def get_optimization_results(
        self, optimizer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get optimization results."""
        try:
            return await self._make_request(
                "GET", f"/optimizers/{optimizer_id}/results"
            )
        except APIError as e:
            logger.error(f"Failed to get results for {optimizer_id}: {e}")
            return None

    async def get_optimization_progress(
        self, optimizer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get current optimization progress."""
        try:
            return await self._make_request(
                "GET", f"/optimizers/{optimizer_id}/progress"
            )
        except APIError as e:
            logger.error(f"Failed to get progress for {optimizer_id}: {e}")
            return None

    async def get_optimization_history(
        self, optimizer_id: str, include_population: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get optimization evolution history."""
        try:
            params = {"include_population": include_population}
            return await self._make_request(
                "GET", f"/optimizers/{optimizer_id}/history", params=params
            )
        except APIError as e:
            logger.error(f"Failed to get history for {optimizer_id}: {e}")
            return None

    # Templates
    async def get_templates(self) -> List[Dict[str, Any]]:
        """Get all problem templates."""
        try:
            response = await self._make_request("GET", "/templates")
            return response.get("templates", [])
        except APIError as e:
            logger.error(f"Failed to get templates: {e}")
            return []

    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific template by ID."""
        try:
            return await self._make_request("GET", f"/templates/{template_id}")
        except APIError as e:
            logger.error(f"Failed to get template {template_id}: {e}")
            return None

    async def create_template(
        self, template_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a new problem template."""
        try:
            return await self._make_request("POST", "/templates", data=template_data)
        except APIError as e:
            logger.error(f"Failed to create template: {e}")
            return None

    async def delete_template(self, template_id: str) -> bool:
        """Delete a template."""
        try:
            await self._make_request("DELETE", f"/templates/{template_id}")
            return True
        except APIError as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False

    # Algorithms
    async def get_algorithms(self) -> List[Dict[str, Any]]:
        """Get available genetic algorithm configurations."""
        try:
            response = await self._make_request("GET", "/algorithms")
            return response.get("algorithms", [])
        except APIError as e:
            logger.error(f"Failed to get algorithms: {e}")
            return []

    async def get_algorithm_info(self, algorithm_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific algorithm."""
        try:
            return await self._make_request("GET", f"/algorithms/{algorithm_name}")
        except APIError as e:
            logger.error(f"Failed to get algorithm info for {algorithm_name}: {e}")
            return None

    # Statistics and Analytics
    async def get_optimization_stats(self) -> Optional[Dict[str, Any]]:
        """Get optimization statistics summary."""
        try:
            return await self._make_request("GET", "/metrics/optimizations")
        except APIError as e:
            logger.error(f"Failed to get optimization stats: {e}")
            return None

    async def get_performance_metrics(
        self, optimizer_id: str, metric_types: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get performance metrics for an optimization run."""
        try:
            params = {}
            if metric_types:
                params["metric_types"] = ",".join(metric_types)

            return await self._make_request(
                "GET", f"/metrics/performance/{optimizer_id}", params=params
            )
        except APIError as e:
            logger.error(f"Failed to get performance metrics for {optimizer_id}: {e}")
            return None

    # Utility Methods
    async def validate_problem_definition(
        self, problem_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Validate a problem definition without creating an optimizer."""
        try:
            return await self._make_request(
                "POST", "/validate/problem", data=problem_data
            )
        except APIError as e:
            logger.error(f"Failed to validate problem definition: {e}")
            return None

    async def estimate_runtime(
        self, problem_data: Dict[str, Any], config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Estimate optimization runtime based on problem and configuration."""
        try:
            data = {"problem": problem_data, "config": config}
            return await self._make_request("POST", "/estimate/runtime", data=data)
        except APIError as e:
            logger.error(f"Failed to estimate runtime: {e}")
            return None

    async def export_results(
        self, optimizer_id: str, format: str = "json"
    ) -> Optional[bytes]:
        """Export optimization results in specified format."""
        try:
            response = await self.client.get(
                f"{self.base_url}/api/v1/optimizers/{optimizer_id}/export",
                params={"format": format},
            )

            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"Failed to export results: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Failed to export results for {optimizer_id}: {e}")
            return None

    # Batch Operations
    async def batch_create_optimizers(
        self, problems: List[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """Create multiple optimizers in batch."""
        tasks = [self.create_optimizer(problem) for problem in problems]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def batch_start_optimizations(self, optimizer_ids: List[str]) -> List[bool]:
        """Start multiple optimizations in batch."""
        tasks = [self.start_optimization(opt_id) for opt_id in optimizer_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            result is not None and not isinstance(result, Exception)
            for result in results
        ]

    async def batch_get_results(
        self, optimizer_ids: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """Get results for multiple optimizations in batch."""
        tasks = [self.get_optimization_results(opt_id) for opt_id in optimizer_ids]
        return await asyncio.gather(*tasks, return_exceptions=True)


# Singleton instance for global access
_api_client_instance: Optional[DarwinAPIClient] = None


def get_api_client(base_url: str = "http://localhost:8000") -> DarwinAPIClient:
    """Get or create a singleton API client instance."""
    global _api_client_instance

    if _api_client_instance is None or _api_client_instance.base_url != base_url:
        _api_client_instance = DarwinAPIClient(base_url=base_url)

    return _api_client_instance


async def close_api_client():
    """Close the singleton API client instance."""
    global _api_client_instance

    if _api_client_instance:
        await _api_client_instance.close()
        _api_client_instance = None
