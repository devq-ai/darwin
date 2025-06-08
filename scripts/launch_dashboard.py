#!/usr/bin/env python3
"""
Darwin Dashboard Launcher Script

This script provides a simple command-line interface for launching the Darwin
genetic algorithm optimization dashboard. It handles environment setup,
configuration validation, and graceful startup/shutdown.

Features:
- Command-line argument parsing for configuration
- Environment variable validation
- API server health checking
- Graceful shutdown handling
- Development and production modes
- Logging configuration
- Error handling and recovery

Usage:
    python scripts/launch_dashboard.py --port 5007 --api-url http://localhost:8000
    python scripts/launch_dashboard.py --dev --autoreload
    python scripts/launch_dashboard.py --config config/dashboard.yaml
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import uvloop
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add the src directory to the Python path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from darwin.dashboard import DarwinDashboard, create_app
from darwin.dashboard.utils import DarwinAPIClient

# Global console for rich output
console = Console()

# Global dashboard instance for cleanup
dashboard_instance: Optional[DarwinDashboard] = None


def setup_logging(level: str = "INFO", rich_logging: bool = True) -> None:
    """Setup logging configuration."""

    log_level = getattr(logging, level.upper(), logging.INFO)

    if rich_logging:
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Suppress noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or environment."""

    config = {
        "dashboard": {
            "port": 5007,
            "host": "0.0.0.0",
            "title": "Darwin Genetic Algorithm Optimizer",
            "show_browser": True,
            "autoreload": False,
        },
        "api": {"base_url": "http://localhost:8000", "timeout": 30.0, "max_retries": 3},
        "websocket": {
            "url": "ws://localhost:8000/ws/optimization/progress",
            "max_reconnect_attempts": 10,
            "heartbeat_interval": 30.0,
        },
        "logging": {"level": "INFO", "rich_logging": True},
    }

    # Load from config file if provided
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                file_config = yaml.safe_load(f)
                config.update(file_config)
        except Exception as e:
            console.print(f"[red]Error loading config file: {e}[/red]")
            sys.exit(1)

    # Override with environment variables
    env_overrides = {
        "DARWIN_DASHBOARD_PORT": ("dashboard", "port", int),
        "DARWIN_DASHBOARD_HOST": ("dashboard", "host", str),
        "DARWIN_API_URL": ("api", "base_url", str),
        "DARWIN_API_TIMEOUT": ("api", "timeout", float),
        "DARWIN_WS_URL": ("websocket", "url", str),
        "DARWIN_LOG_LEVEL": ("logging", "level", str),
    }

    for env_var, (section, key, type_func) in env_overrides.items():
        if env_var in os.environ:
            try:
                config[section][key] = type_func(os.environ[env_var])
            except (ValueError, TypeError) as e:
                console.print(f"[red]Invalid environment variable {env_var}: {e}[/red]")
                sys.exit(1)

    return config


async def check_api_health(api_url: str, timeout: float = 10.0) -> bool:
    """Check if the Darwin API server is healthy."""

    try:
        api_client = DarwinAPIClient(base_url=api_url, timeout=timeout)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Checking API health...", total=None)

            health = await api_client.check_health()
            await api_client.close()

            if health and health.get("status") == "healthy":
                console.print("[green]✓[/green] API server is healthy")
                return True
            else:
                console.print(
                    "[yellow]⚠[/yellow] API server responded but is not healthy"
                )
                return False

    except Exception as e:
        console.print(f"[red]✗[/red] API server health check failed: {e}")
        return False


def print_startup_banner(config: Dict[str, Any]) -> None:
    """Print startup banner with configuration info."""

    banner = Panel.fit(
        f"""[bold blue]🧬 Darwin Genetic Algorithm Optimizer[/bold blue]

[bold]Dashboard Configuration:[/bold]
• Port: {config['dashboard']['port']}
• Host: {config['dashboard']['host']}
• API URL: {config['api']['base_url']}
• WebSocket: {config['websocket']['url']}
• Auto-reload: {config['dashboard']['autoreload']}

[bold]Controls:[/bold]
• Press Ctrl+C to stop the server
• Visit the dashboard in your browser
• Check the console for real-time logs""",
        title="🚀 Starting Darwin Dashboard",
        title_align="center",
        border_style="blue",
    )

    console.print(banner)


def print_shutdown_banner() -> None:
    """Print shutdown banner."""

    banner = Panel.fit(
        """[bold yellow]Dashboard server has been stopped[/bold yellow]

Thank you for using Darwin!

For support and documentation:
• GitHub: https://github.com/devqai/darwin
• Docs: https://darwin.devq.ai
• Issues: https://github.com/devqai/darwin/issues""",
        title="👋 Goodbye",
        title_align="center",
        border_style="yellow",
    )

    console.print(banner)


def setup_signal_handlers() -> None:
    """Setup graceful shutdown signal handlers."""

    def signal_handler(signum: int, frame) -> None:
        console.print(
            f"\n[yellow]Received signal {signum}, shutting down gracefully...[/yellow]"
        )

        if dashboard_instance:
            # TODO: Implement graceful shutdown
            pass

        print_shutdown_banner()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_dashboard(config: Dict[str, Any]) -> None:
    """Run the dashboard with the given configuration."""

    global dashboard_instance

    try:
        # Create dashboard instance
        dashboard_instance = create_app(api_base_url=config["api"]["base_url"])

        console.print("[green]✓[/green] Dashboard created successfully")

        # Print startup information
        print_startup_banner(config)

        # Show URL information
        dashboard_url = (
            f"http://{config['dashboard']['host']}:{config['dashboard']['port']}"
        )
        if config["dashboard"]["host"] == "0.0.0.0":
            dashboard_url = f"http://localhost:{config['dashboard']['port']}"

        url_info = Table(show_header=False, box=None, padding=(0, 1))
        url_info.add_row("🌐 Dashboard URL:", f"[bold blue]{dashboard_url}[/bold blue]")
        url_info.add_row(
            "📡 API Server:", f"[bold green]{config['api']['base_url']}[/bold green]"
        )
        url_info.add_row(
            "🔌 WebSocket:", f"[bold cyan]{config['websocket']['url']}[/bold cyan]"
        )

        console.print(url_info)
        console.print()

        # Start the dashboard server
        console.print("[bold green]Starting dashboard server...[/bold green]")

        dashboard_instance.serve(
            port=config["dashboard"]["port"],
            show=config["dashboard"]["show_browser"],
            autoreload=config["dashboard"]["autoreload"],
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Received keyboard interrupt[/yellow]")

    except Exception as e:
        console.print(f"[red]Error running dashboard: {e}[/red]")
        raise

    finally:
        # Cleanup
        if dashboard_instance:
            # TODO: Implement proper cleanup
            pass


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Launch the Darwin genetic algorithm optimization dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_dashboard.py
  python launch_dashboard.py --port 8080 --api-url http://api.example.com:8000
  python launch_dashboard.py --dev --autoreload --no-browser
  python launch_dashboard.py --config config/production.yaml
        """,
    )

    # Dashboard configuration
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5007,
        help="Port to serve the dashboard on (default: 5007)",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Darwin API server URL (default: http://localhost:8000)",
    )

    parser.add_argument(
        "--ws-url", help="WebSocket URL (default: derived from API URL)"
    )

    # Behavior options
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )

    parser.add_argument(
        "--autoreload", action="store_true", help="Enable auto-reload for development"
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (autoreload, debug logging)",
    )

    # Configuration
    parser.add_argument(
        "--config", "-c", help="Path to configuration file (YAML format)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--no-rich", action="store_true", help="Disable rich console output"
    )

    # Health check options
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip API server health check on startup",
    )

    parser.add_argument(
        "--health-timeout",
        type=float,
        default=10.0,
        help="Timeout for health check in seconds (default: 10.0)",
    )

    return parser.parse_args()


async def main() -> None:
    """Main entry point."""

    # Parse command-line arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(
        level=args.log_level if not args.dev else "DEBUG", rich_logging=not args.no_rich
    )

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.port != 5007:
        config["dashboard"]["port"] = args.port
    if args.host != "0.0.0.0":
        config["dashboard"]["host"] = args.host
    if args.api_url != "http://localhost:8000":
        config["api"]["base_url"] = args.api_url
    if args.ws_url:
        config["websocket"]["url"] = args.ws_url
    else:
        # Derive WebSocket URL from API URL
        api_url = config["api"]["base_url"]
        ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
        config["websocket"]["url"] = f"{ws_url}/ws/optimization/progress"

    config["dashboard"]["show_browser"] = not args.no_browser
    config["dashboard"]["autoreload"] = args.autoreload or args.dev
    config["logging"]["level"] = args.log_level if not args.dev else "DEBUG"
    config["logging"]["rich_logging"] = not args.no_rich

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    # Check API server health (unless skipped)
    if not args.skip_health_check:
        console.print("[bold]Checking API server health...[/bold]")

        api_healthy = await check_api_health(
            config["api"]["base_url"], timeout=args.health_timeout
        )

        if not api_healthy:
            console.print(
                "[yellow]⚠ Warning: API server is not responding or unhealthy[/yellow]"
            )
            console.print(
                "[yellow]The dashboard will still start, but functionality may be limited[/yellow]"
            )
            console.print()

    # Run the dashboard
    await run_dashboard(config)


if __name__ == "__main__":
    # Use uvloop for better performance if available
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {e}[/red]")
        sys.exit(1)
