"""
Responsive Design Utilities for Darwin Dashboard

This module provides comprehensive responsive design utilities for the Panel dashboard,
including adaptive layouts, dynamic sizing, mobile-friendly components, and
responsive visualization configurations.

Features:
- Viewport-aware component sizing
- Adaptive grid layouts for different screen sizes
- Mobile-friendly navigation and controls
- Dynamic plot sizing and responsive visualizations
- Accessibility enhancements
- Theme switching capabilities
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import panel as pn
import param
from bokeh.models import CustomJS

logger = logging.getLogger(__name__)


class ViewportManager(param.Parameterized):
    """Manages viewport detection and responsive behavior."""

    # Viewport parameters
    viewport_width = param.Integer(default=1200, bounds=(320, 3840))
    viewport_height = param.Integer(default=800, bounds=(240, 2160))
    device_type = param.Selector(
        default="desktop", objects=["mobile", "tablet", "desktop"]
    )
    orientation = param.Selector(default="landscape", objects=["portrait", "landscape"])

    # Breakpoints (following Material Design principles)
    BREAKPOINTS = {
        "xs": 320,  # Extra small devices (phones)
        "sm": 576,  # Small devices (phones)
        "md": 768,  # Medium devices (tablets)
        "lg": 992,  # Large devices (desktops)
        "xl": 1200,  # Extra large devices (large desktops)
        "xxl": 1400,  # Extra extra large devices
    }

    def __init__(self, **params):
        super().__init__(**params)
        self._setup_viewport_detection()

    def _setup_viewport_detection(self):
        """Setup JavaScript-based viewport detection."""
        self.viewport_js = CustomJS(
            code="""
            function updateViewport() {
                const width = window.innerWidth;
                const height = window.innerHeight;
                const deviceType = width < 768 ? 'mobile' :
                                 width < 992 ? 'tablet' : 'desktop';
                const orientation = width > height ? 'landscape' : 'portrait';

                // Update Python parameters
                python_obj.viewport_width = width;
                python_obj.viewport_height = height;
                python_obj.device_type = deviceType;
                python_obj.orientation = orientation;
            }

            // Initial detection
            updateViewport();

            // Listen for resize events
            window.addEventListener('resize', updateViewport);
            window.addEventListener('orientationchange', updateViewport);
        """
        )

    def get_current_breakpoint(self) -> str:
        """Get the current breakpoint based on viewport width."""
        width = self.viewport_width

        if width < self.BREAKPOINTS["sm"]:
            return "xs"
        elif width < self.BREAKPOINTS["md"]:
            return "sm"
        elif width < self.BREAKPOINTS["lg"]:
            return "md"
        elif width < self.BREAKPOINTS["xl"]:
            return "lg"
        elif width < self.BREAKPOINTS["xxl"]:
            return "xl"
        else:
            return "xxl"

    def is_mobile(self) -> bool:
        """Check if current viewport is mobile."""
        return self.device_type == "mobile"

    def is_tablet(self) -> bool:
        """Check if current viewport is tablet."""
        return self.device_type == "tablet"

    def is_desktop(self) -> bool:
        """Check if current viewport is desktop."""
        return self.device_type == "desktop"


class ResponsiveGrid:
    """Responsive grid system for organizing dashboard components."""

    def __init__(self, viewport_manager: ViewportManager):
        self.viewport_manager = viewport_manager

    def create_grid(
        self, components: List[Any], columns: Dict[str, int] = None
    ) -> pn.GridSpec:
        """Create a responsive grid layout."""
        if columns is None:
            columns = {"xs": 1, "sm": 1, "md": 2, "lg": 3, "xl": 4, "xxl": 5}

        breakpoint = self.viewport_manager.get_current_breakpoint()
        cols = columns.get(breakpoint, 2)

        # Calculate grid dimensions
        rows = (len(components) + cols - 1) // cols

        # Create grid
        grid = pn.GridSpec(
            width=self.viewport_manager.viewport_width,
            height=min(800, self.viewport_manager.viewport_height - 100),
            ncols=cols,
            nrows=rows,
            sizing_mode="stretch_width",
        )

        # Place components in grid
        for i, component in enumerate(components):
            row = i // cols
            col = i % cols
            grid[row, col] = component

        return grid

    def create_responsive_row(
        self, components: List[Any], weights: Dict[str, List[float]] = None
    ) -> pn.Row:
        """Create a responsive row with adaptive component sizing."""
        if weights is None:
            # Default equal weights
            weights = {
                bp: [1.0] * len(components)
                for bp in ["xs", "sm", "md", "lg", "xl", "xxl"]
            }

        breakpoint = self.viewport_manager.get_current_breakpoint()
        component_weights = weights.get(breakpoint, [1.0] * len(components))

        # Apply sizing based on viewport
        if self.viewport_manager.is_mobile():
            # Stack components vertically on mobile
            return pn.Column(*components, sizing_mode="stretch_width")
        else:
            # Use row layout with responsive sizing
            row_components = []
            for component, weight in zip(components, component_weights):
                if hasattr(component, "width"):
                    component.width = int(
                        self.viewport_manager.viewport_width
                        * weight
                        / sum(component_weights)
                    )
                row_components.append(component)

            return pn.Row(*row_components, sizing_mode="stretch_width")


class ResponsiveComponents:
    """Factory for creating responsive dashboard components."""

    def __init__(self, viewport_manager: ViewportManager):
        self.viewport_manager = viewport_manager

    def create_nav_menu(
        self, items: List[Dict[str, Any]]
    ) -> Union[pn.Tabs, pn.Accordion]:
        """Create responsive navigation menu."""
        if self.viewport_manager.is_mobile():
            # Use accordion for mobile
            accordion = pn.Accordion(sizing_mode="stretch_width")
            for item in items:
                accordion.append((item["title"], item["content"]))
            return accordion
        else:
            # Use tabs for desktop
            tabs = pn.Tabs(sizing_mode="stretch_width", tabs_location="above")
            for item in items:
                tabs.append((item["title"], item["content"]))
            return tabs

    def create_control_panel(self, controls: List[Any]) -> Union[pn.Column, pn.Row]:
        """Create responsive control panel."""
        if self.viewport_manager.is_mobile():
            # Stack controls vertically on mobile
            return pn.Column(*controls, sizing_mode="stretch_width", margin=(10, 5))
        else:
            # Arrange horizontally on desktop
            return pn.Row(*controls, sizing_mode="stretch_width", margin=(10, 5))

    def create_info_cards(self, cards: List[Dict[str, Any]]) -> pn.GridSpec:
        """Create responsive info cards layout."""
        grid_cols = {"xs": 1, "sm": 1, "md": 2, "lg": 3, "xl": 4, "xxl": 4}

        breakpoint = self.viewport_manager.get_current_breakpoint()
        cols = grid_cols[breakpoint]
        rows = (len(cards) + cols - 1) // cols

        grid = pn.GridSpec(
            sizing_mode="stretch_width", ncols=cols, nrows=rows, margin=5
        )

        for i, card in enumerate(cards):
            row = i // cols
            col = i % cols

            # Create card component
            card_component = pn.Card(
                pn.pane.HTML(f"<h3>{card['title']}</h3>"),
                pn.pane.HTML(f"<p>{card['content']}</p>"),
                title=card.get("subtitle", ""),
                sizing_mode="stretch_width",
                margin=5,
            )

            grid[row, col] = card_component

        return grid


class ResponsivePlots:
    """Utilities for creating responsive visualizations."""

    def __init__(self, viewport_manager: ViewportManager):
        self.viewport_manager = viewport_manager

    def get_plot_dimensions(self) -> Tuple[int, int]:
        """Get appropriate plot dimensions for current viewport."""
        breakpoint = self.viewport_manager.get_current_breakpoint()

        dimensions = {
            "xs": (300, 200),  # Mobile portrait
            "sm": (400, 250),  # Mobile landscape
            "md": (500, 350),  # Tablet
            "lg": (600, 400),  # Desktop
            "xl": (700, 450),  # Large desktop
            "xxl": (800, 500),  # Extra large
        }

        return dimensions.get(breakpoint, (600, 400))

    def create_responsive_figure(self, **kwargs) -> Any:
        """Create a responsive Bokeh figure."""
        width, height = self.get_plot_dimensions()

        # Override with responsive dimensions
        kwargs.update(
            {
                "width": width,
                "height": height,
                "sizing_mode": "scale_width"
                if self.viewport_manager.is_mobile()
                else "fixed",
            }
        )

        from bokeh.plotting import figure

        return figure(**kwargs)

    def adapt_plot_for_mobile(self, plot: Any) -> Any:
        """Adapt existing plot for mobile viewing."""
        if self.viewport_manager.is_mobile():
            # Adjust plot properties for mobile
            plot.title.text_font_size = "12pt"
            plot.axis.axis_label_text_font_size = "10pt"
            plot.axis.major_label_text_font_size = "8pt"
            plot.legend.label_text_font_size = "8pt"

            # Reduce toolbar size
            plot.toolbar.autohide = True

        return plot


class ThemeManager:
    """Manages theme switching and responsive styling."""

    def __init__(self, viewport_manager: ViewportManager):
        self.viewport_manager = viewport_manager
        self.current_theme = "light"

    def get_responsive_css(self) -> str:
        """Generate responsive CSS for current viewport."""
        breakpoint = self.viewport_manager.get_current_breakpoint()

        base_css = """
        .darwin-dashboard {
            font-family: 'Roboto', sans-serif;
            transition: all 0.3s ease;
        }

        .darwin-card {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 16px;
            margin: 8px;
            background: white;
        }

        .darwin-mobile .darwin-card {
            margin: 4px;
            padding: 12px;
        }

        .darwin-control-panel {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
        }

        .darwin-mobile .darwin-control-panel {
            flex-direction: column;
            gap: 8px;
        }
        """

        # Add breakpoint-specific styles
        if breakpoint in ["xs", "sm"]:
            base_css += """
            .darwin-plot {
                max-width: 100%;
                height: auto;
            }

            .darwin-table {
                font-size: 12px;
            }

            .darwin-button {
                width: 100%;
                min-height: 44px;
            }
            """

        return base_css

    def apply_theme(self, theme: str = "light") -> str:
        """Apply theme-specific styling."""
        self.current_theme = theme

        themes = {
            "light": {
                "background": "#ffffff",
                "surface": "#f5f5f5",
                "primary": "#1f77b4",
                "text": "#333333",
                "border": "#e0e0e0",
            },
            "dark": {
                "background": "#1e1e1e",
                "surface": "#2d2d2d",
                "primary": "#4fc3f7",
                "text": "#ffffff",
                "border": "#404040",
            },
            "high_contrast": {
                "background": "#000000",
                "surface": "#1a1a1a",
                "primary": "#ffff00",
                "text": "#ffffff",
                "border": "#ffffff",
            },
        }

        colors = themes.get(theme, themes["light"])

        theme_css = f"""
        .darwin-dashboard.theme-{theme} {{
            background-color: {colors['background']};
            color: {colors['text']};
        }}

        .darwin-dashboard.theme-{theme} .darwin-card {{
            background-color: {colors['surface']};
            border: 1px solid {colors['border']};
        }}

        .darwin-dashboard.theme-{theme} .darwin-primary {{
            background-color: {colors['primary']};
        }}
        """

        return theme_css


class AccessibilityManager:
    """Manages accessibility features for the dashboard."""

    def __init__(self, viewport_manager: ViewportManager):
        self.viewport_manager = viewport_manager

    def enhance_accessibility(self, component: Any) -> Any:
        """Add accessibility enhancements to components."""
        # Add ARIA labels and roles
        if hasattr(component, "css_classes"):
            component.css_classes = component.css_classes or []
            component.css_classes.append("accessible-component")

        # Add keyboard navigation support
        if hasattr(component, "styles"):
            component.styles = component.styles or {}
            component.styles.update(
                {"outline": "2px solid transparent", "outline-offset": "2px"}
            )

        return component

    def create_accessible_table(self, data: Dict[str, List]) -> pn.widgets.Tabulator:
        """Create an accessible data table."""
        table = pn.widgets.Tabulator(
            data,
            pagination="remote",
            page_size=10 if self.viewport_manager.is_mobile() else 25,
            sizing_mode="stretch_width",
            theme="site" if self.viewport_manager.is_mobile() else "standard",
        )

        # Add accessibility attributes
        table.configuration = {
            "keybindings": True,
            "accessibility": True,
            "tabindex": 0,
        }

        return table

    def create_accessible_button(
        self, name: str, description: str = None, **kwargs
    ) -> pn.widgets.Button:
        """Create an accessible button with proper ARIA attributes."""
        button = pn.widgets.Button(
            name=name,
            sizing_mode="stretch_width"
            if self.viewport_manager.is_mobile()
            else "fixed",
            **kwargs,
        )

        if description:
            button.description = description

        return button


def create_responsive_dashboard_manager(api_base_url: str = "http://localhost:8000"):
    """Factory function to create a complete responsive dashboard manager."""
    viewport_manager = ViewportManager()
    grid = ResponsiveGrid(viewport_manager)
    components = ResponsiveComponents(viewport_manager)
    plots = ResponsivePlots(viewport_manager)
    theme_manager = ThemeManager(viewport_manager)
    accessibility = AccessibilityManager(viewport_manager)

    return {
        "viewport": viewport_manager,
        "grid": grid,
        "components": components,
        "plots": plots,
        "theme": theme_manager,
        "accessibility": accessibility,
    }
