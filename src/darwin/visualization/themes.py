"""
Darwin Visualization Themes Module

This module provides theme management for Darwin visualizations, including
predefined color schemes, styling configurations, and accessibility features.

Features:
- Multiple built-in themes (light, dark, minimal, high_contrast)
- Custom color palette management
- Accessibility-compliant color schemes
- Dynamic theme switching capabilities
- Export-friendly styling options
- Responsive design configurations
- Brand-consistent styling
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from bokeh.models import Plot
from bokeh.palettes import Category10, Dark2, Set3, Viridis256

logger = logging.getLogger(__name__)


class ThemeManager:
    """Manages visual themes and styling for Darwin visualizations."""

    def __init__(self):
        """Initialize the theme manager with default configurations."""
        self.current_theme = "light"
        self.custom_themes = {}
        
        # Define built-in themes
        self.builtin_themes = {
            "light": self._create_light_theme(),
            "dark": self._create_dark_theme(),
            "minimal": self._create_minimal_theme(),
            "high_contrast": self._create_high_contrast_theme(),
            "darwin_brand": self._create_darwin_brand_theme()
        }
        
        # Color palettes for different data types
        self.color_palettes = {
            "categorical": {
                "default": Category10[10],
                "colorblind_safe": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", 
                                   "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
                                   "#bcbd22", "#17becf"],
                "pastel": ["#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", 
                          "#C5B0D5", "#C49C94", "#F7B6D3", "#C7C7C7", 
                          "#DBDB8D", "#9EDAE5"],
                "vibrant": ["#e60049", "#0bb4ff", "#50e991", "#e6d800", 
                           "#9b19f5", "#ffa300", "#dc0ab4", "#b3d4ff", 
                           "#00bfa0", "#fd7c6e"]
            },
            "sequential": {
                "viridis": Viridis256,
                "blue": ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", 
                        "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"],
                "red": ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", 
                       "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"],
                "green": ["#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", 
                         "#74c476", "#41ab5d", "#238b45", "#006d2c", "#00441b"]
            },
            "diverging": {
                "red_blue": ["#67001f", "#b2182b", "#d6604d", "#f4a582", 
                            "#fddbc7", "#d1e5f0", "#92c5de", "#4393c3", 
                            "#2166ac", "#053061"],
                "purple_green": ["#40004b", "#762a83", "#9970ab", "#c2a5cf", 
                                "#e7d4e8", "#d9f0d3", "#a6dba0", "#5aae61", 
                                "#1b7837", "#00441b"]
            }
        }

    def _create_light_theme(self) -> Dict[str, Any]:
        """Create light theme configuration."""
        return {
            "name": "light",
            "background_fill_color": "#ffffff",
            "border_fill_color": "#ffffff",
            "outline_line_color": "#cccccc",
            "outline_line_alpha": 0.8,
            "grid_line_color": "#e6e6e6",
            "grid_line_alpha": 0.8,
            "axis_line_color": "#cccccc",
            "axis_line_alpha": 1.0,
            "major_tick_line_color": "#cccccc",
            "minor_tick_line_color": "#e6e6e6",
            "major_label_text_color": "#333333",
            "title_text_color": "#333333",
            "legend_background_fill_color": "#ffffff",
            "legend_background_fill_alpha": 0.9,
            "legend_border_line_color": "#cccccc",
            "legend_label_text_color": "#333333",
            "toolbar_background": "#f5f5f5",
            "primary_color": "#1f77b4",
            "secondary_color": "#ff7f0e",
            "success_color": "#2ca02c",
            "warning_color": "#ff7f0e",
            "error_color": "#d62728",
            "text_color": "#333333",
            "muted_color": "#7f7f7f"
        }

    def _create_dark_theme(self) -> Dict[str, Any]:
        """Create dark theme configuration."""
        return {
            "name": "dark",
            "background_fill_color": "#2F2F2F",
            "border_fill_color": "#2F2F2F",
            "outline_line_color": "#666666",
            "outline_line_alpha": 0.8,
            "grid_line_color": "#555555",
            "grid_line_alpha": 0.8,
            "axis_line_color": "#777777",
            "axis_line_alpha": 1.0,
            "major_tick_line_color": "#777777",
            "minor_tick_line_color": "#555555",
            "major_label_text_color": "#ffffff",
            "title_text_color": "#ffffff",
            "legend_background_fill_color": "#2F2F2F",
            "legend_background_fill_alpha": 0.9,
            "legend_border_line_color": "#666666",
            "legend_label_text_color": "#ffffff",
            "toolbar_background": "#404040",
            "primary_color": "#5dade2",
            "secondary_color": "#f39c12",
            "success_color": "#58d68d",
            "warning_color": "#f39c12",
            "error_color": "#ec7063",
            "text_color": "#ffffff",
            "muted_color": "#aaaaaa"
        }

    def _create_minimal_theme(self) -> Dict[str, Any]:
        """Create minimal theme configuration."""
        return {
            "name": "minimal",
            "background_fill_color": "#fafafa",
            "border_fill_color": "#fafafa",
            "outline_line_color": "#e0e0e0",
            "outline_line_alpha": 0.5,
            "grid_line_color": "#f0f0f0",
            "grid_line_alpha": 0.5,
            "axis_line_color": "#dddddd",
            "axis_line_alpha": 0.8,
            "major_tick_line_color": "#dddddd",
            "minor_tick_line_color": "#f0f0f0",
            "major_label_text_color": "#444444",
            "title_text_color": "#444444",
            "legend_background_fill_color": "#fafafa",
            "legend_background_fill_alpha": 0.8,
            "legend_border_line_color": "#e0e0e0",
            "legend_label_text_color": "#444444",
            "toolbar_background": "#f8f8f8",
            "primary_color": "#3498db",
            "secondary_color": "#e67e22",
            "success_color": "#27ae60",
            "warning_color": "#f39c12",
            "error_color": "#e74c3c",
            "text_color": "#444444",
            "muted_color": "#999999"
        }

    def _create_high_contrast_theme(self) -> Dict[str, Any]:
        """Create high contrast theme for accessibility."""
        return {
            "name": "high_contrast",
            "background_fill_color": "#ffffff",
            "border_fill_color": "#ffffff",
            "outline_line_color": "#000000",
            "outline_line_alpha": 1.0,
            "grid_line_color": "#000000",
            "grid_line_alpha": 0.3,
            "axis_line_color": "#000000",
            "axis_line_alpha": 1.0,
            "major_tick_line_color": "#000000",
            "minor_tick_line_color": "#000000",
            "major_label_text_color": "#000000",
            "title_text_color": "#000000",
            "legend_background_fill_color": "#ffffff",
            "legend_background_fill_alpha": 1.0,
            "legend_border_line_color": "#000000",
            "legend_label_text_color": "#000000",
            "toolbar_background": "#f0f0f0",
            "primary_color": "#0000ff",
            "secondary_color": "#ff0000",
            "success_color": "#008000",
            "warning_color": "#ff8c00",
            "error_color": "#dc143c",
            "text_color": "#000000",
            "muted_color": "#666666"
        }

    def _create_darwin_brand_theme(self) -> Dict[str, Any]:
        """Create Darwin brand-specific theme."""
        return {
            "name": "darwin_brand",
            "background_fill_color": "#010B13",  # Rich Black
            "border_fill_color": "#010B13",
            "outline_line_color": "#9D00FF",  # Neon Purple
            "outline_line_alpha": 0.8,
            "grid_line_color": "#1A1A1A",  # Midnight Black
            "grid_line_alpha": 0.6,
            "axis_line_color": "#A3A3A3",  # Stone Grey
            "axis_line_alpha": 1.0,
            "major_tick_line_color": "#A3A3A3",
            "minor_tick_line_color": "#606770",
            "major_label_text_color": "#E3E3E3",  # Soft White
            "title_text_color": "#E3E3E3",
            "legend_background_fill_color": "#0F1111",  # Charcoal Black
            "legend_background_fill_alpha": 0.9,
            "legend_border_line_color": "#9D00FF",
            "legend_label_text_color": "#E3E3E3",
            "toolbar_background": "#1A1A1A",
            "primary_color": "#1B03A3",  # Neon Blue
            "secondary_color": "#FF10F0",  # Neon Pink
            "success_color": "#39FF14",  # Neon Green
            "warning_color": "#E9FF32",  # Neon Yellow
            "error_color": "#FF3131",  # Neon Red
            "text_color": "#E3E3E3",
            "muted_color": "#A3A3A3"
        }

    def get_theme(self, theme_name: str) -> Dict[str, Any]:
        """
        Get theme configuration by name.
        
        Args:
            theme_name: Name of the theme to retrieve
            
        Returns:
            Theme configuration dictionary
        """
        if theme_name in self.builtin_themes:
            return self.builtin_themes[theme_name].copy()
        elif theme_name in self.custom_themes:
            return self.custom_themes[theme_name].copy()
        else:
            logger.warning(f"Theme '{theme_name}' not found, returning light theme")
            return self.builtin_themes["light"].copy()

    def set_theme(self, theme_name: str):
        """
        Set the current active theme.
        
        Args:
            theme_name: Name of the theme to activate
        """
        if theme_name in self.builtin_themes or theme_name in self.custom_themes:
            self.current_theme = theme_name
            logger.info(f"Theme set to: {theme_name}")
        else:
            logger.error(f"Theme '{theme_name}' not found")

    def register_custom_theme(self, name: str, theme_config: Dict[str, Any]):
        """
        Register a custom theme configuration.
        
        Args:
            name: Name for the custom theme
            theme_config: Theme configuration dictionary
        """
        # Validate theme config has required keys
        required_keys = ["background_fill_color", "text_color", "primary_color"]
        if not all(key in theme_config for key in required_keys):
            raise ValueError(f"Theme config must include: {required_keys}")
        
        self.custom_themes[name] = theme_config
        logger.info(f"Custom theme '{name}' registered")

    def apply_theme_to_plot(self, plot: Plot, theme_name: str = None):
        """
        Apply theme styling to a Bokeh plot.
        
        Args:
            plot: Bokeh plot object to style
            theme_name: Theme to apply (uses current theme if None)
        """
        theme_name = theme_name or self.current_theme
        theme = self.get_theme(theme_name)
        
        try:
            # Apply background styling
            plot.background_fill_color = theme["background_fill_color"]
            plot.border_fill_color = theme["border_fill_color"]
            plot.outline_line_color = theme["outline_line_color"]
            plot.outline_line_alpha = theme["outline_line_alpha"]
            
            # Apply grid styling
            plot.grid.grid_line_color = theme["grid_line_color"]
            plot.grid.grid_line_alpha = theme["grid_line_alpha"]
            
            # Apply axis styling
            plot.axis.axis_line_color = theme["axis_line_color"]
            plot.axis.axis_line_alpha = theme["axis_line_alpha"]
            plot.axis.major_tick_line_color = theme["major_tick_line_color"]
            plot.axis.minor_tick_line_color = theme["minor_tick_line_color"]
            plot.axis.major_label_text_color = theme["major_label_text_color"]
            
            # Apply title styling
            plot.title.text_color = theme["title_text_color"]
            
            # Apply legend styling if legend exists
            if plot.legend:
                plot.legend.background_fill_color = theme["legend_background_fill_color"]
                plot.legend.background_fill_alpha = theme["legend_background_fill_alpha"]
                plot.legend.border_line_color = theme["legend_border_line_color"]
                plot.legend.label_text_color = theme["legend_label_text_color"]
            
            logger.debug(f"Applied theme '{theme_name}' to plot")
            
        except Exception as e:
            logger.error(f"Failed to apply theme to plot: {e}")

    def get_color_palette(
        self, 
        palette_type: str, 
        palette_name: str = "default", 
        n_colors: int = None
    ) -> List[str]:
        """
        Get a color palette for data visualization.
        
        Args:
            palette_type: Type of palette ('categorical', 'sequential', 'diverging')
            palette_name: Name of specific palette
            n_colors: Number of colors needed (for categorical palettes)
            
        Returns:
            List of color hex codes
        """
        try:
            if palette_type not in self.color_palettes:
                raise ValueError(f"Unknown palette type: {palette_type}")
            
            palettes = self.color_palettes[palette_type]
            
            if palette_name not in palettes:
                palette_name = list(palettes.keys())[0]  # Use first available
                logger.warning(f"Palette '{palette_name}' not found, using default")
            
            palette = palettes[palette_name]
            
            if palette_type == "categorical" and n_colors:
                # Cycle through colors if more needed
                while len(palette) < n_colors:
                    palette.extend(palette)
                return palette[:n_colors]
            
            return palette
            
        except Exception as e:
            logger.error(f"Failed to get color palette: {e}")
            return ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Fallback colors

    def create_responsive_config(self, screen_size: str = "desktop") -> Dict[str, Any]:
        """
        Create responsive configuration for different screen sizes.
        
        Args:
            screen_size: Target screen size ('mobile', 'tablet', 'desktop')
            
        Returns:
            Responsive configuration dictionary
        """
        configs = {
            "mobile": {
                "plot_width": 350,
                "plot_height": 250,
                "title_text_font_size": "12pt",
                "axis_label_text_font_size": "10pt",
                "major_label_text_font_size": "8pt",
                "legend_label_text_font_size": "8pt",
                "toolbar_location": "below"
            },
            "tablet": {
                "plot_width": 600,
                "plot_height": 400,
                "title_text_font_size": "14pt",
                "axis_label_text_font_size": "12pt",
                "major_label_text_font_size": "10pt",
                "legend_label_text_font_size": "10pt",
                "toolbar_location": "above"
            },
            "desktop": {
                "plot_width": 800,
                "plot_height": 600,
                "title_text_font_size": "16pt",
                "axis_label_text_font_size": "14pt",
                "major_label_text_font_size": "12pt",
                "legend_label_text_font_size": "12pt",
                "toolbar_location": "above"
            }
        }
        
        return configs.get(screen_size, configs["desktop"])

    def get_accessibility_colors(self) -> Dict[str, str]:
        """
        Get accessibility-compliant color scheme.
        
        Returns:
            Dictionary of semantic color mappings
        """
        return {
            "primary": "#0066cc",      # WCAG AA compliant blue
            "secondary": "#6c757d",    # Neutral gray
            "success": "#28a745",      # Green with good contrast
            "warning": "#ffc107",      # Yellow with dark text
            "danger": "#dc3545",       # Red with good contrast
            "info": "#17a2b8",         # Teal/cyan
            "light": "#f8f9fa",        # Light background
            "dark": "#343a40",         # Dark text/background
            "text_primary": "#212529", # Primary text color
            "text_secondary": "#6c757d", # Secondary text color
            "text_muted": "#868e96",   # Muted text color
            "border": "#dee2e6",       # Border color
            "background": "#ffffff"    # Background color
        }

    def validate_color_accessibility(
        self, 
        foreground: str, 
        background: str, 
        level: str = "AA"
    ) -> bool:
        """
        Validate color contrast for accessibility compliance.
        
        Args:
            foreground: Foreground color (hex)
            background: Background color (hex)
            level: WCAG compliance level ('AA' or 'AAA')
            
        Returns:
            True if colors meet accessibility standards
        """
        def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def luminance(rgb: Tuple[int, int, int]) -> float:
            r, g, b = [x/255.0 for x in rgb]
            r = r/12.92 if r <= 0.03928 else ((r+0.055)/1.055)**2.4
            g = g/12.92 if g <= 0.03928 else ((g+0.055)/1.055)**2.4
            b = b/12.92 if b <= 0.03928 else ((b+0.055)/1.055)**2.4
            return 0.2126*r + 0.7152*g + 0.0722*b
        
        try:
            fg_rgb = hex_to_rgb(foreground)
            bg_rgb = hex_to_rgb(background)
            
            fg_lum = luminance(fg_rgb)
            bg_lum = luminance(bg_rgb)
            
            # Calculate contrast ratio
            lighter = max(fg_lum, bg_lum)
            darker = min(fg_lum, bg_lum)
            contrast_ratio = (lighter + 0.05) / (darker + 0.05)
            
            # Check compliance
            if level == "AAA":
                return contrast_ratio >= 7.0
            else:  # AA
                return contrast_ratio >= 4.5
                
        except Exception as e:
            logger.error(f"Failed to validate color accessibility: {e}")
            return False

    def export_theme_css(self, theme_name: str = None) -> str:
        """
        Export theme as CSS for web integration.
        
        Args:
            theme_name: Theme to export (uses current theme if None)
            
        Returns:
            CSS string with theme variables
        """
        theme_name = theme_name or self.current_theme
        theme = self.get_theme(theme_name)
        
        css_vars = []
        css_vars.append(f":root {{")
        
        for key, value in theme.items():
            if isinstance(value, str) and (value.startswith('#') or value.startswith('rgb')):
                css_key = f"--darwin-{key.replace('_', '-')}"
                css_vars.append(f"  {css_key}: {value};")
        
        css_vars.append("}")
        
        return "\n".join(css_vars)

    def get_available_themes(self) -> List[str]:
        """
        Get list of all available theme names.
        
        Returns:
            List of theme names
        """
        return list(self.builtin_themes.keys()) + list(self.custom_themes.keys())

    def get_theme_preview(self, theme_name: str) -> Dict[str, str]:
        """
        Get a preview of theme colors for UI display.
        
        Args:
            theme_name: Name of theme to preview
            
        Returns:
            Dictionary with preview colors
        """
        theme = self.get_theme(theme_name)
        
        return {
            "background": theme.get("background_fill_color", "#ffffff"),
            "primary": theme.get("primary_color", "#1f77b4"),
            "secondary": theme.get("secondary_color", "#ff7f0e"),
            "text": theme.get("text_color", "#333333"),
            "success": theme.get("success_color", "#2ca02c"),
            "warning": theme.get("warning_color", "#ff7f0e"),
            "error": theme.get("error_color", "#d62728")
        }