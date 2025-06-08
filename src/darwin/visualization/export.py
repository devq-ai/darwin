"""
Darwin Visualization Export Manager

This module provides comprehensive export capabilities for Darwin visualizations,
including static images, interactive HTML files, data exports, and report generation.

Features:
- Export plots to various image formats (PNG, SVG, PDF)
- Generate interactive HTML exports with embedded data
- Export raw data to CSV, JSON, and Excel formats
- Create comprehensive PDF reports with multiple visualizations
- Batch export capabilities for multiple plots
- Custom styling and branding options
- Compression and optimization for large exports
- Metadata preservation and documentation
"""

import io
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from bokeh.io import export_png, export_svgs, output_file, save
from bokeh.models import Plot
from bokeh.plotting import figure

logger = logging.getLogger(__name__)


class ExportManager:
    """Manages export operations for visualizations and data."""

    def __init__(self, output_directory: str = "exports"):
        """
        Initialize the export manager.

        Args:
            output_directory: Base directory for exports
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Supported formats
        self.image_formats = ["png", "svg", "pdf"]
        self.data_formats = ["csv", "json", "excel", "parquet"]
        self.report_formats = ["html", "pdf"]
        
        # Export settings
        self.default_image_settings = {
            "width": 800,
            "height": 600,
            "dpi": 300,
            "transparent": False
        }

    def export_plot(
        self,
        plot: Union[figure, Plot],
        filename: str,
        format: str = "png",
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export a single plot to file.

        Args:
            plot: Bokeh plot object to export
            filename: Output filename (without extension)
            format: Export format ('png', 'svg', 'html')
            settings: Export-specific settings

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        if format not in self.image_formats + ["html"]:
            raise ValueError(f"Unsupported format: {format}")

        export_settings = self.default_image_settings.copy()
        if settings:
            export_settings.update(settings)

        output_path = self.output_directory / f"{filename}.{format}"

        try:
            if format == "png":
                export_png(plot, filename=str(output_path), **export_settings)
            elif format == "svg":
                plot.output_backend = "svg"
                export_svgs(plot, filename=str(output_path))
            elif format == "html":
                output_file(str(output_path))
                save(plot)

            logger.info(f"Plot exported to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            raise

    def export_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        filename: str,
        format: str = "csv",
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Export data to file.

        Args:
            data: Data to export
            filename: Output filename (without extension)
            format: Export format ('csv', 'json', 'excel', 'parquet')
            settings: Format-specific settings

        Returns:
            Path to exported file

        Raises:
            ValueError: If format is not supported
        """
        if format not in self.data_formats:
            raise ValueError(f"Unsupported data format: {format}")

        export_settings = settings or {}
        output_path = self.output_directory / f"{filename}.{format}"

        try:
            # Convert to DataFrame if needed
            if isinstance(data, (dict, list)):
                df = pd.DataFrame(data)
            else:
                df = data

            if format == "csv":
                df.to_csv(output_path, index=False, **export_settings)
            elif format == "json":
                df.to_json(output_path, orient="records", **export_settings)
            elif format == "excel":
                df.to_excel(output_path, index=False, **export_settings)
            elif format == "parquet":
                df.to_parquet(output_path, **export_settings)

            logger.info(f"Data exported to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise

    def export_multiple_plots(
        self,
        plots: Dict[str, Union[figure, Plot]],
        format: str = "png",
        settings: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Export multiple plots in batch.

        Args:
            plots: Dictionary of plot names to plot objects
            format: Export format
            settings: Export settings

        Returns:
            List of paths to exported files
        """
        exported_files = []

        for name, plot in plots.items():
            try:
                file_path = self.export_plot(plot, name, format, settings)
                exported_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to export plot '{name}': {e}")

        return exported_files

    def create_export_package(
        self,
        exports: Dict[str, Any],
        package_name: str = None
    ) -> str:
        """
        Create a ZIP package containing multiple exports.

        Args:
            exports: Dictionary of export types and data
            package_name: Name for the package file

        Returns:
            Path to created package

        Example:
            exports = {
                'plots': {'convergence': plot1, 'diversity': plot2},
                'data': {'results': dataframe},
                'settings': {'config': config_dict}
            }
        """
        if package_name is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            package_name = f"darwin_export_{timestamp}"

        package_path = self.output_directory / f"{package_name}.zip"

        try:
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Export plots
                if 'plots' in exports:
                    plots_exported = self.export_multiple_plots(
                        exports['plots'], format="png"
                    )
                    for file_path in plots_exported:
                        zipf.write(file_path, f"plots/{Path(file_path).name}")

                # Export data
                if 'data' in exports:
                    for name, data in exports['data'].items():
                        data_path = self.export_data(data, name, format="csv")
                        zipf.write(data_path, f"data/{Path(data_path).name}")

                # Export settings/metadata
                if 'settings' in exports:
                    settings_path = self.export_data(
                        exports['settings'], "metadata", format="json"
                    )
                    zipf.write(settings_path, f"metadata/{Path(settings_path).name}")

                # Add README
                readme_content = self._generate_package_readme(exports)
                zipf.writestr("README.txt", readme_content)

            logger.info(f"Export package created: {package_path}")
            return str(package_path)

        except Exception as e:
            logger.error(f"Failed to create export package: {e}")
            raise

    def create_html_report(
        self,
        plots: Dict[str, Union[figure, Plot]],
        data: Optional[Dict[str, pd.DataFrame]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template: str = "default"
    ) -> str:
        """
        Create an interactive HTML report with embedded plots.

        Args:
            plots: Dictionary of plot names to plot objects
            data: Optional data tables to include
            metadata: Optional metadata to include
            template: Report template to use

        Returns:
            Path to generated HTML report
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = self.output_directory / f"darwin_report_{timestamp}.html"

        try:
            html_content = self._generate_html_report(plots, data, metadata, template)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"HTML report created: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to create HTML report: {e}")
            raise

    def _generate_html_report(
        self,
        plots: Dict[str, Union[figure, Plot]],
        data: Optional[Dict[str, pd.DataFrame]],
        metadata: Optional[Dict[str, Any]],
        template: str
    ) -> str:
        """Generate HTML content for the report."""
        from bokeh.embed import components
        from bokeh.resources import CDN

        # Get plot components
        script, div_dict = components(plots)

        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <title>Darwin Optimization Report</title>
            {CDN.render_css()}
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .plot-section {{ margin: 20px 0; }}
                .plot-title {{ font-size: 18px; font-weight: bold; margin: 10px 0; }}
                .data-table {{ margin: 20px 0; }}
                .metadata {{ background: #f5f5f5; padding: 15px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Darwin Genetic Algorithm Optimization Report</h1>
                <p>Generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        """

        # Add metadata section
        if metadata:
            html += '<div class="metadata"><h2>Optimization Metadata</h2>'
            for key, value in metadata.items():
                html += f'<p><strong>{key}:</strong> {value}</p>'
            html += '</div>'

        # Add plots
        for plot_name, div in div_dict.items():
            html += f"""
            <div class="plot-section">
                <div class="plot-title">{plot_name.replace('_', ' ').title()}</div>
                {div}
            </div>
            """

        # Add data tables
        if data:
            html += '<h2>Data Tables</h2>'
            for table_name, df in data.items():
                html += f"""
                <div class="data-table">
                    <h3>{table_name.replace('_', ' ').title()}</h3>
                    {df.head(10).to_html(classes='table table-striped')}
                </div>
                """

        html += f"""
            {CDN.render_js()}
            {script}
        </body>
        </html>
        """

        return html

    def _generate_package_readme(self, exports: Dict[str, Any]) -> str:
        """Generate README content for export packages."""
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        
        readme = f"""
Darwin Genetic Algorithm Optimization Export Package
Generated on: {timestamp}

Contents:
"""
        
        if 'plots' in exports:
            readme += f"\nPlots ({len(exports['plots'])} files):\n"
            for plot_name in exports['plots'].keys():
                readme += f"  - {plot_name}.png\n"
        
        if 'data' in exports:
            readme += f"\nData ({len(exports['data'])} files):\n"
            for data_name in exports['data'].keys():
                readme += f"  - {data_name}.csv\n"
        
        if 'settings' in exports:
            readme += "\nMetadata:\n"
            readme += "  - metadata.json\n"
        
        readme += """
Usage:
- Plots are in PNG format and can be viewed with any image viewer
- Data files are in CSV format and can be opened with Excel or any text editor
- Metadata is in JSON format and contains configuration and run information

For questions or support, visit: https://github.com/devqai/darwin
"""
        
        return readme

    def cleanup_old_exports(self, days_old: int = 7) -> int:
        """
        Clean up export files older than specified days.

        Args:
            days_old: Remove files older than this many days

        Returns:
            Number of files removed
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        removed_count = 0

        try:
            for file_path in self.output_directory.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    removed_count += 1

            logger.info(f"Cleaned up {removed_count} old export files")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup exports: {e}")
            return 0

    def get_export_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about exports in the output directory.

        Returns:
            Dictionary with export statistics
        """
        try:
            total_files = 0
            total_size = 0
            file_types = {}

            for file_path in self.output_directory.rglob("*"):
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1

            return {
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types,
                "output_directory": str(self.output_directory)
            }

        except Exception as e:
            logger.error(f"Failed to get export statistics: {e}")
            return {"error": str(e)}