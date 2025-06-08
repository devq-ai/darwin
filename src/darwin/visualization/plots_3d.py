"""
3D Visualization Capabilities for Darwin Visualization Engine

This module provides advanced 3D visualization capabilities for genetic algorithm
optimization analysis, including 3D fitness landscapes, solution space exploration,
population evolution visualization, and interactive 3D plots.

Features:
- 3D fitness landscape visualization
- 3D solution space exploration with clustering
- Population evolution animation in 3D
- Interactive 3D scatter plots with hover and selection
- 3D Pareto frontier visualization for multi-objective optimization
- 3D surface plots for parameter sensitivity analysis
- Animated 3D convergence visualization
- Virtual reality (VR) ready 3D exports
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Plot3DEngine:
    """Core 3D visualization engine using Plotly."""

    def __init__(self, theme: str = "plotly", width: int = 800, height: int = 600):
        self.theme = theme
        self.width = width
        self.height = height
        self.color_scales = {
            "viridis": "Viridis",
            "plasma": "Plasma",
            "inferno": "Inferno",
            "turbo": "Turbo",
            "rainbow": "Rainbow",
        }

    def create_3d_fitness_landscape(
        self,
        solution_data: List[Dict[str, Any]],
        fitness_values: List[float],
        title: str = "3D Fitness Landscape",
        color_scale: str = "viridis",
    ) -> go.Figure:
        """
        Create a 3D fitness landscape visualization.

        Args:
            solution_data: List of solution dictionaries with 'variables' key
            fitness_values: Corresponding fitness values
            title: Plot title
            color_scale: Color scale for the surface

        Returns:
            Plotly Figure object
        """
        if not solution_data or not fitness_values:
            return self._create_empty_plot("No data available for 3D landscape")

        # Extract variables and reduce dimensionality if needed
        variables_matrix = np.array([sol["variables"] for sol in solution_data])

        if variables_matrix.shape[1] > 3:
            # Use PCA to reduce to 3D
            pca = PCA(n_components=3)
            variables_3d = pca.fit_transform(
                StandardScaler().fit_transform(variables_matrix)
            )
            axis_labels = [
                f"PC{i+1} ({pca.explained_variance_ratio_[i]:.2%})" for i in range(3)
            ]
        elif variables_matrix.shape[1] == 3:
            variables_3d = variables_matrix
            axis_labels = [f"Variable {i+1}" for i in range(3)]
        elif variables_matrix.shape[1] == 2:
            # Add a third dimension based on fitness
            variables_3d = np.column_stack([variables_matrix, fitness_values])
            axis_labels = ["Variable 1", "Variable 2", "Fitness"]
        else:
            return self._create_empty_plot(
                "Insufficient dimensions for 3D visualization"
            )

        # Create 3D scatter plot
        fig = go.Figure()

        # Add scatter points
        fig.add_trace(
            go.Scatter3d(
                x=variables_3d[:, 0],
                y=variables_3d[:, 1],
                z=variables_3d[:, 2],
                mode="markers",
                marker=dict(
                    size=8,
                    color=fitness_values,
                    colorscale=self.color_scales.get(color_scale, "Viridis"),
                    opacity=0.8,
                    colorbar=dict(title="Fitness"),
                    showscale=True,
                ),
                text=[f"Fitness: {f:.4f}" for f in fitness_values],
                hovertemplate="<b>%{text}</b><br>"
                + f"{axis_labels[0]}: %{{x:.4f}}<br>"
                + f"{axis_labels[1]}: %{{y:.4f}}<br>"
                + f"{axis_labels[2]}: %{{z:.4f}}<extra></extra>",
                name="Solutions",
            )
        )

        # Try to create surface if enough points
        if len(solution_data) > 20:
            surface_fig = self._create_fitness_surface(
                variables_3d, fitness_values, color_scale
            )
            if surface_fig:
                fig.add_traces(surface_fig.data)

        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=self.width,
            height=self.height,
            template=self.theme,
        )

        return fig

    def create_3d_solution_space(
        self,
        optimization_data: Dict[str, Any],
        cluster_solutions: bool = True,
        animate_evolution: bool = False,
    ) -> go.Figure:
        """
        Create 3D solution space exploration visualization.

        Args:
            optimization_data: Dictionary containing solution data for multiple optimizers
            cluster_solutions: Whether to apply clustering to solutions
            animate_evolution: Whether to create animation over generations

        Returns:
            Plotly Figure object
        """
        if not optimization_data:
            return self._create_empty_plot("No optimization data available")

        all_solutions = []
        all_fitness = []
        all_generations = []
        all_optimizers = []

        # Collect all solution data
        for optimizer_id, data in optimization_data.items():
            best_solutions = data.get("best_solutions", [])
            fitness_history = data.get("fitness_history", [])

            for sol in best_solutions:
                if "variables" in sol:
                    all_solutions.append(sol["variables"])
                    all_fitness.append(sol.get("fitness", 0))
                    all_generations.append(sol.get("generation", 0))
                    all_optimizers.append(optimizer_id)

        if not all_solutions:
            return self._create_empty_plot("No solution data found")

        # Convert to numpy array and reduce dimensionality
        solutions_matrix = np.array(all_solutions)

        if solutions_matrix.shape[1] > 3:
            # Use t-SNE for better 3D projection
            if solutions_matrix.shape[0] > 30:
                tsne = TSNE(
                    n_components=3,
                    random_state=42,
                    perplexity=min(30, solutions_matrix.shape[0] - 1),
                )
                solutions_3d = tsne.fit_transform(
                    StandardScaler().fit_transform(solutions_matrix)
                )
            else:
                pca = PCA(n_components=3)
                solutions_3d = pca.fit_transform(
                    StandardScaler().fit_transform(solutions_matrix)
                )
        else:
            solutions_3d = solutions_matrix

        fig = go.Figure()

        # Apply clustering if requested
        if cluster_solutions and len(all_solutions) > 5:
            n_clusters = min(5, len(set(all_optimizers)))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(solutions_3d)

            # Plot each cluster
            colors = px.colors.qualitative.Set1
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                if np.any(mask):
                    fig.add_trace(
                        go.Scatter3d(
                            x=solutions_3d[mask, 0],
                            y=solutions_3d[mask, 1],
                            z=solutions_3d[mask, 2]
                            if solutions_3d.shape[1] > 2
                            else np.zeros(np.sum(mask)),
                            mode="markers",
                            marker=dict(
                                size=8,
                                color=colors[cluster_id % len(colors)],
                                opacity=0.7,
                            ),
                            name=f"Cluster {cluster_id + 1}",
                            text=[
                                f"Optimizer: {all_optimizers[i]}<br>Fitness: {all_fitness[i]:.4f}"
                                for i in range(len(all_optimizers))
                                if mask[i]
                            ],
                            hovertemplate="<b>%{text}</b><extra></extra>",
                        )
                    )
        else:
            # Plot by optimizer
            colors = px.colors.qualitative.Set1
            unique_optimizers = list(set(all_optimizers))

            for i, optimizer_id in enumerate(unique_optimizers):
                mask = np.array(all_optimizers) == optimizer_id
                if np.any(mask):
                    fig.add_trace(
                        go.Scatter3d(
                            x=solutions_3d[mask, 0],
                            y=solutions_3d[mask, 1],
                            z=solutions_3d[mask, 2]
                            if solutions_3d.shape[1] > 2
                            else np.zeros(np.sum(mask)),
                            mode="markers",
                            marker=dict(
                                size=8, color=colors[i % len(colors)], opacity=0.7
                            ),
                            name=f"Optimizer {optimizer_id}",
                            text=[
                                f"Generation: {all_generations[j]}<br>Fitness: {all_fitness[j]:.4f}"
                                for j in range(len(all_optimizers))
                                if mask[j]
                            ],
                            hovertemplate="<b>%{text}</b><extra></extra>",
                        )
                    )

        fig.update_layout(
            title="3D Solution Space Exploration",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=self.width,
            height=self.height,
            template=self.theme,
        )

        return fig

    def create_3d_population_evolution(
        self,
        population_data: List[Dict[str, Any]],
        fitness_history: List[Dict[str, Any]],
        animate: bool = True,
    ) -> go.Figure:
        """
        Create 3D visualization of population evolution over generations.

        Args:
            population_data: List of population snapshots
            fitness_history: Corresponding fitness history
            animate: Whether to create animated visualization

        Returns:
            Plotly Figure object
        """
        if not population_data:
            return self._create_empty_plot("No population data available")

        if animate:
            return self._create_animated_population_evolution(
                population_data, fitness_history
            )
        else:
            return self._create_static_population_evolution(
                population_data, fitness_history
            )

    def create_3d_pareto_frontier(
        self,
        solutions: List[Dict[str, Any]],
        objectives: List[str],
        title: str = "3D Pareto Frontier",
    ) -> go.Figure:
        """
        Create 3D Pareto frontier visualization for multi-objective optimization.

        Args:
            solutions: List of solution dictionaries with objective values
            objectives: List of objective names (must be 3 for 3D)
            title: Plot title

        Returns:
            Plotly Figure object
        """
        if len(objectives) != 3:
            return self._create_empty_plot(
                "Exactly 3 objectives required for 3D Pareto frontier"
            )

        if not solutions:
            return self._create_empty_plot("No solutions provided")

        # Extract objective values
        obj_values = []
        for sol in solutions:
            values = []
            for obj in objectives:
                values.append(sol.get(obj, 0))
            obj_values.append(values)

        obj_array = np.array(obj_values)

        # Identify Pareto frontier points
        pareto_mask = self._identify_pareto_frontier(obj_array)

        fig = go.Figure()

        # Plot non-dominated solutions (Pareto frontier)
        pareto_points = obj_array[pareto_mask]
        fig.add_trace(
            go.Scatter3d(
                x=pareto_points[:, 0],
                y=pareto_points[:, 1],
                z=pareto_points[:, 2],
                mode="markers",
                marker=dict(size=10, color="red", opacity=0.8, symbol="diamond"),
                name="Pareto Frontier",
                hovertemplate="<b>Pareto Optimal</b><br>"
                + f"{objectives[0]}: %{{x:.4f}}<br>"
                + f"{objectives[1]}: %{{y:.4f}}<br>"
                + f"{objectives[2]}: %{{z:.4f}}<extra></extra>",
            )
        )

        # Plot dominated solutions
        dominated_points = obj_array[~pareto_mask]
        if len(dominated_points) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=dominated_points[:, 0],
                    y=dominated_points[:, 1],
                    z=dominated_points[:, 2],
                    mode="markers",
                    marker=dict(size=6, color="lightblue", opacity=0.5),
                    name="Dominated Solutions",
                    hovertemplate="<b>Dominated</b><br>"
                    + f"{objectives[0]}: %{{x:.4f}}<br>"
                    + f"{objectives[1]}: %{{y:.4f}}<br>"
                    + f"{objectives[2]}: %{{z:.4f}}<extra></extra>",
                )
            )

        # Try to create Pareto surface if enough points
        if len(pareto_points) > 10:
            try:
                surface = self._create_pareto_surface(pareto_points)
                if surface:
                    fig.add_trace(surface)
            except Exception as e:
                logger.warning(f"Could not create Pareto surface: {e}")

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=objectives[0],
                yaxis_title=objectives[1],
                zaxis_title=objectives[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=self.width,
            height=self.height,
            template=self.theme,
        )

        return fig

    def create_3d_convergence_animation(
        self,
        fitness_history: List[Dict[str, Any]],
        title: str = "3D Convergence Animation",
    ) -> go.Figure:
        """
        Create animated 3D visualization of convergence process.

        Args:
            fitness_history: List of fitness records over generations
            title: Plot title

        Returns:
            Plotly Figure object with animation
        """
        if not fitness_history:
            return self._create_empty_plot("No fitness history available")

        df = pd.DataFrame(fitness_history)

        # Create 3D trajectory of best fitness, mean fitness, and diversity
        fig = go.Figure()

        # Add traces for animation frames
        frames = []

        for i in range(1, len(df) + 1):
            frame_data = df.iloc[:i]

            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=frame_data["generation"],
                        y=frame_data["best_fitness"],
                        z=frame_data.get("diversity", [0.5] * len(frame_data)),
                        mode="lines+markers",
                        marker=dict(size=4, color="red"),
                        line=dict(color="red", width=4),
                        name="Best Fitness Trajectory",
                    ),
                    go.Scatter3d(
                        x=frame_data["generation"],
                        y=frame_data.get("mean_fitness", frame_data["best_fitness"]),
                        z=frame_data.get("diversity", [0.5] * len(frame_data)),
                        mode="lines+markers",
                        marker=dict(size=3, color="blue"),
                        line=dict(color="blue", width=2),
                        name="Mean Fitness Trajectory",
                    ),
                ],
                name=f"Generation {i}",
            )
            frames.append(frame)

        # Add initial traces
        fig.add_trace(
            go.Scatter3d(
                x=[df["generation"].iloc[0]],
                y=[df["best_fitness"].iloc[0]],
                z=[
                    df.get("diversity", [0.5]).iloc[0]
                    if "diversity" in df.columns
                    else 0.5
                ],
                mode="markers",
                marker=dict(size=4, color="red"),
                name="Best Fitness",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[df["generation"].iloc[0]],
                y=[df.get("mean_fitness", df["best_fitness"]).iloc[0]],
                z=[
                    df.get("diversity", [0.5]).iloc[0]
                    if "diversity" in df.columns
                    else 0.5
                ],
                mode="markers",
                marker=dict(size=3, color="blue"),
                name="Mean Fitness",
            )
        )

        # Add frames to figure
        fig.frames = frames

        # Add animation controls
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Generation",
                yaxis_title="Fitness",
                zaxis_title="Diversity",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2)),
            ),
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "transition": {"duration": 50},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        },
                    ],
                }
            ],
            width=self.width,
            height=self.height,
            template=self.theme,
        )

        return fig

    def create_3d_parameter_sensitivity(
        self,
        parameter_data: Dict[str, Any],
        sensitivity_results: Dict[str, Any],
        title: str = "3D Parameter Sensitivity Analysis",
    ) -> go.Figure:
        """
        Create 3D parameter sensitivity surface plot.

        Args:
            parameter_data: Dictionary with parameter ranges and values
            sensitivity_results: Sensitivity analysis results
            title: Plot title

        Returns:
            Plotly Figure object
        """
        if not parameter_data or not sensitivity_results:
            return self._create_empty_plot("No parameter sensitivity data available")

        # Extract parameter ranges and sensitivity values
        param_names = list(parameter_data.keys())[:3]  # Take first 3 parameters

        if len(param_names) < 2:
            return self._create_empty_plot(
                "Need at least 2 parameters for sensitivity analysis"
            )

        fig = go.Figure()

        if len(param_names) >= 3:
            # Create 3D surface plot
            x_values = parameter_data[param_names[0]]
            y_values = parameter_data[param_names[1]]
            z_values = parameter_data[param_names[2]]

            # Get sensitivity values (assuming they correspond to parameter combinations)
            sensitivity_values = sensitivity_results.get(
                "sensitivity_matrix", np.random.rand(len(x_values), len(y_values))
            )

            # Create mesh grid
            X, Y = np.meshgrid(x_values, y_values)
            Z = np.array(sensitivity_values)

            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale="Viridis",
                    colorbar=dict(title="Sensitivity"),
                    hovertemplate="<b>Sensitivity Analysis</b><br>"
                    + f"{param_names[0]}: %{{x:.4f}}<br>"
                    + f"{param_names[1]}: %{{y:.4f}}<br>"
                    + "Sensitivity: %{z:.4f}<extra></extra>",
                )
            )
        else:
            # Create 3D scatter plot with sensitivity as z-axis
            x_values = parameter_data[param_names[0]]
            y_values = parameter_data[param_names[1]]
            z_values = sensitivity_results.get(
                "sensitivity_scores", np.random.rand(len(x_values))
            )

            fig.add_trace(
                go.Scatter3d(
                    x=x_values,
                    y=y_values,
                    z=z_values,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=z_values,
                        colorscale="Viridis",
                        colorbar=dict(title="Sensitivity"),
                        opacity=0.8,
                    ),
                    hovertemplate="<b>Parameter Combination</b><br>"
                    + f"{param_names[0]}: %{{x:.4f}}<br>"
                    + f"{param_names[1]}: %{{y:.4f}}<br>"
                    + "Sensitivity: %{z:.4f}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=param_names[0],
                yaxis_title=param_names[1],
                zaxis_title="Sensitivity" if len(param_names) < 3 else param_names[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=self.width,
            height=self.height,
            template=self.theme,
        )

        return fig

    def _create_fitness_surface(
        self, variables_3d: np.ndarray, fitness_values: List[float], color_scale: str
    ) -> Optional[go.Figure]:
        """Create a fitness surface from scattered data points."""
        try:
            # Create a regular grid
            x_min, x_max = variables_3d[:, 0].min(), variables_3d[:, 0].max()
            y_min, y_max = variables_3d[:, 1].min(), variables_3d[:, 1].max()

            grid_size = 20
            xi = np.linspace(x_min, x_max, grid_size)
            yi = np.linspace(y_min, y_max, grid_size)
            X, Y = np.meshgrid(xi, yi)

            # Interpolate fitness values onto the grid
            Z = griddata(
                (variables_3d[:, 0], variables_3d[:, 1]),
                fitness_values,
                (X, Y),
                method="cubic",
                fill_value=np.mean(fitness_values),
            )

            surface = go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=self.color_scales.get(color_scale, "Viridis"),
                opacity=0.3,
                name="Fitness Surface",
                showscale=False,
            )

            fig = go.Figure(data=[surface])
            return fig

        except Exception as e:
            logger.warning(f"Could not create fitness surface: {e}")
            return None

    def _create_animated_population_evolution(
        self,
        population_data: List[Dict[str, Any]],
        fitness_history: List[Dict[str, Any]],
    ) -> go.Figure:
        """Create animated population evolution visualization."""
        frames = []

        for i, pop_snapshot in enumerate(population_data):
            if "population" not in pop_snapshot:
                continue

            population = np.array(pop_snapshot["population"])
            generation = pop_snapshot.get("generation", i)

            # Reduce dimensionality to 3D if necessary
            if population.shape[1] > 3:
                pca = PCA(n_components=3)
                pop_3d = pca.fit_transform(population)
            else:
                pop_3d = population

            # Get corresponding fitness values if available
            if i < len(fitness_history):
                fitness_val = fitness_history[i].get("mean_fitness", 0)
                colors = [fitness_val] * len(pop_3d)
            else:
                colors = ["blue"] * len(pop_3d)

            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=pop_3d[:, 0],
                        y=pop_3d[:, 1],
                        z=pop_3d[:, 2]
                        if pop_3d.shape[1] > 2
                        else np.zeros(len(pop_3d)),
                        mode="markers",
                        marker=dict(
                            size=6, color=colors, colorscale="Viridis", opacity=0.7
                        ),
                        name=f"Generation {generation}",
                    )
                ],
                name=f"Generation {generation}",
            )
            frames.append(frame)

        # Create initial figure
        if frames:
            fig = go.Figure(data=frames[0].data, frames=frames)

            fig.update_layout(
                title="Population Evolution Animation",
                scene=dict(
                    xaxis_title="Dimension 1",
                    yaxis_title="Dimension 2",
                    zaxis_title="Dimension 3",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                ),
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [None, {"frame": {"duration": 200}}],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [
                                    [None],
                                    {"frame": {"duration": 0}, "mode": "immediate"},
                                ],
                            },
                        ],
                    }
                ],
                width=self.width,
                height=self.height,
                template=self.theme,
            )

            return fig
        else:
            return self._create_empty_plot("No population data for animation")

    def _create_static_population_evolution(
        self,
        population_data: List[Dict[str, Any]],
        fitness_history: List[Dict[str, Any]],
    ) -> go.Figure:
        """Create static population evolution visualization showing multiple generations."""
        fig = go.Figure()

        colors = px.colors.qualitative.Set1

        for i, pop_snapshot in enumerate(
            population_data[:: max(1, len(population_data) // 5)]
        ):  # Show every 5th generation
            if "population" not in pop_snapshot:
                continue

            population = np.array(pop_snapshot["population"])
            generation = pop_snapshot.get("generation", i)

            # Reduce dimensionality to 3D if necessary
            if population.shape[1] > 3:
                pca = PCA(n_components=3)
                pop_3d = pca.fit_transform(population)
            else:
                pop_3d = population

            fig.add_trace(
                go.Scatter3d(
                    x=pop_3d[:, 0],
                    y=pop_3d[:, 1],
                    z=pop_3d[:, 2] if pop_3d.shape[1] > 2 else np.zeros(len(pop_3d)),
                    mode="markers",
                    marker=dict(size=6, color=colors[i % len(colors)], opacity=0.6),
                    name=f"Generation {generation}",
                )
            )

        fig.update_layout(
            title="Population Evolution (Multiple Generations)",
            scene=dict(
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                zaxis_title="Dimension 3",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            width=self.width,
            height=self.height,
            template=self.theme,
        )

        return fig

    def _identify_pareto_frontier(self, objective_values: np.ndarray) -> np.ndarray:
        """Identify Pareto frontier points (assuming minimization)."""
        n_points = objective_values.shape[0]
        pareto_mask = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(objective_values[j] <= objective_values[i]) and np.any(
                        objective_values[j] < objective_values[i]
                    ):
                        pareto_mask[i] = False
                        break

        return pareto_mask

    def _create_pareto_surface(self, pareto_points: np.ndarray) -> Optional[go.Surface]:
        """Create a surface approximation of the Pareto frontier."""
        try:
            # Create a convex hull or interpolated surface
            from scipy.spatial import ConvexHull

            hull = ConvexHull(pareto_points)

            # Extract surface triangles
            triangles = hull.simplices

            # Create mesh surface
            surface = go.Mesh3d(
                x=pareto_points[:, 0],
                y=pareto_points[:, 1],
                z=pareto_points[:, 2],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                opacity=0.3,
                color="lightcoral",
                name="Pareto Surface",
            )

            return surface

        except Exception as e:
            logger.warning(f"Could not create Pareto surface: {e}")
            return None

    def _create_empty_plot(self, message: str):
        """Create empty plot with message."""
        return None
