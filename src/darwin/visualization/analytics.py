"""
Advanced Analytics Algorithms for Darwin Visualization Engine

This module provides sophisticated analytics algorithms for genetic algorithm
optimization analysis, including convergence analysis, diversity metrics,
performance profiling, and statistical correlation analysis.

Features:
- Convergence rate analysis and prediction
- Population diversity metrics and trends
- Performance profiling and bottleneck detection
- Statistical correlation and dependency analysis
- Multi-objective optimization analysis
- Fitness landscape analysis
- Parameter sensitivity analysis
- Algorithm comparison metrics
"""

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

# Suppress scientific computing warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ConvergenceAnalyzer:
    """Analyzes convergence patterns and predicts optimization behavior."""

    def __init__(self):
        self.convergence_models = {
            "exponential": lambda x, a, b, c: a * np.exp(-b * x) + c,
            "power": lambda x, a, b, c: a * np.power(x + 1, -b) + c,
            "logarithmic": lambda x, a, b, c: a * np.log(b * x + 1) + c,
            "sigmoid": lambda x, a, b, c, d: a / (1 + np.exp(-b * (x - c))) + d,
        }

    def analyze_convergence(
        self, fitness_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze convergence patterns in fitness evolution.

        Args:
            fitness_history: List of fitness records with generation and fitness data

        Returns:
            Dictionary containing convergence analysis results
        """
        if not fitness_history:
            return {"error": "No fitness history provided"}

        df = pd.DataFrame(fitness_history)

        # Extract key metrics
        generations = df["generation"].values
        best_fitness = df["best_fitness"].values
        mean_fitness = df.get("mean_fitness", best_fitness).values

        analysis = {
            "convergence_rate": self._calculate_convergence_rate(
                generations, best_fitness
            ),
            "convergence_point": self._detect_convergence_point(best_fitness),
            "improvement_phases": self._identify_improvement_phases(best_fitness),
            "stagnation_periods": self._detect_stagnation(best_fitness),
            "fitness_trend": self._analyze_fitness_trend(generations, best_fitness),
            "model_fit": self._fit_convergence_models(generations, best_fitness),
            "convergence_prediction": self._predict_convergence(
                generations, best_fitness
            ),
            "efficiency_metrics": self._calculate_efficiency_metrics(df),
        }

        return analysis

    def _calculate_convergence_rate(
        self, generations: np.ndarray, fitness: np.ndarray
    ) -> float:
        """Calculate the convergence rate using fitness improvement per generation."""
        if len(fitness) < 2:
            return 0.0

        # Calculate improvement rate over sliding windows
        window_size = min(10, len(fitness) // 4)
        if window_size < 2:
            return (fitness[-1] - fitness[0]) / len(fitness)

        rates = []
        for i in range(window_size, len(fitness)):
            start_idx = i - window_size
            rate = (fitness[i] - fitness[start_idx]) / window_size
            rates.append(rate)

        return np.mean(rates) if rates else 0.0

    def _detect_convergence_point(
        self, fitness: np.ndarray, threshold: float = 1e-6
    ) -> Optional[int]:
        """Detect the generation where convergence occurs."""
        if len(fitness) < 10:
            return None

        # Calculate rolling variance
        window_size = min(20, len(fitness) // 5)
        variances = []

        for i in range(window_size, len(fitness)):
            window = fitness[i - window_size : i]
            variances.append(np.var(window))

        # Find first point where variance stays below threshold
        for i, var in enumerate(variances):
            if var < threshold:
                # Check if it stays low for at least 10 generations
                if i + 10 < len(variances):
                    if all(v < threshold * 2 for v in variances[i : i + 10]):
                        return i + window_size

        return None

    def _identify_improvement_phases(self, fitness: np.ndarray) -> List[Dict[str, Any]]:
        """Identify distinct improvement phases in the optimization."""
        if len(fitness) < 5:
            return []

        # Calculate improvement rates
        improvements = np.diff(fitness)

        # Find peaks in improvement (significant jumps)
        peaks, properties = find_peaks(improvements, height=np.std(improvements))

        phases = []
        start_gen = 0

        for peak in peaks:
            if peak > start_gen + 5:  # Minimum phase length
                phase = {
                    "start_generation": start_gen,
                    "end_generation": peak,
                    "improvement": fitness[peak] - fitness[start_gen],
                    "rate": (fitness[peak] - fitness[start_gen]) / (peak - start_gen),
                    "type": "gradual",
                }
                phases.append(phase)
                start_gen = peak

        # Add final phase
        if start_gen < len(fitness) - 5:
            phases.append(
                {
                    "start_generation": start_gen,
                    "end_generation": len(fitness) - 1,
                    "improvement": fitness[-1] - fitness[start_gen],
                    "rate": (fitness[-1] - fitness[start_gen])
                    / (len(fitness) - 1 - start_gen),
                    "type": "final",
                }
            )

        return phases

    def _detect_stagnation(
        self, fitness: np.ndarray, min_length: int = 10
    ) -> List[Dict[str, Any]]:
        """Detect periods of stagnation in the optimization."""
        if len(fitness) < min_length * 2:
            return []

        stagnation_threshold = np.std(fitness) * 0.1  # 10% of standard deviation
        stagnations = []
        current_start = None

        for i in range(1, len(fitness)):
            improvement = abs(fitness[i] - fitness[i - 1])

            if improvement < stagnation_threshold:
                if current_start is None:
                    current_start = i - 1
            else:
                if current_start is not None and i - current_start >= min_length:
                    stagnations.append(
                        {
                            "start_generation": current_start,
                            "end_generation": i - 1,
                            "length": i - current_start,
                            "fitness_variance": np.var(fitness[current_start:i]),
                        }
                    )
                current_start = None

        # Check for stagnation at the end
        if current_start is not None and len(fitness) - current_start >= min_length:
            stagnations.append(
                {
                    "start_generation": current_start,
                    "end_generation": len(fitness) - 1,
                    "length": len(fitness) - current_start,
                    "fitness_variance": np.var(fitness[current_start:]),
                }
            )

        return stagnations

    def _analyze_fitness_trend(
        self, generations: np.ndarray, fitness: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze overall fitness trend using statistical methods."""
        if len(fitness) < 3:
            return {"trend": "insufficient_data"}

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            generations, fitness
        )

        # Determine trend type
        if abs(r_value) < 0.1:
            trend_type = "no_trend"
        elif slope > 0:
            trend_type = "improving"
        else:
            trend_type = "degrading"

        # Calculate trend strength
        trend_strength = abs(r_value)

        return {
            "trend": trend_type,
            "strength": trend_strength,
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "confidence": 1 - p_value if p_value < 0.05 else 0.5,
        }

    def _fit_convergence_models(
        self, generations: np.ndarray, fitness: np.ndarray
    ) -> Dict[str, Any]:
        """Fit various mathematical models to the convergence curve."""
        if len(fitness) < 5:
            return {"error": "Insufficient data for model fitting"}

        model_results = {}

        for model_name, model_func in self.convergence_models.items():
            try:
                # Initial parameter guesses
                if model_name == "exponential":
                    p0 = [fitness[0] - fitness[-1], 0.1, fitness[-1]]
                elif model_name == "power":
                    p0 = [fitness[0] - fitness[-1], 0.5, fitness[-1]]
                elif model_name == "logarithmic":
                    p0 = [fitness[-1] - fitness[0], 1.0, fitness[0]]
                elif model_name == "sigmoid":
                    p0 = [
                        fitness[-1] - fitness[0],
                        0.1,
                        len(generations) / 2,
                        fitness[0],
                    ]

                # Fit model
                popt, pcov = curve_fit(
                    model_func, generations, fitness, p0=p0, maxfev=1000
                )

                # Calculate fit quality
                predicted = model_func(generations, *popt)
                r_squared = 1 - (
                    np.sum((fitness - predicted) ** 2)
                    / np.sum((fitness - np.mean(fitness)) ** 2)
                )

                model_results[model_name] = {
                    "parameters": popt.tolist(),
                    "r_squared": r_squared,
                    "rmse": np.sqrt(np.mean((fitness - predicted) ** 2)),
                    "aic": len(fitness)
                    * np.log(np.sum((fitness - predicted) ** 2) / len(fitness))
                    + 2 * len(popt),
                }

            except Exception as e:
                model_results[model_name] = {"error": str(e)}

        # Find best model
        valid_models = {k: v for k, v in model_results.items() if "error" not in v}
        if valid_models:
            best_model = max(valid_models.items(), key=lambda x: x[1]["r_squared"])
            model_results["best_model"] = best_model[0]

        return model_results

    def _predict_convergence(
        self, generations: np.ndarray, fitness: np.ndarray, horizon: int = 50
    ) -> Dict[str, Any]:
        """Predict future convergence behavior."""
        if len(fitness) < 10:
            return {"error": "Insufficient data for prediction"}

        # Use the last 50% of data for prediction
        split_point = len(fitness) // 2
        recent_gens = generations[split_point:]
        recent_fitness = fitness[split_point:]

        try:
            # Fit exponential decay model to recent data
            def exp_model(x, a, b, c):
                return a * np.exp(-b * (x - recent_gens[0])) + c

            popt, _ = curve_fit(exp_model, recent_gens, recent_fitness, maxfev=1000)

            # Predict future values
            future_gens = np.arange(generations[-1] + 1, generations[-1] + horizon + 1)
            predicted_fitness = exp_model(future_gens, *popt)

            # Estimate convergence value
            convergence_value = popt[2]  # c parameter (asymptote)

            # Estimate generations to convergence
            improvement_threshold = abs(fitness[-1] - convergence_value) * 0.01  # 1%
            gens_to_convergence = None

            for i, pred_fitness in enumerate(predicted_fitness):
                if abs(pred_fitness - convergence_value) < improvement_threshold:
                    gens_to_convergence = future_gens[i]
                    break

            return {
                "predicted_convergence_value": convergence_value,
                "generations_to_convergence": gens_to_convergence,
                "prediction_confidence": min(
                    0.95, 1 - np.std(recent_fitness) / abs(np.mean(recent_fitness))
                ),
                "future_trajectory": {
                    "generations": future_gens.tolist(),
                    "predicted_fitness": predicted_fitness.tolist(),
                },
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _calculate_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate various efficiency metrics for the optimization."""
        if len(df) < 2:
            return {"error": "Insufficient data"}

        best_fitness = df["best_fitness"].values
        generations = df["generation"].values

        # Total improvement
        total_improvement = best_fitness[-1] - best_fitness[0]

        # Improvement per generation
        improvement_per_gen = (
            total_improvement / len(generations) if len(generations) > 0 else 0
        )

        # Efficiency score (improvement / evaluations)
        # Assuming population size can be inferred or use default
        pop_size = 50  # Default assumption
        total_evaluations = len(generations) * pop_size
        efficiency_score = (
            abs(total_improvement) / total_evaluations if total_evaluations > 0 else 0
        )

        # Success rate (generations with improvement)
        improvements = np.diff(best_fitness)
        success_rate = (
            np.sum(improvements > 0) / len(improvements) if len(improvements) > 0 else 0
        )

        return {
            "total_improvement": total_improvement,
            "improvement_per_generation": improvement_per_gen,
            "efficiency_score": efficiency_score,
            "success_rate": success_rate,
            "total_evaluations": total_evaluations,
            "final_fitness": best_fitness[-1],
            "convergence_speed": self._calculate_convergence_speed(best_fitness),
        }

    def _calculate_convergence_speed(self, fitness: np.ndarray) -> float:
        """Calculate convergence speed metric."""
        if len(fitness) < 3:
            return 0.0

        # Calculate the area under the improvement curve
        improvements = np.diff(fitness)
        cumulative_improvement = np.cumsum(np.abs(improvements))

        # Normalize by total improvement and generations
        total_improvement = np.sum(np.abs(improvements))
        if total_improvement == 0:
            return 0.0

        # Speed is inversely related to the area under the curve
        normalized_area = np.sum(cumulative_improvement) / (
            len(improvements) * total_improvement
        )
        convergence_speed = 1.0 / (1.0 + normalized_area)

        return convergence_speed


class DiversityAnalyzer:
    """Analyzes population diversity metrics and trends."""

    def analyze_diversity(
        self,
        population_data: List[Dict[str, Any]],
        fitness_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze diversity patterns in population evolution.

        Args:
            population_data: List of population snapshots
            fitness_history: List of fitness records

        Returns:
            Dictionary containing diversity analysis results
        """
        if not population_data and not fitness_history:
            return {"error": "No data provided"}

        analysis = {}

        # Analyze explicit diversity data if available
        if fitness_history:
            df = pd.DataFrame(fitness_history)
            if "diversity" in df.columns:
                analysis.update(self._analyze_diversity_trend(df["diversity"].values))

        # Analyze population-based diversity if available
        if population_data:
            analysis.update(self._analyze_population_diversity(population_data))

        return analysis

    def _analyze_diversity_trend(self, diversity_values: np.ndarray) -> Dict[str, Any]:
        """Analyze trends in diversity metrics."""
        if len(diversity_values) < 2:
            return {"diversity_trend": "insufficient_data"}

        # Calculate trend
        generations = np.arange(len(diversity_values))
        slope, _, r_value, p_value, _ = stats.linregress(generations, diversity_values)

        # Classify trend
        if abs(r_value) < 0.1:
            trend_type = "stable"
        elif slope < 0:
            trend_type = "decreasing"
        else:
            trend_type = "increasing"

        # Calculate diversity statistics
        diversity_stats = {
            "mean_diversity": np.mean(diversity_values),
            "min_diversity": np.min(diversity_values),
            "max_diversity": np.max(diversity_values),
            "diversity_variance": np.var(diversity_values),
            "diversity_trend": trend_type,
            "trend_strength": abs(r_value),
            "trend_slope": slope,
            "diversity_loss_rate": -slope if slope < 0 else 0,
        }

        # Detect diversity crisis points
        crisis_threshold = np.mean(diversity_values) - 2 * np.std(diversity_values)
        crisis_points = np.where(diversity_values < crisis_threshold)[0]

        if len(crisis_points) > 0:
            diversity_stats["diversity_crises"] = {
                "count": len(crisis_points),
                "generations": crisis_points.tolist(),
                "severity": float(np.mean(diversity_values[crisis_points])),
            }

        return diversity_stats

    def _analyze_population_diversity(
        self, population_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze diversity from population data."""
        if not population_data:
            return {"population_diversity": "no_data"}

        diversity_measures = []

        for pop_snapshot in population_data:
            if "population" in pop_snapshot:
                pop_array = np.array(pop_snapshot["population"])

                # Calculate various diversity measures
                measures = {
                    "generation": pop_snapshot.get("generation", 0),
                    "euclidean_diversity": self._calculate_euclidean_diversity(
                        pop_array
                    ),
                    "hamming_diversity": self._calculate_hamming_diversity(pop_array),
                    "entropy_diversity": self._calculate_entropy_diversity(pop_array),
                    "cluster_diversity": self._calculate_cluster_diversity(pop_array),
                }
                diversity_measures.append(measures)

        if not diversity_measures:
            return {"population_diversity": "calculation_failed"}

        # Aggregate results
        df = pd.DataFrame(diversity_measures)

        return {
            "population_diversity_evolution": df.to_dict("records"),
            "average_euclidean_diversity": float(np.mean(df["euclidean_diversity"])),
            "average_entropy_diversity": float(np.mean(df["entropy_diversity"])),
            "diversity_correlation": self._analyze_diversity_correlation(df),
        }

    def _calculate_euclidean_diversity(self, population: np.ndarray) -> float:
        """Calculate average pairwise Euclidean distance."""
        if len(population) < 2:
            return 0.0

        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)

        return float(np.mean(distances))

    def _calculate_hamming_diversity(self, population: np.ndarray) -> float:
        """Calculate average Hamming distance for discrete variables."""
        if len(population) < 2:
            return 0.0

        # Discretize continuous variables
        discretized = np.round(population * 10).astype(int)

        distances = []
        for i in range(len(discretized)):
            for j in range(i + 1, len(discretized)):
                dist = np.sum(discretized[i] != discretized[j]) / len(discretized[i])
                distances.append(dist)

        return float(np.mean(distances))

    def _calculate_entropy_diversity(self, population: np.ndarray) -> float:
        """Calculate entropy-based diversity measure."""
        if len(population) < 2:
            return 0.0

        # Discretize and calculate entropy for each dimension
        entropies = []
        for dim in range(population.shape[1]):
            values = np.round(population[:, dim] * 10).astype(int)
            unique, counts = np.unique(values, return_counts=True)
            probabilities = counts / len(values)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropies.append(entropy)

        return float(np.mean(entropies))

    def _calculate_cluster_diversity(
        self, population: np.ndarray, n_clusters: int = 5
    ) -> float:
        """Calculate diversity based on clustering."""
        if len(population) < n_clusters:
            return 0.0

        try:
            # Perform clustering
            kmeans = KMeans(
                n_clusters=min(n_clusters, len(population)), random_state=42, n_init=10
            )
            labels = kmeans.fit_predict(population)

            # Calculate silhouette score as diversity measure
            if len(np.unique(labels)) > 1:
                diversity = silhouette_score(population, labels)
                return float(diversity)
            else:
                return 0.0

        except Exception:
            return 0.0

    def _analyze_diversity_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different diversity measures."""
        numeric_cols = [
            "euclidean_diversity",
            "hamming_diversity",
            "entropy_diversity",
            "cluster_diversity",
        ]
        numeric_cols = [col for col in numeric_cols if col in df.columns]

        if len(numeric_cols) < 2:
            return {"correlation": "insufficient_measures"}

        correlation_matrix = df[numeric_cols].corr()

        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "strong_correlations": self._find_strong_correlations(correlation_matrix),
        }

    def _find_strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find strong correlations between diversity measures."""
        strong_corrs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    strong_corrs.append(
                        {
                            "measure_1": corr_matrix.columns[i],
                            "measure_2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                            "strength": "strong"
                            if abs(corr_value) > 0.8
                            else "moderate",
                        }
                    )

        return strong_corrs


class PerformanceAnalyzer:
    """Analyzes algorithm performance and identifies bottlenecks."""

    def analyze_performance(self, optimization_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze optimization performance metrics.

        Args:
            optimization_data: Complete optimization data including timing info

        Returns:
            Dictionary containing performance analysis results
        """
        analysis = {
            "computational_efficiency": self._analyze_computational_efficiency(
                optimization_data
            ),
            "convergence_efficiency": self._analyze_convergence_efficiency(
                optimization_data
            ),
            "scalability_analysis": self._analyze_scalability(optimization_data),
            "resource_utilization": self._analyze_resource_utilization(
                optimization_data
            ),
            "bottleneck_detection": self._detect_bottlenecks(optimization_data),
        }

        return analysis

    def _analyze_computational_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze computational efficiency metrics."""
        fitness_history = data.get("fitness_history", [])
        metadata = data.get("metadata", {})

        if not fitness_history:
            return {"error": "No fitness history available"}

        # Calculate time-based metrics if available
        if "timestamp" in fitness_history[0]:
            timestamps = [record["timestamp"] for record in fitness_history]
            time_per_generation = []

            for i in range(1, len(timestamps)):
                if hasattr(timestamps[i], "timestamp") and hasattr(
                    timestamps[i - 1], "timestamp"
                ):
                    time_diff = (
                        timestamps[i].timestamp() - timestamps[i - 1].timestamp()
                    )
                    time_per_generation.append(time_diff)

            if time_per_generation:
                return {
                    "average_time_per_generation": float(np.mean(time_per_generation)),
                    "total_runtime": float(np.sum(time_per_generation)),
                    "time_variance": float(np.var(time_per_generation)),
                    "efficiency_score": self._calculate_efficiency_score(
                        fitness_history, time_per_generation
                    ),
                }

        # Fallback to generation-based analysis
        total_generations = len(fitness_history)
        population_size = metadata.get("population_size", 50)
        total_evaluations = total_generations * population_size

        return {
            "total_generations": total_generations,
            "total_evaluations": total_evaluations,
            "convergence_efficiency": self._calculate_convergence_efficiency_score(
                fitness_history
            ),
        }

    def _calculate_efficiency_score(
        self, fitness_history: List[Dict], time_per_gen: List[float]
    ) -> float:
        """Calculate efficiency score based on improvement per time unit."""
        if not fitness_history or not time_per_gen:
            return 0.0

        fitness_values = [record["best_fitness"] for record in fitness_history]
        total_improvement = abs(fitness_values[-1] - fitness_values[0])
        total_time = sum(time_per_gen)

        if total_time == 0:
            return 0.0

        return total_improvement / total_time

    def _analyze_convergence_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how efficiently the algorithm converges."""
        fitness_history = data.get("fitness_history", [])

        if not fitness_history:
            return {"error": "No fitness history available"}

        fitness_values = [record["best_fitness"] for record in fitness_history]

        # Calculate various efficiency metrics
        return {
            "convergence_rate": self._calculate_convergence_rate(fitness_values),
            "early_convergence_potential": self._assess_early_convergence(
                fitness_values
            ),
            "improvement_distribution": self._analyze_improvement_distribution(
                fitness_values
            ),
            "convergence_quality": self._assess_convergence_quality(fitness_values),
        }

    def _calculate_convergence_rate(self, fitness_values: List[float]) -> float:
        """Calculate the rate of convergence."""
        if len(fitness_values) < 2:
            return 0.0

        improvements = []
        for i in range(1, len(fitness_values)):
            improvement = abs(fitness_values[i] - fitness_values[i - 1])
            improvements.append(improvement)

        # Calculate exponential decay rate
        if len(improvements) > 5:
            x = np.arange(len(improvements))
            y = np.array(improvements)

            # Avoid log of zero
            y = np.maximum(y, 1e-10)

            try:
                # Fit exponential decay
                slope, _, _, _, _ = stats.linregress(x, np.log(y))
                return float(-slope)  # Positive value for decay rate
            except:
                return float(np.mean(improvements))

        return float(np.mean(improvements))

    def _assess_early_convergence(
        self, fitness_values: List[float], early_fraction: float = 0.3
    ) -> Dict[str, Any]:
        """Assess potential for early convergence."""
        if len(fitness_values) < 10:
            return {"assessment": "insufficient_data"}

        early_point = int(len(fitness_values) * early_fraction)
        early_improvement = abs(fitness_values[early_point] - fitness_values[0])
        total_improvement = abs(fitness_values[-1] - fitness_values[0])

        if total_improvement == 0:
            early_convergence_ratio = 0.0
        else:
            early_convergence_ratio = early_improvement / total_improvement

        return {
            "early_convergence_ratio": float(early_convergence_ratio),
            "early_convergence_potential": early_convergence_ratio > 0.8,
            "recommended_early_stopping": early_point
            if early_convergence_ratio > 0.95
            else None,
        }

    def _analyze_improvement_distribution(
        self, fitness_values: List[float]
    ) -> Dict[str, Any]:
        """Analyze the distribution of improvements across generations."""
        if len(fitness_values) < 2:
            return {"error": "Insufficient data"}

        improvements = np.diff(fitness_values)
        improvements = np.abs(improvements)

        # Remove zero improvements for analysis
        non_zero_improvements = improvements[improvements > 0]

        if len(non_zero_improvements) == 0:
            return {"distribution": "no_improvements"}

        return {
            "improvement_count": len(non_zero_improvements),
            "improvement_ratio": len(non_zero_improvements) / len(improvements),
            "mean_improvement": float(np.mean(non_zero_improvements)),
            "improvement_variance": float(np.var(non_zero_improvements)),
            "large_improvements": int(
                np.sum(non_zero_improvements > np.mean(non_zero_improvements) * 2)
            ),
            "improvement_consistency": float(
                1.0
                / (1.0 + np.var(non_zero_improvements) / np.mean(non_zero_improvements))
            ),
        }

    def _assess_convergence_quality(
        self, fitness_values: List[float]
    ) -> Dict[str, Any]:
        """Assess the quality of convergence."""
        if len(fitness_values) < 10:
            return {"quality": "insufficient_data"}

        # Analyze final portion stability
        final_portion = fitness_values[-10:]
        stability = 1.0 / (1.0 + np.var(final_portion))

        # Analyze overall trajectory smoothness
        second_derivatives = np.diff(fitness_values, 2)
        smooth
