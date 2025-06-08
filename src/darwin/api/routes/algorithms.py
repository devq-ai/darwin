"""
Darwin API Algorithms Routes

This module implements API endpoints for retrieving information about available
genetic algorithm configurations and optimization algorithms.
"""

from typing import Optional

import logfire
from fastapi import APIRouter, HTTPException, Query

from darwin.api.models.responses import AlgorithmInfo, AlgorithmsResponse

router = APIRouter()

# Available genetic algorithm configurations
AVAILABLE_ALGORITHMS = [
    {
        "name": "Standard Genetic Algorithm",
        "description": "Classic genetic algorithm with configurable operators for general-purpose optimization.",
        "parameters": {
            "population_size": {"type": "int", "range": [10, 1000], "default": 50},
            "max_generations": {"type": "int", "range": [1, 10000], "default": 100},
            "crossover_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.8,
            },
            "mutation_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.1,
            },
            "elitism": {"type": "bool", "default": True},
            "tournament_size": {"type": "int", "range": [2, 20], "default": 3},
        },
        "suitable_for": ["continuous", "discrete", "binary"],
        "complexity": "O(g * n * f)",
        "references": [
            "Holland, J. H. (1975). Adaptation in Natural and Artificial Systems.",
            "Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning.",
        ],
    },
    {
        "name": "NSGA-II",
        "description": "Non-dominated Sorting Genetic Algorithm II for multi-objective optimization with crowding distance.",
        "parameters": {
            "population_size": {"type": "int", "range": [20, 500], "default": 100},
            "max_generations": {"type": "int", "range": [10, 5000], "default": 250},
            "crossover_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.9,
            },
            "mutation_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.1,
            },
            "eta_c": {"type": "float", "range": [0.5, 50.0], "default": 20.0},
            "eta_m": {"type": "float", "range": [0.5, 50.0], "default": 20.0},
        },
        "suitable_for": ["multi_objective", "continuous"],
        "complexity": "O(g * n² * m)",
        "references": [
            "Deb, K., et al. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.",
            "IEEE Transactions on Evolutionary Computation, 6(2), 182-197.",
        ],
    },
    {
        "name": "NSGA-III",
        "description": "Non-dominated Sorting Genetic Algorithm III optimized for many-objective optimization problems.",
        "parameters": {
            "population_size": {"type": "int", "range": [50, 1000], "default": 100},
            "max_generations": {"type": "int", "range": [10, 5000], "default": 300},
            "crossover_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 1.0,
            },
            "mutation_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.1,
            },
            "reference_points": {"type": "int", "range": [1, 500], "default": 91},
        },
        "suitable_for": ["many_objective", "continuous"],
        "complexity": "O(g * n * m²)",
        "references": [
            "Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm using reference-point-based nondominated sorting approach.",
            "IEEE Transactions on Evolutionary Computation, 18(4), 577-601.",
        ],
    },
    {
        "name": "Differential Evolution",
        "description": "Population-based optimization algorithm using differential mutation and one-to-one competition.",
        "parameters": {
            "population_size": {"type": "int", "range": [10, 500], "default": 50},
            "max_generations": {"type": "int", "range": [1, 5000], "default": 200},
            "mutation_factor": {"type": "float", "range": [0.0, 2.0], "default": 0.8},
            "crossover_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.7,
            },
            "strategy": {
                "type": "string",
                "options": ["rand/1", "best/1", "rand/2", "best/2"],
                "default": "rand/1",
            },
        },
        "suitable_for": ["continuous", "real_valued"],
        "complexity": "O(g * n * d)",
        "references": [
            "Storn, R., & Price, K. (1997). Differential evolution–a simple and efficient heuristic for global optimization over continuous spaces.",
            "Journal of Global Optimization, 11(4), 341-359.",
        ],
    },
    {
        "name": "Particle Swarm Optimization",
        "description": "Swarm intelligence algorithm inspired by social behavior of birds and fish for continuous optimization.",
        "parameters": {
            "swarm_size": {"type": "int", "range": [10, 200], "default": 30},
            "max_iterations": {"type": "int", "range": [1, 2000], "default": 100},
            "inertia_weight": {"type": "float", "range": [0.1, 1.0], "default": 0.7},
            "cognitive_coefficient": {
                "type": "float",
                "range": [0.0, 4.0],
                "default": 1.5,
            },
            "social_coefficient": {
                "type": "float",
                "range": [0.0, 4.0],
                "default": 1.5,
            },
            "velocity_clamp": {"type": "float", "range": [0.1, 1.0], "default": 0.5},
        },
        "suitable_for": ["continuous", "real_valued"],
        "complexity": "O(i * n * d)",
        "references": [
            "Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.",
            "Proceedings of ICNN'95-International Conference on Neural Networks.",
        ],
    },
    {
        "name": "Covariance Matrix Adaptation Evolution Strategy",
        "description": "Advanced evolution strategy using covariance matrix adaptation for high-dimensional continuous problems.",
        "parameters": {
            "population_size": {"type": "int", "range": [4, 300], "default": 50},
            "max_generations": {"type": "int", "range": [1, 5000], "default": 300},
            "sigma": {"type": "float", "range": [0.01, 10.0], "default": 0.3},
            "learning_rate_c": {"type": "float", "range": [0.0, 1.0], "default": 0.004},
            "learning_rate_cc": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.004,
            },
        },
        "suitable_for": ["continuous", "high_dimensional"],
        "complexity": "O(g * n * d²)",
        "references": [
            "Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies.",
            "Evolutionary Computation, 9(2), 159-195.",
        ],
    },
    {
        "name": "Genetic Programming",
        "description": "Evolutionary algorithm that evolves computer programs represented as tree structures.",
        "parameters": {
            "population_size": {"type": "int", "range": [50, 1000], "default": 200},
            "max_generations": {"type": "int", "range": [1, 1000], "default": 50},
            "crossover_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.8,
            },
            "mutation_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.2,
            },
            "max_tree_depth": {"type": "int", "range": [3, 20], "default": 8},
            "tournament_size": {"type": "int", "range": [2, 10], "default": 3},
        },
        "suitable_for": ["symbolic_regression", "classification", "program_evolution"],
        "complexity": "O(g * n * t)",
        "references": [
            "Koza, J. R. (1992). Genetic Programming: On the Programming of Computers by Means of Natural Selection.",
            "MIT Press.",
        ],
    },
    {
        "name": "Memetic Algorithm",
        "description": "Hybrid evolutionary algorithm combining genetic algorithms with local search for improved convergence.",
        "parameters": {
            "population_size": {"type": "int", "range": [20, 200], "default": 50},
            "max_generations": {"type": "int", "range": [1, 1000], "default": 100},
            "crossover_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.8,
            },
            "mutation_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.1,
            },
            "local_search_probability": {
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.3,
            },
            "local_search_iterations": {
                "type": "int",
                "range": [1, 100],
                "default": 10,
            },
        },
        "suitable_for": ["continuous", "combinatorial", "hybrid_problems"],
        "complexity": "O(g * n * (f + l))",
        "references": [
            "Moscato, P. (1989). On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms.",
            "Caltech Concurrent Computation Program, C3P Report, 826.",
        ],
    },
    {
        "name": "Island Model Genetic Algorithm",
        "description": "Parallel genetic algorithm with multiple subpopulations (islands) and periodic migration.",
        "parameters": {
            "islands_count": {"type": "int", "range": [2, 20], "default": 4},
            "population_per_island": {"type": "int", "range": [10, 200], "default": 25},
            "max_generations": {"type": "int", "range": [1, 2000], "default": 150},
            "migration_rate": {"type": "float", "range": [0.0, 1.0], "default": 0.1},
            "migration_interval": {"type": "int", "range": [1, 100], "default": 10},
            "topology": {
                "type": "string",
                "options": ["ring", "star", "mesh"],
                "default": "ring",
            },
        },
        "suitable_for": [
            "parallel_processing",
            "large_populations",
            "diversity_maintenance",
        ],
        "complexity": "O(g * i * n * f)",
        "references": [
            "Whitley, D., Rana, S., & Heckendorn, R. B. (1998). The island model genetic algorithm: On separability, population size and convergence.",
            "Journal of Computing and Information Technology, 7(1), 33-47.",
        ],
    },
    {
        "name": "Adaptive Genetic Algorithm",
        "description": "Self-adaptive genetic algorithm that adjusts parameters during evolution based on population diversity.",
        "parameters": {
            "population_size": {"type": "int", "range": [20, 300], "default": 60},
            "max_generations": {"type": "int", "range": [1, 2000], "default": 200},
            "initial_crossover_prob": {
                "type": "float",
                "range": [0.5, 1.0],
                "default": 0.8,
            },
            "initial_mutation_prob": {
                "type": "float",
                "range": [0.01, 0.5],
                "default": 0.1,
            },
            "adaptation_rate": {"type": "float", "range": [0.01, 0.1], "default": 0.05},
            "diversity_threshold": {
                "type": "float",
                "range": [0.01, 0.5],
                "default": 0.1,
            },
        },
        "suitable_for": [
            "dynamic_environments",
            "parameter_tuning",
            "robust_optimization",
        ],
        "complexity": "O(g * n * f + d)",
        "references": [
            "Srinivas, M., & Patnaik, L. M. (1994). Adaptive probabilities of crossover and mutation in genetic algorithms.",
            "IEEE Transactions on Systems, Man, and Cybernetics, 24(4), 656-667.",
        ],
    },
]


@router.get("/algorithms", response_model=AlgorithmsResponse)
async def get_available_algorithms(
    suitable_for: Optional[str] = Query(
        None, description="Filter algorithms by problem type"
    ),
    complexity: Optional[str] = Query(None, description="Filter by complexity level"),
):
    """
    Get available optimization algorithms.

    Returns information about all available genetic algorithms and evolutionary
    optimization methods, including their parameters, suitable problem types,
    and performance characteristics.
    """
    try:
        # Filter algorithms based on query parameters
        filtered_algorithms = AVAILABLE_ALGORITHMS.copy()

        if suitable_for:
            filtered_algorithms = [
                alg
                for alg in filtered_algorithms
                if suitable_for.lower() in [s.lower() for s in alg["suitable_for"]]
            ]

        if complexity:
            # Simple complexity filtering based on notation
            complexity_levels = {
                "low": ["O(g * n * f)", "O(i * n * d)"],
                "medium": ["O(g * n² * m)", "O(g * n * d)", "O(g * n * (f + l))"],
                "high": ["O(g * n * d²)", "O(g * n * t)", "O(g * i * n * f)"],
            }

            if complexity.lower() in complexity_levels:
                target_complexities = complexity_levels[complexity.lower()]
                filtered_algorithms = [
                    alg
                    for alg in filtered_algorithms
                    if alg["complexity"] in target_complexities
                ]

        # Convert to response models
        algorithms = [AlgorithmInfo(**algorithm) for algorithm in filtered_algorithms]

        logfire.info(
            "Available algorithms retrieved",
            total_algorithms=len(algorithms),
            suitable_for_filter=suitable_for,
            complexity_filter=complexity,
        )

        return AlgorithmsResponse(algorithms=algorithms, total_count=len(algorithms))

    except Exception as e:
        logfire.error(
            "Failed to retrieve available algorithms",
            error=str(e),
            suitable_for=suitable_for,
            complexity=complexity,
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve available algorithms"
        )


@router.get("/algorithms/{algorithm_name}", response_model=AlgorithmInfo)
async def get_algorithm_by_name(algorithm_name: str):
    """
    Get specific algorithm information by name.

    Returns detailed information about a specific optimization algorithm
    including parameters, suitable problem types, and implementation details.
    """
    try:
        # Find algorithm by name (case-insensitive)
        algorithm_data = None
        for algorithm in AVAILABLE_ALGORITHMS:
            if algorithm["name"].lower().replace(
                " ", "_"
            ) == algorithm_name.lower().replace(" ", "_"):
                algorithm_data = algorithm
                break

        if not algorithm_data:
            raise HTTPException(
                status_code=404, detail=f"Algorithm '{algorithm_name}' not found"
            )

        logfire.info(
            "Algorithm information retrieved by name",
            algorithm_name=algorithm_name,
            found_algorithm=algorithm_data["name"],
        )

        return AlgorithmInfo(**algorithm_data)

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to retrieve algorithm information",
            algorithm_name=algorithm_name,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve algorithm information"
        )


@router.get("/algorithms/types")
async def get_algorithm_types():
    """
    Get available algorithm types and categories.

    Returns information about different types of optimization algorithms
    and their primary characteristics.
    """
    try:
        # Extract unique problem types from all algorithms
        all_types = set()
        for algorithm in AVAILABLE_ALGORITHMS:
            all_types.update(algorithm["suitable_for"])

        # Categorize algorithm types
        type_categories = {
            "Problem Types": {
                "continuous": "Continuous optimization problems with real-valued variables",
                "discrete": "Discrete optimization with integer variables",
                "binary": "Binary optimization problems (0/1 variables)",
                "combinatorial": "Combinatorial optimization problems",
                "multi_objective": "Multi-objective optimization with trade-offs",
                "many_objective": "Many-objective optimization (>3 objectives)",
            },
            "Application Areas": {
                "real_valued": "Real-valued function optimization",
                "symbolic_regression": "Symbolic regression and formula discovery",
                "classification": "Classification problem optimization",
                "program_evolution": "Automatic program generation",
                "parameter_tuning": "Hyperparameter optimization",
            },
            "Special Features": {
                "parallel_processing": "Parallel and distributed optimization",
                "high_dimensional": "High-dimensional optimization problems",
                "dynamic_environments": "Time-varying optimization problems",
                "diversity_maintenance": "Algorithms focused on population diversity",
                "robust_optimization": "Robust optimization under uncertainty",
            },
        }

        # Build response with counts
        result = {}
        for category, types in type_categories.items():
            result[category] = {}
            for type_name, description in types.items():
                if type_name in all_types:
                    count = len(
                        [
                            alg
                            for alg in AVAILABLE_ALGORITHMS
                            if type_name in alg["suitable_for"]
                        ]
                    )
                    result[category][type_name] = {
                        "description": description,
                        "algorithm_count": count,
                    }

        logfire.info(
            "Algorithm types retrieved",
            total_types=len(all_types),
            categories=len(type_categories),
        )

        return {
            "types": result,
            "total_algorithms": len(AVAILABLE_ALGORITHMS),
            "unique_types": len(all_types),
        }

    except Exception as e:
        logfire.error("Failed to retrieve algorithm types", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve algorithm types"
        )


@router.get("/algorithms/complexity")
async def get_complexity_analysis():
    """
    Get algorithm complexity analysis and performance characteristics.

    Returns information about algorithm computational complexity and
    performance trade-offs for different problem sizes.
    """
    try:
        # Analyze complexity patterns
        complexity_groups = {}
        for algorithm in AVAILABLE_ALGORITHMS:
            complexity = algorithm["complexity"]
            if complexity not in complexity_groups:
                complexity_groups[complexity] = []
            complexity_groups[complexity].append(
                {"name": algorithm["name"], "suitable_for": algorithm["suitable_for"]}
            )

        # Add complexity explanations
        complexity_explanations = {
            "O(g * n * f)": {
                "description": "Linear complexity in generations, population size, and fitness evaluations",
                "performance": "Good for most problems with moderate populations",
                "scalability": "Scales well with problem size",
            },
            "O(g * n² * m)": {
                "description": "Quadratic in population size, linear in objectives (multi-objective)",
                "performance": "Higher cost for large populations but handles multiple objectives well",
                "scalability": "May be slower for very large populations",
            },
            "O(g * n * d²)": {
                "description": "Quadratic in problem dimensionality",
                "performance": "Excellent for high-dimensional continuous problems",
                "scalability": "Cost increases with problem dimensionality",
            },
            "O(i * n * d)": {
                "description": "Linear in iterations, population, and dimensions",
                "performance": "Very efficient for continuous optimization",
                "scalability": "Excellent scalability characteristics",
            },
            "O(g * n * t)": {
                "description": "Linear in tree evaluation complexity (for GP)",
                "performance": "Depends on tree size and evaluation complexity",
                "scalability": "Controlled by maximum tree depth limits",
            },
            "O(g * n * (f + l))": {
                "description": "Combined genetic and local search complexity",
                "performance": "Higher cost but improved solution quality",
                "scalability": "Depends on local search intensity",
            },
            "O(g * i * n * f)": {
                "description": "Multiplied by number of islands (parallel GA)",
                "performance": "Can be parallelized effectively",
                "scalability": "Excellent with proper parallelization",
            },
            "O(g * n * f + d)": {
                "description": "Additional diversity calculation overhead",
                "performance": "Slight overhead for adaptation benefits",
                "scalability": "Minimal impact on overall performance",
            },
        }

        # Combine groups with explanations
        result = {}
        for complexity, algorithms in complexity_groups.items():
            result[complexity] = {
                "algorithms": algorithms,
                "algorithm_count": len(algorithms),
                "explanation": complexity_explanations.get(
                    complexity,
                    {
                        "description": "Complex algorithmic behavior",
                        "performance": "Varies by implementation",
                        "scalability": "Depends on problem characteristics",
                    },
                ),
            }

        logfire.info(
            "Algorithm complexity analysis retrieved",
            complexity_groups=len(complexity_groups),
            total_algorithms=len(AVAILABLE_ALGORITHMS),
        )

        return {
            "complexity_analysis": result,
            "legend": {
                "g": "Number of generations/iterations",
                "n": "Population size",
                "f": "Fitness evaluation complexity",
                "m": "Number of objectives",
                "d": "Problem dimensionality",
                "i": "Number of islands/subpopulations",
                "t": "Tree evaluation complexity",
                "l": "Local search complexity",
            },
            "performance_tips": [
                "Choose algorithms with appropriate complexity for your problem size",
                "Consider parallel algorithms for large-scale problems",
                "Balance exploration vs exploitation based on problem characteristics",
                "Use adaptive algorithms for problems with unknown optimal parameters",
            ],
        }

    except Exception as e:
        logfire.error("Failed to retrieve complexity analysis", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve complexity analysis"
        )
