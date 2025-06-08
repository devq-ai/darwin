"""
Darwin API Templates Routes

This module implements API endpoints for managing optimization problem templates.
Provides predefined problem templates for common optimization scenarios.
"""

from typing import Optional

import logfire
from fastapi import APIRouter, HTTPException, Query

from darwin.api.models.responses import ProblemTemplate, TemplatesResponse

router = APIRouter()

# Predefined problem templates
PROBLEM_TEMPLATES = [
    {
        "id": "sphere_function",
        "name": "Sphere Function",
        "description": "Classic sphere function optimization problem. Minimize sum of squares.",
        "category": "continuous",
        "difficulty": "easy",
        "variables_count": 2,
        "constraints_count": 0,
        "default_config": {
            "population_size": 50,
            "max_generations": 100,
            "selection_type": "tournament",
            "crossover_type": "arithmetic",
            "mutation_type": "gaussian",
        },
        "example_usage": "Good starting problem for testing genetic algorithms on continuous optimization.",
    },
    {
        "id": "rastrigin_function",
        "name": "Rastrigin Function",
        "description": "Highly multimodal function with many local optima. Tests algorithm's ability to escape local minima.",
        "category": "continuous",
        "difficulty": "hard",
        "variables_count": 10,
        "constraints_count": 0,
        "default_config": {
            "population_size": 100,
            "max_generations": 500,
            "selection_type": "tournament",
            "crossover_type": "blend",
            "mutation_type": "polynomial",
        },
        "example_usage": "Challenging benchmark for testing genetic algorithm performance on multimodal landscapes.",
    },
    {
        "id": "knapsack_problem",
        "name": "0/1 Knapsack Problem",
        "description": "Classic combinatorial optimization problem. Maximize value while staying within weight constraint.",
        "category": "discrete",
        "difficulty": "medium",
        "variables_count": 20,
        "constraints_count": 1,
        "default_config": {
            "population_size": 80,
            "max_generations": 200,
            "selection_type": "rank",
            "crossover_type": "uniform",
            "mutation_type": "random_resetting",
        },
        "example_usage": "Fundamental problem in combinatorial optimization and operations research.",
    },
    {
        "id": "traveling_salesman",
        "name": "Traveling Salesman Problem",
        "description": "Find shortest route visiting all cities exactly once. Classic NP-hard problem.",
        "category": "permutation",
        "difficulty": "hard",
        "variables_count": 25,
        "constraints_count": 0,
        "default_config": {
            "population_size": 100,
            "max_generations": 1000,
            "selection_type": "tournament",
            "crossover_type": "order",
            "mutation_type": "swap",
        },
        "example_usage": "Benchmark problem for permutation-based optimization algorithms.",
    },
    {
        "id": "portfolio_optimization",
        "name": "Portfolio Optimization",
        "description": "Optimize investment portfolio to maximize return while minimizing risk (Markowitz model).",
        "category": "continuous",
        "difficulty": "medium",
        "variables_count": 10,
        "constraints_count": 2,
        "default_config": {
            "population_size": 100,
            "max_generations": 300,
            "selection_type": "tournament",
            "crossover_type": "arithmetic",
            "mutation_type": "gaussian",
        },
        "example_usage": "Real-world financial optimization problem with risk-return trade-offs.",
    },
    {
        "id": "neural_network_topology",
        "name": "Neural Network Topology Optimization",
        "description": "Optimize neural network architecture including layer sizes and connections.",
        "category": "discrete",
        "difficulty": "hard",
        "variables_count": 15,
        "constraints_count": 3,
        "default_config": {
            "population_size": 50,
            "max_generations": 100,
            "selection_type": "tournament",
            "crossover_type": "uniform",
            "mutation_type": "random_resetting",
        },
        "example_usage": "Automated machine learning (AutoML) problem for neural architecture search.",
    },
    {
        "id": "job_shop_scheduling",
        "name": "Job Shop Scheduling",
        "description": "Schedule jobs on machines to minimize makespan. Classic scheduling optimization problem.",
        "category": "permutation",
        "difficulty": "hard",
        "variables_count": 30,
        "constraints_count": 5,
        "default_config": {
            "population_size": 80,
            "max_generations": 500,
            "selection_type": "rank",
            "crossover_type": "order",
            "mutation_type": "insertion",
        },
        "example_usage": "Manufacturing and production scheduling optimization problem.",
    },
    {
        "id": "feature_selection",
        "name": "Feature Selection",
        "description": "Select optimal subset of features for machine learning models to improve performance.",
        "category": "binary",
        "difficulty": "medium",
        "variables_count": 50,
        "constraints_count": 1,
        "default_config": {
            "population_size": 60,
            "max_generations": 200,
            "selection_type": "tournament",
            "crossover_type": "uniform",
            "mutation_type": "bit_flip",
        },
        "example_usage": "Machine learning preprocessing step to reduce dimensionality and improve model performance.",
    },
    {
        "id": "multi_objective_optimization",
        "name": "Multi-Objective Test Problem (ZDT1)",
        "description": "Classic multi-objective optimization test problem with known Pareto front.",
        "category": "continuous",
        "difficulty": "medium",
        "variables_count": 5,
        "constraints_count": 0,
        "default_config": {
            "population_size": 100,
            "max_generations": 250,
            "selection_type": "nsga2",
            "crossover_type": "sbx",
            "mutation_type": "polynomial",
        },
        "example_usage": "Benchmark for testing multi-objective genetic algorithms like NSGA-II.",
    },
    {
        "id": "constrained_optimization",
        "name": "Constrained Engineering Design",
        "description": "Engineering design optimization with multiple constraints (pressure vessel design).",
        "category": "continuous",
        "difficulty": "medium",
        "variables_count": 4,
        "constraints_count": 4,
        "default_config": {
            "population_size": 60,
            "max_generations": 200,
            "selection_type": "tournament",
            "crossover_type": "arithmetic",
            "mutation_type": "gaussian",
        },
        "example_usage": "Real-world engineering optimization with equality and inequality constraints.",
    },
]


@router.get("/templates", response_model=TemplatesResponse)
async def get_problem_templates(
    category: Optional[str] = Query(None, description="Filter by problem category"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty level"),
):
    """
    Get available problem templates.

    Returns a list of predefined optimization problem templates that can be used
    as starting points for optimization runs. Templates include problem definitions,
    recommended algorithm configurations, and usage examples.
    """
    try:
        # Filter templates based on query parameters
        filtered_templates = PROBLEM_TEMPLATES.copy()

        if category:
            filtered_templates = [
                t
                for t in filtered_templates
                if t["category"].lower() == category.lower()
            ]

        if difficulty:
            filtered_templates = [
                t
                for t in filtered_templates
                if t["difficulty"].lower() == difficulty.lower()
            ]

        # Convert to response models
        templates = [ProblemTemplate(**template) for template in filtered_templates]

        # Get unique categories
        categories = list(set(t["category"] for t in PROBLEM_TEMPLATES))

        logfire.info(
            "Problem templates retrieved",
            total_templates=len(templates),
            category_filter=category,
            difficulty_filter=difficulty,
        )

        return TemplatesResponse(
            templates=templates,
            categories=sorted(categories),
            total_count=len(templates),
        )

    except Exception as e:
        logfire.error(
            "Failed to retrieve problem templates",
            error=str(e),
            category=category,
            difficulty=difficulty,
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve problem templates"
        )


@router.get("/templates/{template_id}", response_model=ProblemTemplate)
async def get_template_by_id(template_id: str):
    """
    Get specific problem template by ID.

    Returns detailed information about a specific problem template including
    the complete problem definition and recommended configuration.
    """
    try:
        # Find template by ID
        template_data = None
        for template in PROBLEM_TEMPLATES:
            if template["id"] == template_id:
                template_data = template
                break

        if not template_data:
            raise HTTPException(
                status_code=404, detail=f"Template '{template_id}' not found"
            )

        logfire.info(
            "Problem template retrieved by ID",
            template_id=template_id,
            template_name=template_data["name"],
        )

        return ProblemTemplate(**template_data)

    except HTTPException:
        raise
    except Exception as e:
        logfire.error(
            "Failed to retrieve problem template", template_id=template_id, error=str(e)
        )
        raise HTTPException(
            status_code=500, detail="Failed to retrieve problem template"
        )


@router.get("/templates/categories")
async def get_template_categories():
    """
    Get available template categories.

    Returns a list of all available problem categories for filtering templates.
    """
    try:
        categories = list(set(t["category"] for t in PROBLEM_TEMPLATES))

        # Add category descriptions
        category_info = {
            "continuous": {
                "name": "Continuous",
                "description": "Problems with continuous real-valued variables",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["category"] == "continuous"]
                ),
            },
            "discrete": {
                "name": "Discrete",
                "description": "Problems with discrete integer variables",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["category"] == "discrete"]
                ),
            },
            "binary": {
                "name": "Binary",
                "description": "Problems with binary (0/1) variables",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["category"] == "binary"]
                ),
            },
            "permutation": {
                "name": "Permutation",
                "description": "Problems requiring permutation-based solutions",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["category"] == "permutation"]
                ),
            },
        }

        result = []
        for category in sorted(categories):
            if category in category_info:
                result.append(category_info[category])

        logfire.info("Template categories retrieved", categories_count=len(result))

        return {"categories": result, "total_categories": len(result)}

    except Exception as e:
        logfire.error("Failed to retrieve template categories", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve template categories"
        )


@router.get("/templates/difficulties")
async def get_template_difficulties():
    """
    Get available difficulty levels.

    Returns information about template difficulty levels for filtering.
    """
    try:
        difficulties = list(set(t["difficulty"] for t in PROBLEM_TEMPLATES))

        # Add difficulty descriptions
        difficulty_info = {
            "easy": {
                "name": "Easy",
                "description": "Simple problems good for learning and testing",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["difficulty"] == "easy"]
                ),
            },
            "medium": {
                "name": "Medium",
                "description": "Moderately complex problems with some challenges",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["difficulty"] == "medium"]
                ),
            },
            "hard": {
                "name": "Hard",
                "description": "Complex problems requiring advanced algorithms",
                "count": len(
                    [t for t in PROBLEM_TEMPLATES if t["difficulty"] == "hard"]
                ),
            },
        }

        result = []
        difficulty_order = ["easy", "medium", "hard"]
        for difficulty in difficulty_order:
            if difficulty in difficulties and difficulty in difficulty_info:
                result.append(difficulty_info[difficulty])

        logfire.info("Template difficulties retrieved", difficulties_count=len(result))

        return {"difficulties": result, "total_difficulties": len(result)}

    except Exception as e:
        logfire.error("Failed to retrieve template difficulties", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve template difficulties"
        )
