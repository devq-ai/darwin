"""
Darwin Core Problem Definition Module

This module provides classes for defining optimization problems
including variables, constraints, and objectives.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ObjectiveType(str, Enum):
    """Optimization objective types."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MULTI_OBJECTIVE = "multi_objective"


class VariableType(str, Enum):
    """Variable types for optimization problems."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


@dataclass
class Variable:
    """Optimization variable definition."""

    name: str
    type: VariableType
    bounds: Tuple[float, float]
    gene_space: Optional[List[Union[float, int, str]]] = None
    encoding: str = "real"

    def __post_init__(self):
        """Validate variable definition after initialization."""
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError(
                f"Invalid bounds for variable {self.name}: lower bound must be less than upper bound"
            )

        if self.type in [VariableType.DISCRETE, VariableType.CATEGORICAL]:
            if self.gene_space is None or len(self.gene_space) == 0:
                raise ValueError(
                    f"Gene space is required for {self.type} variable {self.name}"
                )


@dataclass
class Constraint:
    """Optimization constraint definition."""

    name: str
    type: str
    expression: str
    tolerance: float = 1e-6

    def __post_init__(self):
        """Validate constraint definition after initialization."""
        valid_types = ["equality", "inequality", "<=", ">=", "=="]
        if self.type not in valid_types:
            raise ValueError(
                f"Invalid constraint type: {self.type}. Must be one of {valid_types}"
            )


class OptimizationProblem:
    """Optimization problem definition."""

    def __init__(
        self,
        name: str,
        description: str = "",
        objective_type: ObjectiveType = ObjectiveType.MINIMIZE,
        variables: Optional[List[Variable]] = None,
        constraints: Optional[List[Constraint]] = None,
        fitness_function: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize optimization problem.

        Args:
            name: Problem name
            description: Problem description
            objective_type: Type of optimization (minimize, maximize, multi-objective)
            variables: List of problem variables
            constraints: List of problem constraints
            fitness_function: Custom fitness function code
            metadata: Additional problem metadata
        """
        self.name = name
        self.description = description
        self.objective_type = objective_type
        self.variables = variables or []
        self.constraints = constraints or []
        self.fitness_function = fitness_function
        self.metadata = metadata or {}

        # Validate the problem definition
        self._validate()

    def _validate(self) -> None:
        """Validate the problem definition."""
        if not self.name:
            raise ValueError("Problem name is required")

        if not self.variables:
            raise ValueError("At least one variable is required")

        # Check for duplicate variable names
        variable_names = [var.name for var in self.variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError("Variable names must be unique")

        # Check for duplicate constraint names
        if self.constraints:
            constraint_names = [const.name for const in self.constraints]
            if len(constraint_names) != len(set(constraint_names)):
                raise ValueError("Constraint names must be unique")

    def add_variable(self, variable: Variable) -> None:
        """Add a variable to the problem."""
        if any(var.name == variable.name for var in self.variables):
            raise ValueError(f"Variable with name '{variable.name}' already exists")
        self.variables.append(variable)

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the problem."""
        if any(const.name == constraint.name for const in self.constraints):
            raise ValueError(f"Constraint with name '{constraint.name}' already exists")
        self.constraints.append(constraint)

    def get_variable_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for all variables as a list of tuples."""
        return [var.bounds for var in self.variables]

    def get_variable_names(self) -> List[str]:
        """Get names of all variables."""
        return [var.name for var in self.variables]

    def get_constraint_names(self) -> List[str]:
        """Get names of all constraints."""
        return [const.name for const in self.constraints]

    def evaluate_fitness(self, solution: List[float]) -> float:
        """
        Evaluate fitness for a given solution.

        Args:
            solution: List of variable values

        Returns:
            Fitness value
        """
        if len(solution) != len(self.variables):
            raise ValueError("Solution length must match number of variables")

        # Simple placeholder fitness function
        # In practice, this would use the custom fitness_function or built-in functions
        if self.fitness_function:
            # This would execute custom fitness function
            # For now, just return a simple sphere function
            return sum(x**2 for x in solution)
        else:
            # Default sphere function
            return sum(x**2 for x in solution)

    def is_solution_feasible(self, solution: List[float]) -> bool:
        """
        Check if a solution satisfies all constraints.

        Args:
            solution: List of variable values

        Returns:
            True if solution is feasible, False otherwise
        """
        if not self.constraints:
            return True

        # Placeholder constraint checking
        # In practice, this would evaluate the constraint expressions
        for constraint in self.constraints:
            # This would evaluate the constraint expression with the solution
            # For now, assume all solutions are feasible
            pass

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert problem to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "objective_type": self.objective_type.value,
            "variables": [
                {
                    "name": var.name,
                    "type": var.type.value,
                    "bounds": var.bounds,
                    "gene_space": var.gene_space,
                    "encoding": var.encoding,
                }
                for var in self.variables
            ],
            "constraints": [
                {
                    "name": const.name,
                    "type": const.type,
                    "expression": const.expression,
                    "tolerance": const.tolerance,
                }
                for const in self.constraints
            ],
            "fitness_function": self.fitness_function,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationProblem":
        """Create problem from dictionary representation."""
        variables = [
            Variable(
                name=var_data["name"],
                type=VariableType(var_data["type"]),
                bounds=tuple(var_data["bounds"]),
                gene_space=var_data.get("gene_space"),
                encoding=var_data.get("encoding", "real"),
            )
            for var_data in data.get("variables", [])
        ]

        constraints = [
            Constraint(
                name=const_data["name"],
                type=const_data["type"],
                expression=const_data["expression"],
                tolerance=const_data.get("tolerance", 1e-6),
            )
            for const_data in data.get("constraints", [])
        ]

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            objective_type=ObjectiveType(data.get("objective_type", "minimize")),
            variables=variables,
            constraints=constraints,
            fitness_function=data.get("fitness_function"),
            metadata=data.get("metadata", {}),
        )
