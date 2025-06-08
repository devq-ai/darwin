#!/usr/bin/env python3
"""
Task Validation Script for Darwin Project

This script validates that tasks meet their requirements and dependencies
before allowing progression to dependent tasks. It enforces:

1. All tests must pass for a task to be considered complete
2. Dependencies must be satisfied before starting a task
3. Task status updates are validated against test results

Usage:
    python scripts/validate_task.py --task-id 1.3 --action start
    python scripts/validate_task.py --task-id 1.2 --action complete
    python scripts/validate_task.py --list-ready
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TaskValidationError(Exception):
    """Custom exception for task validation errors."""

    pass


class TaskValidator:
    """Main task validation class."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tasks_file = project_root / ".taskmaster" / "tasks" / "tasks.json"
        self.reports_dir = project_root / ".taskmaster" / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_tasks(self) -> Dict:
        """Load tasks from tasks.json file."""
        if not self.tasks_file.exists():
            raise FileNotFoundError(f"Tasks file not found: {self.tasks_file}")

        try:
            with open(self.tasks_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise TaskValidationError(f"Invalid JSON in tasks file: {e}")

    def save_tasks(self, tasks_data: Dict) -> None:
        """Save tasks back to tasks.json file."""
        try:
            with open(self.tasks_file, "w") as f:
                json.dump(tasks_data, f, indent=2)
        except Exception as e:
            raise TaskValidationError(f"Failed to save tasks file: {e}")

    def get_task_by_id(self, task_id: str) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Get task by ID from tasks.json.
        Returns (task_dict, task_type) where task_type is 'main' or 'subtask'
        """
        tasks = self.load_tasks()

        # Handle main tasks
        for task in tasks.get("tasks", []):
            if str(task["id"]) == str(task_id):
                return task, "main"

            # Handle subtasks
            for subtask in task.get("subtasks", []):
                if str(subtask["id"]) == str(task_id):
                    return subtask, "subtask"

        return None, None

    def run_task_tests(self, task_id: str) -> Tuple[bool, str]:
        """
        Run tests for a specific task.
        Returns (success, output)
        """
        # Convert task_id to test file name
        test_file = self.project_root / "tests" / f"test_task_{task_id}.py"

        if not test_file.exists():
            logger.warning(f"No test file found for task {task_id}: {test_file}")
            return False, f"No test file found: {test_file}"

        logger.info(f"Running tests for task {task_id}...")

        try:
            # Check if Poetry is available and use it
            poetry_result = subprocess.run(
                ["poetry", "--version"],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if poetry_result.returncode == 0:
                # Use Poetry to run tests
                cmd = ["poetry", "run", "pytest", str(test_file), "-v", "--tb=short"]
            else:
                # Fall back to system Python
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(test_file),
                    "-v",
                    "--tb=short",
                ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=self.project_root
            )

            # Create test report
            self._create_test_report(
                task_id, result.returncode == 0, result.stdout, result.stderr
            )

            return result.returncode == 0, result.stdout + result.stderr

        except Exception as e:
            error_msg = f"Failed to run tests for task {task_id}: {e}"
            logger.error(error_msg)
            return False, error_msg

    def _create_test_report(
        self, task_id: str, passed: bool, stdout: str, stderr: str
    ) -> None:
        """Create a test report for a task."""
        import datetime

        report = {
            "task_id": task_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "passed": passed,
            "stdout": stdout,
            "stderr": stderr,
        }

        report_file = self.reports_dir / f"task_{task_id}_test_report.json"

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Test report saved: {report_file}")

    def validate_dependencies(self, task_id: str) -> Tuple[bool, List[str]]:
        """
        Validate that all dependencies are completed with passing tests.
        Returns (all_satisfied, missing_dependencies)
        """
        task, _ = self.get_task_by_id(task_id)
        if not task:
            return False, [f"Task {task_id} not found"]

        dependencies = task.get("dependencies", [])
        missing = []

        for dep_id in dependencies:
            dep_task, _ = self.get_task_by_id(dep_id)
            if not dep_task:
                missing.append(f"Dependency {dep_id} not found")
                continue

            # Check if dependency is completed
            if dep_task.get("status") != "done":
                missing.append(
                    f"Dependency {dep_id} not completed (status: {dep_task.get('status')})"
                )
                continue

            # Verify tests passed for dependency
            tests_passed, _ = self.run_task_tests(str(dep_id))
            if not tests_passed:
                missing.append(f"Dependency {dep_id} tests failed")

        return len(missing) == 0, missing

    def can_start_task(self, task_id: str) -> Tuple[bool, List[str]]:
        """
        Check if a task can be started based on dependencies.
        Returns (can_start, reasons)
        """
        task, _ = self.get_task_by_id(task_id)
        if not task:
            return False, [f"Task {task_id} not found"]

        # Check if task is already completed
        if task.get("status") == "done":
            return False, [f"Task {task_id} is already completed"]

        # Check if task is already in progress
        if task.get("status") == "in-progress":
            return True, [f"Task {task_id} is already in progress"]

        # Validate dependencies
        deps_satisfied, missing_deps = self.validate_dependencies(task_id)
        if not deps_satisfied:
            return False, missing_deps

        return True, ["All dependencies satisfied"]

    def validate_task_completion(self, task_id: str) -> Tuple[bool, List[str]]:
        """
        Validate that a task can be marked as complete.
        Returns (can_complete, reasons)
        """
        task, _ = self.get_task_by_id(task_id)
        if not task:
            return False, [f"Task {task_id} not found"]

        # Check if task is already completed
        if task.get("status") == "done":
            return True, [f"Task {task_id} is already completed"]

        # Run tests to validate completion
        tests_passed, test_output = self.run_task_tests(task_id)
        if not tests_passed:
            return False, [f"Task {task_id} tests failed", test_output]

        return True, ["All tests passed"]

    def update_task_status(
        self, task_id: str, new_status: str, force: bool = False
    ) -> bool:
        """
        Update task status with validation.
        Returns True if successful, False otherwise.
        """
        if new_status not in ["pending", "in-progress", "done"]:
            logger.error(f"Invalid status: {new_status}")
            return False

        # Validate the status change
        if new_status == "in-progress":
            can_start, reasons = self.can_start_task(task_id)
            if not can_start and not force:
                logger.error(f"Cannot start task {task_id}: {'; '.join(reasons)}")
                return False

        elif new_status == "done":
            can_complete, reasons = self.validate_task_completion(task_id)
            if not can_complete and not force:
                logger.error(f"Cannot complete task {task_id}: {'; '.join(reasons)}")
                return False

        # Update the task status
        tasks_data = self.load_tasks()

        # Find and update the task
        for task in tasks_data.get("tasks", []):
            if str(task["id"]) == str(task_id):
                task["status"] = new_status
                self.save_tasks(tasks_data)
                logger.info(f"Updated task {task_id} status to {new_status}")
                return True

            # Check subtasks
            for subtask in task.get("subtasks", []):
                if str(subtask["id"]) == str(task_id):
                    subtask["status"] = new_status
                    self.save_tasks(tasks_data)
                    logger.info(f"Updated subtask {task_id} status to {new_status}")
                    return True

        logger.error(f"Task {task_id} not found for status update")
        return False

    def list_ready_tasks(self) -> List[Dict]:
        """List all tasks that are ready to be started."""
        tasks_data = self.load_tasks()
        ready_tasks = []

        for task in tasks_data.get("tasks", []):
            task_id = str(task["id"])
            if task.get("status") == "pending":
                can_start, _ = self.can_start_task(task_id)
                if can_start:
                    ready_tasks.append(
                        {
                            "id": task_id,
                            "title": task.get("title", ""),
                            "priority": task.get("priority", "medium"),
                            "type": "main",
                        }
                    )

            # Check subtasks
            for subtask in task.get("subtasks", []):
                subtask_id = str(subtask["id"])
                if subtask.get("status") == "pending":
                    can_start, _ = self.can_start_task(subtask_id)
                    if can_start:
                        ready_tasks.append(
                            {
                                "id": subtask_id,
                                "title": subtask.get("title", ""),
                                "priority": task.get("priority", "medium"),
                                "type": "subtask",
                            }
                        )

        return ready_tasks

    def get_task_status_summary(self) -> Dict:
        """Get a summary of all task statuses."""
        tasks_data = self.load_tasks()
        summary = {"pending": 0, "in-progress": 0, "done": 0, "total": 0}

        for task in tasks_data.get("tasks", []):
            summary["total"] += 1
            status = task.get("status", "pending")
            summary[status] = summary.get(status, 0) + 1

            # Count subtasks
            for subtask in task.get("subtasks", []):
                summary["total"] += 1
                subtask_status = subtask.get("status", "pending")
                summary[subtask_status] = summary.get(subtask_status, 0) + 1

        return summary


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Task Validation Script")
    parser.add_argument("--task-id", help="Task ID to validate")
    parser.add_argument(
        "--action", choices=["start", "complete", "test"], help="Action to perform"
    )
    parser.add_argument(
        "--list-ready", action="store_true", help="List tasks ready to start"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show task status summary"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force action without validation"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find project root
    current_dir = Path(__file__).parent.parent
    if not (current_dir / ".taskmaster").exists():
        logger.error("Not in a taskmaster project directory")
        sys.exit(1)

    validator = TaskValidator(current_dir)

    try:
        if args.list_ready:
            ready_tasks = validator.list_ready_tasks()
            if ready_tasks:
                print("\n🚀 Ready to Start:")
                for task in ready_tasks:
                    print(
                        f"  {task['id']}: {task['title']} [{task['type']}, {task['priority']} priority]"
                    )
            else:
                print("\n⏳ No tasks ready to start")

        elif args.status:
            summary = validator.get_task_status_summary()
            print("\n📊 Task Status Summary:")
            print(f"  Total: {summary['total']}")
            print(f"  Pending: {summary['pending']}")
            print(f"  In Progress: {summary['in-progress']}")
            print(f"  Done: {summary['done']}")

            completion_rate = (
                (summary["done"] / summary["total"] * 100)
                if summary["total"] > 0
                else 0
            )
            print(f"  Completion: {completion_rate:.1f}%")

        elif args.task_id and args.action:
            if args.action == "start":
                success = validator.update_task_status(
                    args.task_id, "in-progress", args.force
                )
                if success:
                    print(f"✅ Task {args.task_id} started")
                else:
                    print(f"❌ Failed to start task {args.task_id}")
                    sys.exit(1)

            elif args.action == "complete":
                success = validator.update_task_status(args.task_id, "done", args.force)
                if success:
                    print(f"✅ Task {args.task_id} completed")
                else:
                    print(f"❌ Failed to complete task {args.task_id}")
                    sys.exit(1)

            elif args.action == "test":
                tests_passed, output = validator.run_task_tests(args.task_id)
                if tests_passed:
                    print(f"✅ Tests passed for task {args.task_id}")
                else:
                    print(f"❌ Tests failed for task {args.task_id}")
                    if args.verbose:
                        print(output)
                    sys.exit(1)

        else:
            parser.print_help()

    except TaskValidationError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
