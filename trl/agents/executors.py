# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code Executors for Agentic AI Systems.

This module provides a flexible and extensible system for executing code in various
environments, with support for state tracking, function call monitoring, and
comparison against golden execution trails.
"""

import asyncio
import ast
import contextlib
import copy
import inspect
import io
import logging
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from ..import_utils import is_vllm_available

try:
    from e2b_code_interpreter import AsyncSandbox
    _e2b_available = True
except ImportError:
    _e2b_available = False

try:
    import modal
    from concurrent.futures import ThreadPoolExecutor
    _modal_available = True
except ImportError:
    _modal_available = False


logger = logging.getLogger(__name__)


default_system_prompt = "You can answer questions and solve problems. If running code helps, write it inside <code> </code>, and you will see the result. Example: To calculate 2 + 2, write <code> print(2 + 2) </code>."

default_environment_prompt = "This is a user-provided script containing tools that you can use to help complete tasks. It will be added to the sandbox environment so you can call its functions. If you are unsure how to use the available tools, you can use the `help()` function to inspect them."


def read_script(user_script_path: str) -> str:
    """
    Reads and returns the content of the provided script.

    Args:
        user_script_path (`str`):
            Path to the user-provided script.

    Returns:
        `str`:
            The content of the script.
    """
    return Path(user_script_path).read_text()


# Executor Interface and Base Classes

class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """
    Result of code execution.
    
    Args:
        status: Execution status.
        output: Output from the execution (stdout).
        error: Error message if execution failed.
        return_value: Return value from the execution (if any).
        execution_time: Time taken for execution in seconds.
        variables_changed: Dictionary of variables that were modified during execution.
    """
    status: ExecutionStatus
    output: str = ""
    error: str = ""
    return_value: Any = None
    execution_time: float = 0.0
    variables_changed: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionCall:
    """
    Represents a function call made during code execution.
    
    Args:
        function_name: Name of the called function.
        args: Positional arguments passed to the function.
        kwargs: Keyword arguments passed to the function.
        return_value: Value returned by the function.
        execution_time: Time taken for the function call.
        success: Whether the function call was successful.
        error: Error message if the function call failed.
    """
    function_name: str
    args: Tuple[Any, ...] = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    return_value: Any = None
    execution_time: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class ExecutionTrace:
    """
    Complete trace of code execution including function calls and state changes.
    
    Args:
        function_calls: List of function calls made during execution.
        state_snapshots: Snapshots of state at different points in execution.
        execution_result: Final result of the execution.
    """
    function_calls: List[FunctionCall] = field(default_factory=list)
    state_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    execution_result: Optional[ExecutionResult] = None


class ExecutorProtocol(Protocol):
    """
    Protocol defining the interface that all executors must implement.
    
    This protocol ensures that all executor implementations provide consistent
    methods for code execution, regardless of their underlying implementation.
    """
    
    def execute(self, code_blocks: List[str]) -> List[str]:
        """
        Execute a list of code blocks and return their outputs.
        
        Args:
            code_blocks: List of code strings to execute.
            
        Returns:
            List of execution results as strings.
        """
        ...


class BaseExecutor(ABC):
    """
    Base class for all code executors.
    
    Provides common functionality and interface for code execution with
    proper error handling, logging, and extensibility points.
    """
    
    def __init__(self, timeout: float = 30.0, max_output_length: int = 10000):
        """
        Initialize the executor.
        
        Args:
            timeout: Maximum time to allow for code execution.
            max_output_length: Maximum length of output to capture.
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self._execution_count = 0
        
    @abstractmethod
    def _execute_single(self, code: str) -> ExecutionResult:
        """
        Execute a single code block.
        
        Args:
            code: Code string to execute.
            
        Returns:
            ExecutionResult object with execution details.
        """
        raise NotImplementedError
        
    def execute(self, code_blocks: List[str]) -> List[str]:
        """
        Execute multiple code blocks and return string results.
        
        This method provides the protocol-compatible interface while
        internally using the richer ExecutionResult objects.
        
        Args:
            code_blocks: List of code strings to execute.
            
        Returns:
            List of execution results as strings.
        """
        results = []
        for code in code_blocks:
            execution_result = self._execute_single(code)
            if execution_result.status == ExecutionStatus.SUCCESS:
                results.append(execution_result.output)
            else:
                results.append(f"Error: {execution_result.error}")
        return results
        
    def execute_with_results(self, code_blocks: List[str]) -> List[ExecutionResult]:
        """
        Execute code blocks and return detailed ExecutionResult objects.
        
        Args:
            code_blocks: List of code strings to execute.
            
        Returns:
            List of ExecutionResult objects with detailed execution information.
        """
        return [self._execute_single(code) for code in code_blocks]
        
    def reset(self) -> None:
        """
        Reset the executor state.
        
        Subclasses should override this method to reset any persistent state.
        """
        self._execution_count = 0


# Python REPL Implementation

class PythonREPL:
    """
    A Python REPL (Read-Eval-Print Loop) for executing Python code.
    
    This class provides a sandboxed environment for executing Python code
    with proper output capture and error handling.
    """
    
    def __init__(self):
        """Initialize the Python REPL with a clean namespace."""
        self.namespace = {"__builtins__": __builtins__}
        
    def run(self, code: str) -> str:
        """
        Execute Python code and return the output.
        
        Args:
            code: Python code to execute.
            
        Returns:
            Output from the code execution.
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Try to evaluate as an expression first
            try:
                result = eval(code, self.namespace)
                if result is not None:
                    print(result)
            except SyntaxError:
                # If it's not an expression, execute as statements
                exec(code, self.namespace)
            
            output = captured_output.getvalue()
            return output.strip() if output else ""
            
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout


# Basic Local Executor

class LocalExecutor(BaseExecutor):
    """
    Local code executor using Python REPL.
    
    This executor runs code in the local Python environment using a REPL
    instance for each execution session.
    """
    
    def __init__(self, timeout: float = 30.0, max_output_length: int = 10000):
        """
        Initialize the local executor.
        
        Args:
            timeout: Maximum time to allow for code execution.
            max_output_length: Maximum length of output to capture.
        """
        super().__init__(timeout, max_output_length)
        self.repl = PythonREPL()
        
    def _execute_single(self, code: str) -> ExecutionResult:
        """
        Execute a single code block using the Python REPL.
        
        Args:
            code: Code string to execute.
            
        Returns:
            ExecutionResult object with execution details.
        """
        start_time = time.time()
        self._execution_count += 1
        
        try:
            output = self.repl.run(code)
            execution_time = time.time() - start_time
            
            # Truncate output if it's too long
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "... (truncated)"
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}"
            
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=error_message,
                execution_time=execution_time
            )
    
    def reset(self) -> None:
        """Reset the executor by creating a new REPL instance."""
        super().reset()
        self.repl = PythonREPL()


# Function Call Tracker

class FunctionCallTracker:
    """
    Tracks function calls made during code execution.
    
    This class can wrap functions to monitor their calls, arguments,
    return values, and execution times.
    """
    
    def __init__(self):
        """Initialize the function call tracker."""
        self.calls: List[FunctionCall] = []
        self.wrapped_functions: Dict[str, Callable] = {}
        
    def wrap_function(self, name: str, func: Callable) -> Callable:
        """
        Wrap a function to track its calls.
        
        Args:
            name: Name of the function for tracking.
            func: Function to wrap.
            
        Returns:
            Wrapped function that tracks calls.
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                call = FunctionCall(
                    function_name=name,
                    args=args,
                    kwargs=kwargs,
                    return_value=result,
                    execution_time=execution_time,
                    success=True
                )
                self.calls.append(call)
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                call = FunctionCall(
                    function_name=name,
                    args=args,
                    kwargs=kwargs,
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
                self.calls.append(call)
                raise
                
        self.wrapped_functions[name] = wrapper
        return wrapper
    
    def get_calls(self) -> List[FunctionCall]:
        """Get all tracked function calls."""
        return self.calls.copy()
    
    def clear_calls(self) -> None:
        """Clear all tracked function calls."""
        self.calls.clear()


# Stateful Executor with Function Tracking

class StatefulLocalExecutor(BaseExecutor):
    """
    Stateful local code executor with function call tracking and state management.
    
    This executor maintains persistent state between code executions and tracks
    all function calls made during execution. It's designed for scenarios where
    you need to monitor how models interact with provided functions and how
    state variables change over time.
    """
    
    def __init__(
        self, 
        functions: Optional[Dict[str, Callable]] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        max_output_length: int = 10000
    ):
        """
        Initialize the stateful executor.
        
        Args:
            functions: Dictionary of functions to make available in the execution environment.
            initial_state: Initial state variables to set in the execution environment.
            timeout: Maximum time to allow for code execution.
            max_output_length: Maximum length of output to capture.
        """
        super().__init__(timeout, max_output_length)
        
        # Initialize state and function tracking
        self.state = initial_state.copy() if initial_state else {}
        self.function_tracker = FunctionCallTracker()
        self.execution_traces: List[ExecutionTrace] = []
        
        # Set up the execution namespace
        self.namespace = {"__builtins__": __builtins__}
        self.namespace.update(self.state)
        
        # Wrap and add provided functions
        if functions:
            for name, func in functions.items():
                wrapped_func = self.function_tracker.wrap_function(name, func)
                self.namespace[name] = wrapped_func
                
        logger.info(f"Initialized StatefulLocalExecutor with {len(functions or {})} functions and {len(self.state)} state variables")
    
    def _execute_single(self, code: str) -> ExecutionResult:
        """
        Execute a single code block with state and function call tracking.
        
        Args:
            code: Code string to execute.
            
        Returns:
            ExecutionResult with detailed execution information.
        """
        start_time = time.time()
        self._execution_count += 1
        
        # Take a snapshot of the state before execution
        state_before = copy.deepcopy(self.state)
        calls_before_count = len(self.function_tracker.calls)
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Execute the code
            try:
                # Try to evaluate as an expression first
                result = eval(code, self.namespace)
                if result is not None:
                    print(result)
            except SyntaxError:
                # If it's not an expression, execute as statements
                exec(code, self.namespace)
            
            # Update our state with any changes made to namespace
            self._update_state_from_namespace()
            
            execution_time = time.time() - start_time
            output = captured_output.getvalue()
            
            # Truncate output if necessary
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length] + "... (truncated)"
            
            # Detect state changes
            variables_changed = {}
            for key, value in self.state.items():
                if key not in state_before or state_before[key] != value:
                    variables_changed[key] = value
            
            # Get function calls made during this execution
            new_calls = self.function_tracker.calls[calls_before_count:]
            
            # Create execution result
            result = ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output.strip(),
                execution_time=execution_time,
                variables_changed=variables_changed
            )
            
            # Create and store execution trace
            trace = ExecutionTrace(
                function_calls=new_calls.copy(),
                state_snapshots=[state_before, copy.deepcopy(self.state)],
                execution_result=result
            )
            self.execution_traces.append(trace)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"{type(e).__name__}: {str(e)}"
            
            result = ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=error_message,
                execution_time=execution_time
            )
            
            # Still create a trace even for failed executions
            new_calls = self.function_tracker.calls[calls_before_count:]
            trace = ExecutionTrace(
                function_calls=new_calls.copy(),
                state_snapshots=[state_before, copy.deepcopy(self.state)],
                execution_result=result
            )
            self.execution_traces.append(trace)
            
            return result
            
        finally:
            sys.stdout = old_stdout
    
    def _update_state_from_namespace(self) -> None:
        """Update the state dictionary with changes from the execution namespace."""
        for key, value in self.namespace.items():
            if key not in ["__builtins__"] and not key.startswith("_"):
                # Only track serializable state variables
                try:
                    copy.deepcopy(value)  # Test if it's serializable
                    self.state[key] = value
                except (TypeError, AttributeError):
                    # Skip non-serializable objects like functions, modules, etc.
                    pass
    
    def get_function_calls(self) -> List[FunctionCall]:
        """Get all function calls made during execution."""
        return self.function_tracker.get_calls()
    
    def get_execution_traces(self) -> List[ExecutionTrace]:
        """Get all execution traces."""
        return self.execution_traces.copy()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current state of all variables."""
        return self.state.copy()
    
    def compare_with_golden_trail(self, golden_calls: List[FunctionCall]) -> Dict[str, Any]:
        """
        Compare the actual function calls with a golden execution trail.
        
        Args:
            golden_calls: List of expected function calls.
            
        Returns:
            Dictionary containing comparison results.
        """
        actual_calls = self.get_function_calls()
        
        comparison = {
            "total_actual_calls": len(actual_calls),
            "total_expected_calls": len(golden_calls),
            "matches": [],
            "missing_calls": [],
            "extra_calls": [],
            "incorrect_calls": []
        }
        
        # Simple comparison - can be made more sophisticated
        min_len = min(len(actual_calls), len(golden_calls))
        
        for i in range(min_len):
            actual = actual_calls[i]
            expected = golden_calls[i]
            
            if (actual.function_name == expected.function_name and 
                actual.args == expected.args and 
                actual.kwargs == expected.kwargs):
                comparison["matches"].append({
                    "index": i,
                    "function": actual.function_name,
                    "matched": True
                })
            else:
                comparison["incorrect_calls"].append({
                    "index": i,
                    "actual": actual,
                    "expected": expected
                })
        
        # Handle extra/missing calls
        if len(actual_calls) > len(golden_calls):
            comparison["extra_calls"] = actual_calls[len(golden_calls):]
        elif len(golden_calls) > len(actual_calls):
            comparison["missing_calls"] = golden_calls[len(actual_calls):]
        
        return comparison
    
    def reset(self) -> None:
        """Reset the executor to its initial state."""
        super().reset()
        self.function_tracker.clear_calls()
        self.execution_traces.clear()
        # Reset namespace but keep the wrapped functions
        self.namespace = {"__builtins__": __builtins__}
        self.namespace.update(self.function_tracker.wrapped_functions)
        self.state.clear()


# Legacy compatibility
LocalExecuter = LocalExecutor  # Maintain backward compatibility


# Executor Registry

class ExecutorRegistry:
    """
    Registry for executor types, enabling easy discovery and instantiation.
    """
    
    _executors: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, executor_class: type) -> None:
        """Register an executor class with a given name."""
        if not issubclass(executor_class, BaseExecutor):
            raise ValueError(f"Executor class {executor_class} must inherit from BaseExecutor")
        
        cls._executors[name] = executor_class
        logger.info(f"Registered executor: {name} -> {executor_class.__name__}")
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseExecutor:
        """Create an executor instance by name."""
        if name not in cls._executors:
            raise ValueError(f"Executor '{name}' not found. Available: {list(cls._executors.keys())}")
        
        executor_class = cls._executors[name]
        return executor_class(**kwargs)
    
    @classmethod
    def list_executors(cls) -> List[str]:
        """List all registered executor names."""
        return list(cls._executors.keys())


# Register built-in executors
ExecutorRegistry.register("local", LocalExecutor)
ExecutorRegistry.register("stateful_local", StatefulLocalExecutor)


# Convenience functions
def create_executor(name: str, **kwargs) -> BaseExecutor:
    """Create an executor by name."""
    return ExecutorRegistry.create(name, **kwargs)


def list_available_executors() -> List[str]:
    """List all available executor types."""
    return ExecutorRegistry.list_executors()


def register_executor(name: str, executor_class: type) -> None:
    """Register a custom executor."""
    ExecutorRegistry.register(name, executor_class)


# Usage Examples and Documentation

def create_simple_task_example():
    """
    Example showing how to create a simple task with function tracking.
    
    This demonstrates a realistic scenario where an AI model is given functions
    to manage a person's financial account and we want to track what it does.
    """
    # Define the functions the model can use
    def update_salary(person, amount):
        """Update a person's salary by the given amount."""
        person['salary'] += amount
        return person['salary']

    def calculate_tax(salary, rate=0.2):
        """Calculate tax based on salary and rate."""
        return salary * rate

    def pay_tax(person, tax_amount):
        """Deduct tax from person's account balance."""
        person['balance'] -= tax_amount
        return person['balance']

    # Initial state
    initial_state = {
        'person': {
            'name': 'Alice',
            'salary': 50000,
            'balance': 10000
        }
    }

    # Available functions
    functions = {
        'update_salary': update_salary,
        'calculate_tax': calculate_tax,
        'pay_tax': pay_tax
    }

    # Create the stateful executor
    executor = create_executor(
        'stateful_local',
        functions=functions,
        initial_state=initial_state
    )

    # Example code that a model might generate
    model_generated_code = [
        "print('Initial state:')",
        "print(f'Salary: ${person[\"salary\"]}, Balance: ${person[\"balance\"]}')",
        "",
        "# Give Alice a raise",
        "new_salary = update_salary(person, 10000)",
        "print(f'New salary: ${new_salary}')",
        "",
        "# Calculate and pay taxes",
        "tax_owed = calculate_tax(person['salary'])",
        "print(f'Tax owed: ${tax_owed}')",
        "remaining_balance = pay_tax(person, tax_owed)",
        "print(f'Remaining balance: ${remaining_balance}')"
    ]

    # Execute the code
    results = executor.execute(model_generated_code)
    
    # Analyze what happened
    print("=== Execution Results ===")
    for i, result in enumerate(results):
        if result.strip():  # Only print non-empty results
            print(f"Step {i+1}: {result}")
    
    print("\n=== Function Calls Made ===")
    for call in executor.get_function_calls():
        print(f"- {call.function_name}({call.args}, {call.kwargs}) -> {call.return_value}")
    
    print("\n=== Final State ===")
    final_state = executor.get_current_state()
    print(f"Person: {final_state.get('person', {})}")
    
    # Compare with expected behavior
    expected_calls = [
        FunctionCall(
            function_name='update_salary',
            args=({'name': 'Alice', 'salary': 60000, 'balance': 10000}, 10000),
            success=True
        ),
        FunctionCall(
            function_name='calculate_tax',
            args=(60000,),
            success=True
        ),
        FunctionCall(
            function_name='pay_tax',
            args=({'name': 'Alice', 'salary': 60000, 'balance': 10000}, 12000),
            success=True
        )
    ]
    
    comparison = executor.compare_with_golden_trail(expected_calls)
    print("\n=== Golden Trail Comparison ===")
    print(f"Matches: {len(comparison['matches'])}/{comparison['total_expected_calls']}")
    
    return executor


if __name__ == "__main__":
    # Run the example
    print("Running StatefulLocalExecutor example...")
    executor = create_simple_task_example()