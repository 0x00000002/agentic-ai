#!/usr/bin/env python
"""
Simple script to test tool execution directly.
"""
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the package
sys.path.insert(0, str(Path(__file__).parent))

from src.tools.core.calculator_tool import calculate
from src.tools.core.datetime_tool import get_datetime
from src.tools.core.dummy_tool import dummy_tool_function
from src.tools.models import ToolDefinition, ToolResult

def test_direct_execution():
    """Test executing tools directly."""
    # Test calculator tool
    try:
        print("Directly calling calculate('2 + 2'):")
        result = calculate("2 + 2")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test datetime tool
    try:
        print("\nDirectly calling get_datetime():")
        result = get_datetime()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test dummy tool
    try:
        print("\nDirectly calling dummy_tool_function('test'):")
        result = dummy_tool_function("test")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

def test_through_tooldef():
    """Test executing tools through ToolDefinition."""
    # Create tool definitions
    calc_tool = ToolDefinition(
        name="calculator",
        description="Calculator tool",
        parameters_schema={},
        module_path="src.tools.core.calculator_tool",
        function_name="calculate",
        function=calculate
    )
    
    datetime_tool = ToolDefinition(
        name="get_current_datetime",
        description="Datetime tool",
        parameters_schema={},
        module_path="src.tools.core.datetime_tool",
        function_name="get_datetime",
        function=get_datetime
    )
    
    dummy_tool = ToolDefinition(
        name="dummy_tool",
        description="Dummy tool",
        parameters_schema={},
        module_path="src.tools.core.dummy_tool",
        function_name="dummy_tool_function",
        function=dummy_tool_function
    )
    
    # Test calculator tool
    try:
        print("\nCalling calc_tool.function(expression='2 + 2'):")
        result = calc_tool.function(expression="2 + 2")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test datetime tool 
    try:
        print("\nCalling datetime_tool.function():")
        result = datetime_tool.function()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test dummy tool
    try:
        print("\nCalling dummy_tool.function(query='test'):")
        result = dummy_tool.function(query="test")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with extra 'name' parameter
    try:
        print("\nCalling calc_tool.function(expression='2 + 2', name='calculator'):")
        result = calc_tool.function(expression="2 + 2", name="calculator")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("=== Testing Direct Execution ===")
    test_direct_execution()
    
    print("\n=== Testing Through ToolDefinition ===")
    test_through_tooldef() 