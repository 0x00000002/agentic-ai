"""Unit tests for the calculator tool."""
import pytest
from unittest.mock import patch
from src.tools.core.calculator_tool import calculate, ASTEVAL_AVAILABLE

# Test cases: (input_expression, expected_output_string)
VALID_TEST_CASES = [
    ("2+2", "4"),
    ("10 / 2", "5.0"), # asteval might return float
    ("3 * 5", "15"),
    ("(1 + 2) * 3", "9"),
    ("2**3", "8"),
    ("sqrt(16)", "4.0"), # asteval supports sqrt
    ("  1 +   1  ", "2"), # Check whitespace handling
]

INVALID_TEST_CASES = [
    # ("2++2", "Error evaluating expression"), # Removed: asteval handles this as 4
    ("a + b", "Error evaluating expression: name 'a' is not defined"), # Match actual msg
    ("import os", "Error evaluating expression: Import not supported"), # Match actual msg
    ("", "Error: Empty expression provided."), # Test empty input
]

class TestCalculatorTool:
    """Tests for the calculate function."""

    # @pytest.mark.skipif(not ASTEVAL_AVAILABLE, reason="asteval library not installed") # Removed skip
    @pytest.mark.parametrize("expression, expected", VALID_TEST_CASES)
    def test_calculate_valid_expressions(self, expression, expected):
        """Test calculate with valid mathematical expressions."""
        result = calculate(expression)
        assert result == expected

    # @pytest.mark.skipif(not ASTEVAL_AVAILABLE, reason="asteval library not installed") # Removed skip
    @pytest.mark.parametrize("expression, error_expected_message", INVALID_TEST_CASES)
    def test_calculate_invalid_expressions(self, expression, error_expected_message):
        """Test calculate with invalid or disallowed expressions."""
        result = calculate(expression)
        assert isinstance(result, str)
        # Use 'in' check for flexibility in exact error message wording
        assert error_expected_message in result

    def test_calculate_without_asteval(self):
        """Test calculate behavior when asteval is not available."""
        # Use patch to simulate asteval not being installed
        with patch('src.tools.core.calculator_tool.ASTEVAL_AVAILABLE', False):
            result = calculate("2+2")
            assert result == "Error: Calculator dependency (asteval) not installed." 