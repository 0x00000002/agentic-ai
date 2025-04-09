"""Core tool for performing mathematical calculations safely."""
try:
    from asteval import Interpreter
    aeval = Interpreter()
    ASTEVAL_AVAILABLE = True
except ImportError:
    ASTEVAL_AVAILABLE = False
    # Fallback or raise error - let's log a warning for now
    # print("Warning: asteval library not found. Calculator tool will be limited or unavailable.")
    # Alternative: Use a simpler, regex-based approach for basic math? Or just disable.

def calculate(expression: str) -> str:
    """Evaluates a mathematical expression string safely using asteval.
    
    Args:
        expression: The mathematical expression string (e.g., '2 * (3 + 5)').
        
    Returns:
        The result of the calculation as a string, or an error message.
    """
    if not ASTEVAL_AVAILABLE:
        return "Error: Calculator dependency (asteval) not installed."
        
    try:
        # Evaluate the expression
        result = aeval(expression)
        # Return result as string
        return str(result)
    except Exception as e:
        # Return specific error from asteval if possible, otherwise generic error
        return f"Error evaluating expression: {str(e)}" 