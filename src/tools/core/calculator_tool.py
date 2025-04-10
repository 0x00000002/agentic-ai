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
        # Create a new interpreter for each call to ensure clean state
        local_aeval = Interpreter()
        
        # Strip whitespace before evaluation
        clean_expression = expression.strip()
        if not clean_expression:
             return "Error: Empty expression provided."
             
        # Evaluate the cleaned expression using the local interpreter
        result = local_aeval(clean_expression)
        
        # Check if asteval returned None, often indicating an evaluation error
        if result is None:
            error_list = getattr(local_aeval, 'error', None)
            if error_list and isinstance(error_list, list) and len(error_list) > 0:
                # Get the message from the first EvalError object
                error_message = getattr(error_list[0], 'msg', 'Unknown evaluation error')
                # Format the error message similar to caught exceptions
                # We might need to parse the error_message further if it includes the type
                # For now, let's return a generic prefix + the message
                return f"Error evaluating expression: {error_message}"
            else:
                # Fallback if result is None but no specific error info is found
                return "Error evaluating expression: Unknown (result was None)"

        # Return result as string if evaluation was successful
        return str(result)
    except NameError:
        # This might be redundant if asteval handles it via None, but keep for safety
        return "Error evaluating expression: NameError"
    except SyntaxError:
        # This might be redundant if asteval handles it via None, but keep for safety
        return "Error evaluating expression: SyntaxError"
    except Exception as e:
        # Catch any other unexpected exceptions during the setup or str() conversion
        return f"Error evaluating expression: {type(e).__name__}" 