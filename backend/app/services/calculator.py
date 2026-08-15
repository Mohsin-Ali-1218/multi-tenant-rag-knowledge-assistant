from asteval import Interpreter

_aeval = Interpreter()

def safe_calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression without using standard eval()."""
    expression = expression.strip()
    if not expression:
        return "Error: Empty expression"
    try:
        _aeval.error = []
        result = _aeval(expression)
        if _aeval.error:
            errs = [err.get_error()[1] for err in _aeval.error if hasattr(err, "get_error")]
            return f"Error evaluating '{expression}': {', '.join(errs) if errs else 'Evaluation failed'}"
        return str(result)
    except Exception as e:
        return f"Error evaluating '{expression}': {str(e)}"
