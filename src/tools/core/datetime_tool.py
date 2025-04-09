"""Core tool for getting the current date and time."""
import datetime

def get_datetime() -> str:
    """Returns the current date and time in ISO 8601 format."""
    return datetime.datetime.now().isoformat() 