# Contributing to Agentic-AI

Thank you for your interest in contributing to Agentic-AI!

## Development Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/0x00000002/agentic-ai.git
    cd agentic-ai
    ```
2.  **Create Environment:** Use Conda (recommended) or another virtual environment manager.
    ```bash
    # Using Conda
    conda env create -f environment.yml
    conda activate agentic-ai
    ```
3.  **Install Dependencies:** Install the project in editable mode.
    ```bash
    pip install -e .
    # Install dev dependencies (if any, e.g., in requirements-dev.txt)
    # pip install -r requirements-dev.txt
    ```

## Running Tests

Ensure all tests pass before submitting changes.

```bash
pytest
```

## Coding Standards

- Follow PEP 8 guidelines.
- Use type hints.
- Ensure code is well-documented with docstrings.
- Run linters/formatters if configured (e.g., Black, Flake8).

## Submitting Changes

1.  Create a fork of the repository.
2.  Create a new branch for your feature or bugfix.
3.  Make your changes and add tests.
4.  Ensure all tests pass and code meets standards.
5.  Push your changes to your fork.
6.  Open a Pull Request (PR) against the main repository.
7.  Provide a clear description of your changes in the PR.

We appreciate your contributions!
