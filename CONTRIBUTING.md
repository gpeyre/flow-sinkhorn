# Contributing to Flow Sinkhorn

Thank you for your interest in contributing to Flow Sinkhorn!

## Code Structure

The repository is organized as follows:

- `flowsinkhorn/` - Main Python package
  - `sinkhorn.py` - Regularized solvers
  - `exact.py` - Exact linear programming solvers
  - `__init__.py` - Package exports
- `examples/` - Jupyter notebooks demonstrating usage
- `paper/` - LaTeX source for the paper
- `tests/` - Unit tests (to be added)

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/flow-sinkhorn.git
cd flow-sinkhorn
```

2. Install in development mode with all dependencies:
```bash
pip install -e ".[all]"
```

3. Make your changes in a new branch:
```bash
git checkout -b feature/your-feature-name
```

## Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to all functions (NumPy style)
- Include type hints where appropriate
- Keep functions focused and modular

## Documentation

- All functions should have complete docstrings with:
  - One-line summary
  - Extended description
  - Parameters section
  - Returns section
  - Examples (if applicable)
  - Notes (if applicable)

Example:
```python
def my_function(param1, param2):
    """
    Brief description of function.

    Longer description with more details about what the function does,
    its algorithm, or special considerations.

    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2

    Returns
    -------
    result : type
        Description of return value

    Examples
    --------
    >>> my_function(1, 2)
    3
    """
```

## Testing

Before submitting a pull request:

1. Test your code with different inputs
2. Verify numerical stability
3. Check that examples still run
4. Ensure backward compatibility

## Submitting Changes

1. Commit your changes with clear messages:
```bash
git commit -m "Add feature: description of feature"
```

2. Push to your fork:
```bash
git push origin feature/your-feature-name
```

3. Open a Pull Request with:
   - Clear description of changes
   - Motivation for the change
   - Any relevant issue numbers

## Areas for Contribution

We welcome contributions in:

- **Performance improvements**: Optimizing hot loops, better sparse operations
- **New features**: Additional solvers, distance metrics, regularizers
- **Documentation**: Improving docstrings, adding examples, tutorials
- **Testing**: Unit tests, integration tests, benchmarks
- **Bug fixes**: Fixing numerical issues, edge cases

## Questions?

Feel free to open an issue for discussion before starting major work.

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.
