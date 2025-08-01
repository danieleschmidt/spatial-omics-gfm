# Contributing to Spatial-Omics GFM

Thank you for considering contributing to Spatial-Omics GFM! This document provides guidelines for contributing to this project.

## Quick Start

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/spatial-omics-gfm.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`
6. Create a feature branch: `git checkout -b feature/your-feature-name`
7. Make your changes and commit them
8. Push to your fork and submit a pull request

## Development Environment Setup

### Prerequisites
- Python 3.9 or higher
- Git
- CUDA-compatible GPU (optional, for GPU acceleration)

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/spatial-omics-gfm.git
cd spatial-omics-gfm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific test categories
pytest -m "not slow"  # Skip slow tests
pytest -m integration  # Run integration tests only
pytest -m gpu  # Run GPU tests (requires CUDA)

# Run with coverage
pytest --cov=spatial_omics_gfm --cov-report=html
```

### Code Quality Checks
```bash
# Format code
black spatial_omics_gfm tests

# Lint code
ruff check spatial_omics_gfm tests

# Type checking
mypy spatial_omics_gfm

# Security scanning
bandit -r spatial_omics_gfm
```

## Development Guidelines

### Code Style
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 88)
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Ruff](https://beta.ruff.rs/) for linting
- Add type hints for all public functions
- Write docstrings for all public classes and functions

### Testing
- Write tests for all new features and bug fixes
- Maintain test coverage above 80%
- Use descriptive test names that explain what is being tested
- Group related tests in classes
- Use fixtures for common test data

### Documentation
- Update docstrings for any changed functions
- Add examples to docstrings when helpful
- Update README.md if adding new features
- Add tutorial notebooks for significant new functionality

### Git Workflow
- Create feature branches from `main`
- Use descriptive commit messages
- Keep commits atomic (one logical change per commit)
- Rebase feature branches before merging
- Use conventional commit format when possible

### Commit Message Format
```
type(scope): short description

Longer description if needed.

Fixes #issue_number
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Types of Contributions

### Bug Reports
When filing a bug report, please include:
- Clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- System information (OS, Python version, package versions)
- Minimal code example if possible

### Feature Requests
For feature requests, please provide:
- Clear description of the proposed feature
- Use case and motivation
- Possible implementation approach
- Any related issues or discussions

### Code Contributions
We welcome:
- Bug fixes
- New features
- Performance improvements
- Documentation improvements
- Test coverage improvements

### Documentation
- API documentation improvements
- Tutorial notebooks
- Example scripts
- Installation guides
- Troubleshooting guides

## Code Architecture

### Project Structure
```
spatial-omics-gfm/
├── spatial_omics_gfm/          # Main package
│   ├── models/                 # Model implementations
│   ├── data/                   # Data loading and preprocessing
│   ├── tasks/                  # Task-specific implementations
│   ├── training/               # Training utilities
│   ├── inference/              # Inference utilities
│   ├── visualization/          # Plotting and visualization
│   └── cli.py                 # Command-line interface
├── tests/                      # Test suite
├── docs/                       # Documentation
├── examples/                   # Example scripts and notebooks
└── benchmarks/                 # Benchmarking scripts
```

### Design Principles
- **Modularity**: Keep components loosely coupled and highly cohesive
- **Extensibility**: Design for easy addition of new models and datasets
- **Performance**: Optimize for large-scale spatial transcriptomics data
- **Usability**: Provide simple APIs for common use cases
- **Reproducibility**: Ensure deterministic results where possible

## Review Process

### Pull Request Guidelines
- Provide clear description of changes
- Include tests for new functionality
- Update documentation as needed
- Ensure all CI checks pass
- Link to related issues

### Review Criteria
- Code quality and style compliance
- Test coverage and quality
- Documentation completeness
- Performance impact
- API design consistency
- Backward compatibility

## Community Guidelines

### Code of Conduct
This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

### Communication
- Use GitHub Issues for bug reports and feature requests
- Use GitHub Discussions for questions and general discussion
- Be respectful and constructive in all interactions
- Help others learn and contribute

## Getting Help

- Check existing [Issues](https://github.com/danielschmidt/spatial-omics-gfm/issues)
- Read the [Documentation](https://spatial-omics-gfm.readthedocs.io)
- Ask questions in [Discussions](https://github.com/danielschmidt/spatial-omics-gfm/discussions)
- Join our community chat (link to be added)

## Recognition

Contributors will be recognized in:
- `AUTHORS.md` file
- Release notes for significant contributions
- Annual contributor spotlights

Thank you for contributing to Spatial-Omics GFM!