# CI/CD Pipeline Setup Guide

## Overview

This guide provides instructions for setting up a complete CI/CD pipeline for the Spatial-Omics GFM project. Since GitHub Actions workflows require special permissions, this documentation provides the configuration files and setup instructions for manual implementation.

## GitHub Actions Configuration

### Main CI/CD Pipeline

Create `.github/workflows/ci.yml` in your repository with the following content:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Run quality gates
      run: |
        python run_quality_gates.py
    
    - name: Run tests
      run: |
        pytest --cov=spatial_omics_gfm --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install security tools
      run: |
        pip install bandit safety
    
    - name: Run security scan
      run: |
        bandit -r spatial_omics_gfm/
        safety check

  build-docker:
    runs-on: ubuntu-latest
    needs: [test, security]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Build Docker image
      run: |
        docker build -t spatial-omics-gfm:latest .
    
    - name: Test Docker image
      run: |
        docker run --rm spatial-omics-gfm:latest python -c "import spatial_omics_gfm; print('OK')"

  deploy:
    runs-on: ubuntu-latest
    needs: [test, security, build-docker]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add actual deployment steps here
```

### Quality Gates Workflow

Create `.github/workflows/quality-gates.yml`:

```yaml
name: Quality Gates

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-gates:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install numpy pandas matplotlib psutil
        pip install -e .
    
    - name: Run enhanced quality gates
      run: |
        python enhanced_quality_gates.py
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      with:
        name: quality-gate-results
        path: |
          quality_gate_results.json
          enhanced_quality_results.json
```

## GitLab CI Configuration

If using GitLab, create `.gitlab-ci.yml`:

```yaml
stages:
  - test
  - security
  - build
  - deploy

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

before_script:
  - python -m venv venv
  - source venv/bin/activate
  - python -m pip install --upgrade pip

test:
  stage: test
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.9", "3.10", "3.11", "3.12"]
  image: python:${PYTHON_VERSION}
  script:
    - pip install -e ".[dev]"
    - python run_quality_gates.py
    - pytest --cov=spatial_omics_gfm --cov-report=xml
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security_scan:
  stage: security
  image: python:3.11
  script:
    - pip install bandit safety
    - bandit -r spatial_omics_gfm/
    - safety check

build_docker:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t spatial-omics-gfm:$CI_COMMIT_SHA .
    - docker run --rm spatial-omics-gfm:$CI_COMMIT_SHA python -c "import spatial_omics_gfm; print('OK')"

deploy_staging:
  stage: deploy
  script:
    - echo "Deploying to staging"
    # Add deployment commands
  only:
    - main
```

## Local Development Setup

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.270
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.3.0
    hooks:
      - id: mypy
```

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

### Local Quality Gates

Run quality gates locally:

```bash
# Install dependencies
pip install numpy pandas matplotlib psutil

# Run enhanced quality gates
python enhanced_quality_gates.py

# Run standard quality gates
python run_quality_gates.py
```

## Docker-based CI/CD

### Dockerfile for CI

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy source code
COPY . .

# Run quality gates
RUN python enhanced_quality_gates.py

# Run tests
RUN python -m pytest

CMD ["python", "-c", "import spatial_omics_gfm; print('Spatial-Omics GFM ready!')"]
```

### Build and Test Script

Create `scripts/ci_build.sh`:

```bash
#!/bin/bash
set -e

echo "🚀 Building and testing Spatial-Omics GFM"

# Build Docker image
docker build -t spatial-omics-gfm:test .

# Run tests in container
docker run --rm spatial-omics-gfm:test python -m pytest

# Run quality gates
docker run --rm spatial-omics-gfm:test python enhanced_quality_gates.py

echo "✅ All tests passed!"
```

## Cloud Platform Integration

### AWS CodeBuild

Create `buildspec.yml`:

```yaml
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.11
    commands:
      - pip install --upgrade pip
      
  pre_build:
    commands:
      - pip install -e ".[dev]"
      
  build:
    commands:
      - python enhanced_quality_gates.py
      - python -m pytest --cov=spatial_omics_gfm
      
  post_build:
    commands:
      - echo "Build completed on `date`"
      
artifacts:
  files:
    - '**/*'
  base-directory: '.'
```

### Azure DevOps

Create `azure-pipelines.yml`:

```yaml
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

strategy:
  matrix:
    Python39:
      python.version: '3.9'
    Python310:
      python.version: '3.10'
    Python311:
      python.version: '3.11'
    Python312:
      python.version: '3.12'

steps:
- task: UsePythonVersion@0
  inputs:
    versionSpec: '$(python.version)'
  displayName: 'Use Python $(python.version)'

- script: |
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
  displayName: 'Install dependencies'

- script: |
    python enhanced_quality_gates.py
  displayName: 'Run quality gates'

- script: |
    pytest --cov=spatial_omics_gfm --cov-report=xml
  displayName: 'Run tests'
```

## Setup Instructions

1. **Choose your CI/CD platform** (GitHub Actions, GitLab CI, etc.)
2. **Copy the appropriate configuration** to your repository
3. **Install local development tools**:
   ```bash
   pip install pre-commit
   pre-commit install
   ```
4. **Configure secrets** (if needed):
   - API keys
   - Docker registry credentials
   - Deployment tokens
5. **Test the pipeline** with a test commit
6. **Monitor results** and adjust as needed

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Add to `pyproject.toml` or install in CI
2. **Permission errors**: Check GitHub App permissions or runner permissions
3. **Docker build failures**: Verify Dockerfile and dependencies
4. **Test failures**: Check test configuration and environment

### Support

For issues with CI/CD setup:
1. Check the logs in your CI/CD platform
2. Verify all dependencies are installed
3. Ensure proper permissions are set
4. Review the configuration syntax

This setup provides a robust CI/CD pipeline that maintains code quality, security, and reliability for the Spatial-Omics GFM project.