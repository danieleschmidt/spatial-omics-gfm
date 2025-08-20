# Security Hardening Guide

## Current Security Issues Identified

The security scan found the following issues that need attention:

### Code Security Issues
- `eval()` and `exec()` usage in production_readiness_check.py
- `__import__` dynamic imports
- `subprocess.call` usage without input validation

### Secret Management Issues
- Hardcoded references to "password", "api_key", "secret" in test files

## Security Hardening Recommendations

### 1. Code Security
```python
# AVOID: Dynamic code execution
eval(user_input)  # Never do this
exec(code_string)  # Security risk

# USE: Safe alternatives
import ast
ast.literal_eval(safe_string)  # For literals only
```

### 2. Input Validation
```python
# Implement strict input validation
def validate_file_path(path: str) -> bool:
    import os.path
    # Check for path traversal
    if '..' in path or path.startswith('/'):
        return False
    # Check for allowed extensions
    allowed_extensions = {'.h5', '.csv', '.txt', '.json'}
    return any(path.endswith(ext) for ext in allowed_extensions)
```

### 3. Secure Configuration
```python
# Use environment variables for secrets
import os
API_KEY = os.getenv('SPATIAL_GFM_API_KEY')
if not API_KEY:
    raise ValueError("API_KEY environment variable required")
```

### 4. Dependency Security
- Regularly update dependencies: `pip-audit`
- Use security scanners: `bandit`, `safety`
- Pin dependency versions in production

### 5. Data Protection
- Encrypt sensitive data at rest
- Use HTTPS for all network communications
- Implement access controls for data files

### 6. Production Security
- Enable container security scanning
- Use non-root users in containers
- Implement network security groups
- Enable audit logging

## Security Checklist

- [ ] Remove eval() and exec() usage
- [ ] Implement input validation
- [ ] Use environment variables for secrets
- [ ] Enable dependency scanning
- [ ] Implement access controls
- [ ] Enable audit logging
- [ ] Use HTTPS everywhere
- [ ] Regular security audits
