#!/usr/bin/env python
"""
Security scanning and validation for Spatial-Omics GFM.
Implements comprehensive security checks and vulnerability assessment.
"""

import os
import sys
import subprocess
import argparse
import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import ast
import importlib.util

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SecurityScanner:
    """Comprehensive security scanner for Python projects."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.scan_results = {}
        self.security_issues = []
        
        # Security patterns to detect
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']',
                r'aws_access_key_id\s*=\s*["\'][^"\']+["\']',
                r'["\'][A-Za-z0-9]{20,}["\']',  # Potential API keys
            ],
            'unsafe_functions': [
                r'eval\s*\(',
                r'exec\s*\(',
                r'subprocess\.call\s*\(',
                r'os\.system\s*\(',
                r'input\s*\(',  # In production code
                r'pickle\.loads\s*\(',
                r'yaml\.load\s*\(',  # Without safe_load
            ],
            'sql_injection': [
                r'\.execute\s*\(\s*["\'].*%.*["\']',
                r'\.execute\s*\(\s*f["\'].*\{.*\}.*["\']',
                r'cursor\.execute\s*\(\s*["\'].*\+.*["\']',
            ],
            'path_traversal': [
                r'\.\./',
                r'open\s*\(\s*.*\+.*\)',  # Dynamic file paths
            ],
            'debug_code': [
                r'print\s*\(',  # In production code
                r'pdb\.set_trace\s*\(',
                r'debugpy\.',
                r'DEBUG\s*=\s*True',
            ]
        }
    
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities."""
        logger.info("Scanning dependencies for vulnerabilities...")
        
        results = {
            'safety_scan': None,
            'pip_audit': None,
            'outdated_packages': None
        }
        
        # Run safety scan
        try:
            safety_result = subprocess.run(
                ['python', '-m', 'safety', 'check', '--json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if safety_result.returncode == 0:
                results['safety_scan'] = {
                    'status': 'clean',
                    'vulnerabilities': []
                }
            else:
                try:
                    vuln_data = json.loads(safety_result.stdout)
                    results['safety_scan'] = {
                        'status': 'vulnerabilities_found',
                        'vulnerabilities': vuln_data
                    }
                except json.JSONDecodeError:
                    results['safety_scan'] = {
                        'status': 'error',
                        'error': safety_result.stderr
                    }
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results['safety_scan'] = {
                'status': 'tool_not_available',
                'error': 'safety tool not installed or timed out'
            }
        
        # Run pip-audit if available
        try:
            audit_result = subprocess.run(
                ['python', '-m', 'pip_audit', '--format=json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if audit_result.returncode == 0:
                try:
                    audit_data = json.loads(audit_result.stdout)
                    results['pip_audit'] = {
                        'status': 'clean' if not audit_data else 'vulnerabilities_found',
                        'vulnerabilities': audit_data
                    }
                except json.JSONDecodeError:
                    results['pip_audit'] = {
                        'status': 'clean',
                        'vulnerabilities': []
                    }
            else:
                results['pip_audit'] = {
                    'status': 'error',
                    'error': audit_result.stderr
                }
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results['pip_audit'] = {
                'status': 'tool_not_available',
                'error': 'pip-audit tool not installed or timed out'
            }
        
        # Check for outdated packages
        try:
            outdated_result = subprocess.run(
                ['python', '-m', 'pip', 'list', '--outdated', '--format=json'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if outdated_result.returncode == 0:
                outdated_data = json.loads(outdated_result.stdout)
                results['outdated_packages'] = {
                    'count': len(outdated_data),
                    'packages': outdated_data
                }
            else:
                results['outdated_packages'] = {
                    'error': outdated_result.stderr
                }
        
        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            results['outdated_packages'] = {
                'error': 'Failed to check outdated packages'
            }
        
        self.scan_results['dependencies'] = results
        return results
    
    def scan_source_code(self) -> Dict[str, Any]:
        """Scan source code for security issues."""
        logger.info("Scanning source code for security issues...")
        
        issues = []
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip test files and virtual environments
            if any(part in str(file_path) for part in ['test', 'tests', 'venv', '.venv', '__pycache__']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_issues = self._scan_file_content(file_path, content)
                issues.extend(file_issues)
                
            except (UnicodeDecodeError, IOError) as e:
                logger.warning(f"Could not read file {file_path}: {e}")
                continue
        
        results = {
            'total_files_scanned': len(python_files),
            'issues_found': len(issues),
            'issues': issues
        }
        
        self.scan_results['source_code'] = results
        self.security_issues.extend(issues)
        
        return results
    
    def _scan_file_content(self, file_path: Path, content: str) -> List[Dict[str, Any]]:
        """Scan individual file content for security issues."""
        issues = []
        lines = content.split('\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Skip comments unless they contain actual secrets
                        if line.strip().startswith('#') and category != 'hardcoded_secrets':
                            continue
                        
                        issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'column': match.start(),
                            'category': category,
                            'pattern': pattern,
                            'matched_text': match.group(),
                            'context': line.strip(),
                            'severity': self._get_severity(category)
                        })
        
        return issues
    
    def _get_severity(self, category: str) -> str:
        """Get severity level for security issue category."""
        severity_map = {
            'hardcoded_secrets': 'high',
            'unsafe_functions': 'medium',
            'sql_injection': 'high',
            'path_traversal': 'medium',
            'debug_code': 'low'
        }
        return severity_map.get(category, 'medium')
    
    def scan_configuration(self) -> Dict[str, Any]:
        """Scan configuration files for security issues."""
        logger.info("Scanning configuration files...")
        
        config_issues = []
        
        # Check common config files
        config_files = [
            'requirements.txt',
            'requirements-dev.txt',
            'requirements-test.txt',
            'setup.py',
            'setup.cfg',
            'pyproject.toml',
            '.env',
            'config.ini',
            'config.yaml',
            'config.json'
        ]
        
        for config_file in config_files:
            file_path = self.project_root / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for secrets in config files
                    if any(pattern in content.lower() for pattern in ['password', 'secret', 'key', 'token']):
                        # Do deeper analysis
                        file_issues = self._scan_file_content(file_path, content)
                        config_issues.extend(file_issues)
                
                except (UnicodeDecodeError, IOError):
                    continue
        
        results = {
            'config_files_scanned': len([f for f in config_files if (self.project_root / f).exists()]),
            'issues_found': len(config_issues),
            'issues': config_issues
        }
        
        self.scan_results['configuration'] = results
        return results
    
    def scan_permissions(self) -> Dict[str, Any]:
        """Scan file permissions for security issues."""
        logger.info("Scanning file permissions...")
        
        permission_issues = []
        
        # Check for overly permissive files
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Check for world-writable files
                    if mode.endswith('6') or mode.endswith('7'):
                        permission_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'permissions': mode,
                            'issue': 'world_writable',
                            'severity': 'medium'
                        })
                    
                    # Check for executable config files
                    if file_path.suffix in ['.json', '.yaml', '.yml', '.ini', '.cfg'] and mode.startswith('7'):
                        permission_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'permissions': mode,
                            'issue': 'executable_config',
                            'severity': 'low'
                        })
                
                except (OSError, PermissionError):
                    continue
        
        results = {
            'issues_found': len(permission_issues),
            'issues': permission_issues
        }
        
        self.scan_results['permissions'] = results
        return results
    
    def scan_imports(self) -> Dict[str, Any]:
        """Scan imports for potentially dangerous modules."""
        logger.info("Scanning imports for dangerous modules...")
        
        dangerous_imports = []
        dangerous_modules = [
            'subprocess',
            'os',
            'eval',
            'exec',
            'pickle',
            'marshal',
            'shelve',
            'dill',
            'yaml',  # If not using safe_load
            'sqlite3',  # Raw SQL
        ]
        
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            if any(part in str(file_path) for part in ['test', 'tests', 'venv', '.venv']):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name in dangerous_modules:
                                    dangerous_imports.append({
                                        'file': str(file_path.relative_to(self.project_root)),
                                        'line': node.lineno,
                                        'module': alias.name,
                                        'type': 'import',
                                        'severity': self._get_import_severity(alias.name)
                                    })
                        
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.module in dangerous_modules:
                                dangerous_imports.append({
                                    'file': str(file_path.relative_to(self.project_root)),
                                    'line': node.lineno,
                                    'module': node.module,
                                    'type': 'from_import',
                                    'severity': self._get_import_severity(node.module)
                                })
                
                except SyntaxError:
                    continue
            
            except (UnicodeDecodeError, IOError):
                continue
        
        results = {
            'dangerous_imports_found': len(dangerous_imports),
            'imports': dangerous_imports
        }
        
        self.scan_results['imports'] = results
        return results
    
    def _get_import_severity(self, module_name: str) -> str:
        """Get severity for dangerous import."""
        high_risk = ['subprocess', 'os', 'eval', 'exec', 'pickle']
        medium_risk = ['marshal', 'shelve', 'dill']
        
        if module_name in high_risk:
            return 'medium'  # Not always dangerous, depends on usage
        elif module_name in medium_risk:
            return 'low'
        else:
            return 'low'
    
    def run_bandit(self) -> Dict[str, Any]:
        """Run Bandit security scanner."""
        logger.info("Running Bandit security scanner...")
        
        try:
            result = subprocess.run(
                [
                    'python', '-m', 'bandit', 
                    '-r', 'spatial_omics_gfm/',
                    '-f', 'json',
                    '-x', 'tests/'
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    
                    results = {
                        'status': 'completed',
                        'issues_found': len(bandit_data.get('results', [])),
                        'results': bandit_data.get('results', []),
                        'metrics': bandit_data.get('metrics', {}),
                        'errors': bandit_data.get('errors', [])
                    }
                    
                except json.JSONDecodeError:
                    results = {
                        'status': 'error',
                        'error': 'Failed to parse Bandit output',
                        'raw_output': result.stdout
                    }
            else:
                results = {
                    'status': 'no_output',
                    'stderr': result.stderr
                }
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results = {
                'status': 'tool_not_available',
                'error': 'Bandit not installed or timed out'
            }
        
        self.scan_results['bandit'] = results
        return results
    
    def generate_security_report(self, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        logger.info("Generating security report...")
        
        # Count total issues by severity
        all_issues = []
        for scan_type, results in self.scan_results.items():
            if isinstance(results, dict) and 'issues' in results:
                all_issues.extend(results['issues'])
        
        severity_counts = {
            'high': len([i for i in all_issues if i.get('severity') == 'high']),
            'medium': len([i for i in all_issues if i.get('severity') == 'medium']),
            'low': len([i for i in all_issues if i.get('severity') == 'low'])
        }
        
        report = {
            'scan_results': self.scan_results,
            'summary': {
                'total_scans': len(self.scan_results),
                'total_issues': len(all_issues),
                'severity_breakdown': severity_counts,
                'risk_score': self._calculate_risk_score(severity_counts)
            },
            'recommendations': self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Security report saved to {output_file}")
        
        return report
    
    def _calculate_risk_score(self, severity_counts: Dict[str, int]) -> int:
        """Calculate overall risk score (0-100)."""
        # Weight different severities
        score = (
            severity_counts['high'] * 10 +
            severity_counts['medium'] * 5 +
            severity_counts['low'] * 1
        )
        
        # Cap at 100
        return min(score, 100)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []
        
        if self.scan_results.get('dependencies', {}).get('safety_scan', {}).get('status') == 'vulnerabilities_found':
            recommendations.append("Update vulnerable dependencies identified by safety scan")
        
        if self.scan_results.get('source_code', {}).get('issues_found', 0) > 0:
            recommendations.append("Review and fix source code security issues")
        
        if self.scan_results.get('permissions', {}).get('issues_found', 0) > 0:
            recommendations.append("Fix file permission issues")
        
        if not recommendations:
            recommendations.append("No major security issues found - maintain current security practices")
        
        # Add general recommendations
        recommendations.extend([
            "Regularly update dependencies",
            "Use environment variables for secrets",
            "Implement proper input validation",
            "Use parameterized queries for database operations",
            "Enable security linting in CI/CD pipeline"
        ])
        
        return recommendations
    
    def run_full_scan(self) -> Dict[str, Any]:
        """Run complete security scan."""
        logger.info("Starting comprehensive security scan...")
        
        # Run all scans
        self.scan_dependencies()
        self.scan_source_code()
        self.scan_configuration()
        self.scan_permissions()
        self.scan_imports()
        self.run_bandit()
        
        # Generate report
        report = self.generate_security_report()
        
        logger.info(f"Security scan completed. Found {report['summary']['total_issues']} issues.")
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run security scan for Spatial-Omics GFM')
    
    parser.add_argument(
        '--scan-type',
        choices=['dependencies', 'source', 'config', 'permissions', 'imports', 'bandit', 'all'],
        default='all',
        help='Type of security scan to run'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for security report'
    )
    
    parser.add_argument(
        '--fail-on-high',
        action='store_true',
        help='Fail if high severity issues are found'
    )
    
    parser.add_argument(
        '--fail-on-medium',
        action='store_true',
        help='Fail if medium or high severity issues are found'
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Initialize scanner
    scanner = SecurityScanner(project_root)
    
    try:
        # Run scans based on type
        if args.scan_type == 'dependencies':
            scanner.scan_dependencies()
        elif args.scan_type == 'source':
            scanner.scan_source_code()
        elif args.scan_type == 'config':
            scanner.scan_configuration()
        elif args.scan_type == 'permissions':
            scanner.scan_permissions()
        elif args.scan_type == 'imports':
            scanner.scan_imports()
        elif args.scan_type == 'bandit':
            scanner.run_bandit()
        else:
            scanner.run_full_scan()
        
        # Generate report
        report = scanner.generate_security_report(args.output)
        
        # Print summary
        summary = report['summary']
        logger.info(f"Security scan summary:")
        logger.info(f"  Total issues: {summary['total_issues']}")
        logger.info(f"  High severity: {summary['severity_breakdown']['high']}")
        logger.info(f"  Medium severity: {summary['severity_breakdown']['medium']}")
        logger.info(f"  Low severity: {summary['severity_breakdown']['low']}")
        logger.info(f"  Risk score: {summary['risk_score']}/100")
        
        # Check failure conditions
        if args.fail_on_high and summary['severity_breakdown']['high'] > 0:
            logger.error("High severity security issues found!")
            sys.exit(1)
        
        if args.fail_on_medium and (summary['severity_breakdown']['high'] > 0 or summary['severity_breakdown']['medium'] > 0):
            logger.error("Medium or high severity security issues found!")
            sys.exit(1)
        
        logger.info("Security scan completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Security scan interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Security scan failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()