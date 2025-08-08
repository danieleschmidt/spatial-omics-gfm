#!/usr/bin/env python
"""
Comprehensive test runner for Spatial-Omics GFM.
Runs all tests, generates coverage reports, and validates quality gates.
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRunner:
    """Comprehensive test runner with quality gates."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_results = {}
        self.quality_gates = {
            'min_coverage': 85.0,
            'max_test_duration': 300,  # 5 minutes
            'max_failed_tests': 0,
            'required_test_categories': ['unit', 'integration', 'performance']
        }
    
    def run_unit_tests(self, verbose: bool = False) -> Dict:
        """Run unit tests."""
        logger.info("Running unit tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/',
            '--cov=spatial_omics_gfm',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            '--cov-fail-under=85',
            '-m', 'not slow and not integration and not performance',
            '--tb=short'
        ]
        
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['-q'])
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        # Parse coverage from output
        coverage = self._parse_coverage_from_output(result.stdout)
        
        test_result = {
            'type': 'unit',
            'return_code': result.returncode,
            'duration': duration,
            'coverage': coverage,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        self.test_results['unit'] = test_result
        
        if result.returncode == 0:
            logger.info(f"Unit tests passed in {duration:.2f}s with {coverage:.1f}% coverage")
        else:
            logger.error(f"Unit tests failed (return code: {result.returncode})")
        
        return test_result
    
    def run_integration_tests(self, verbose: bool = False) -> Dict:
        """Run integration tests."""
        logger.info("Running integration tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/',
            '-m', 'integration',
            '--tb=short'
        ]
        
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['-q'])
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        test_result = {
            'type': 'integration',
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        self.test_results['integration'] = test_result
        
        if result.returncode == 0:
            logger.info(f"Integration tests passed in {duration:.2f}s")
        else:
            logger.error(f"Integration tests failed (return code: {result.returncode})")
        
        return test_result
    
    def run_performance_tests(self, verbose: bool = False) -> Dict:
        """Run performance tests."""
        logger.info("Running performance tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/',
            '-m', 'performance',
            '--tb=short',
            '--timeout=60'  # 1 minute timeout per test
        ]
        
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['-q'])
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        test_result = {
            'type': 'performance',
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        self.test_results['performance'] = test_result
        
        if result.returncode == 0:
            logger.info(f"Performance tests passed in {duration:.2f}s")
        else:
            logger.error(f"Performance tests failed (return code: {result.returncode})")
        
        return test_result
    
    def run_slow_tests(self, verbose: bool = False) -> Dict:
        """Run slow tests."""
        logger.info("Running slow tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/',
            '-m', 'slow',
            '--tb=short',
            '--timeout=300'  # 5 minute timeout per test
        ]
        
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['-q'])
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        test_result = {
            'type': 'slow',
            'return_code': result.returncode,
            'duration': duration,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        self.test_results['slow'] = test_result
        
        if result.returncode == 0:
            logger.info(f"Slow tests passed in {duration:.2f}s")
        else:
            logger.error(f"Slow tests failed (return code: {result.returncode})")
        
        return test_result
    
    def run_all_tests(self, verbose: bool = False) -> Dict:
        """Run all tests."""
        logger.info("Running all tests...")
        
        cmd = [
            'python', '-m', 'pytest',
            'tests/',
            '--cov=spatial_omics_gfm',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov',
            '--cov-report=xml:coverage.xml',
            '--cov-fail-under=85',
            '--tb=short'
        ]
        
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['-q'])
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time
        
        # Parse coverage from output
        coverage = self._parse_coverage_from_output(result.stdout)
        
        test_result = {
            'type': 'all',
            'return_code': result.returncode,
            'duration': duration,
            'coverage': coverage,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        self.test_results['all'] = test_result
        
        if result.returncode == 0:
            logger.info(f"All tests passed in {duration:.2f}s with {coverage:.1f}% coverage")
        else:
            logger.error(f"Some tests failed (return code: {result.returncode})")
        
        return test_result
    
    def run_linting(self) -> Dict:
        """Run code linting."""
        logger.info("Running code linting...")
        
        # Run flake8
        flake8_result = subprocess.run(
            ['python', '-m', 'flake8', 'spatial_omics_gfm/', '--max-line-length=100'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        # Run black check
        black_result = subprocess.run(
            ['python', '-m', 'black', '--check', 'spatial_omics_gfm/'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        # Run isort check
        isort_result = subprocess.run(
            ['python', '-m', 'isort', '--check-only', 'spatial_omics_gfm/'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        lint_result = {
            'type': 'linting',
            'flake8': {
                'return_code': flake8_result.returncode,
                'stdout': flake8_result.stdout,
                'stderr': flake8_result.stderr
            },
            'black': {
                'return_code': black_result.returncode,
                'stdout': black_result.stdout,
                'stderr': black_result.stderr
            },
            'isort': {
                'return_code': isort_result.returncode,
                'stdout': isort_result.stdout,
                'stderr': isort_result.stderr
            },
            'passed': all(r.returncode == 0 for r in [flake8_result, black_result, isort_result])
        }
        
        self.test_results['linting'] = lint_result
        
        if lint_result['passed']:
            logger.info("Code linting passed")
        else:
            logger.error("Code linting failed")
        
        return lint_result
    
    def run_type_checking(self) -> Dict:
        """Run type checking with mypy."""
        logger.info("Running type checking...")
        
        result = subprocess.run(
            ['python', '-m', 'mypy', 'spatial_omics_gfm/', '--ignore-missing-imports'],
            cwd=self.project_root,
            capture_output=True,
            text=True
        )
        
        type_result = {
            'type': 'type_checking',
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'passed': result.returncode == 0
        }
        
        self.test_results['type_checking'] = type_result
        
        if result.returncode == 0:
            logger.info("Type checking passed")
        else:
            logger.error("Type checking failed")
        
        return type_result
    
    def validate_quality_gates(self) -> Tuple[bool, List[str]]:
        """Validate quality gates."""
        logger.info("Validating quality gates...")
        
        violations = []
        
        # Check minimum coverage
        if 'unit' in self.test_results:
            coverage = self.test_results['unit'].get('coverage', 0)
            if coverage < self.quality_gates['min_coverage']:
                violations.append(
                    f"Coverage {coverage:.1f}% below minimum {self.quality_gates['min_coverage']:.1f}%"
                )
        
        # Check test duration
        total_duration = sum(
            result.get('duration', 0) for result in self.test_results.values()
            if isinstance(result, dict) and 'duration' in result
        )
        if total_duration > self.quality_gates['max_test_duration']:
            violations.append(
                f"Total test duration {total_duration:.1f}s exceeds maximum {self.quality_gates['max_test_duration']}s"
            )
        
        # Check for failed tests
        failed_tests = sum(
            1 for result in self.test_results.values()
            if isinstance(result, dict) and not result.get('passed', True)
        )
        if failed_tests > self.quality_gates['max_failed_tests']:
            violations.append(f"{failed_tests} test categories failed")
        
        passed = len(violations) == 0
        
        if passed:
            logger.info("All quality gates passed")
        else:
            logger.error(f"Quality gate violations: {violations}")
        
        return passed, violations
    
    def generate_report(self, output_file: Optional[Path] = None) -> Dict:
        """Generate comprehensive test report."""
        report = {
            'timestamp': time.time(),
            'test_results': self.test_results,
            'quality_gates': self.quality_gates,
            'summary': {
                'total_tests': len(self.test_results),
                'passed_tests': sum(
                    1 for r in self.test_results.values()
                    if isinstance(r, dict) and r.get('passed', False)
                ),
                'total_duration': sum(
                    r.get('duration', 0) for r in self.test_results.values()
                    if isinstance(r, dict) and 'duration' in r
                ),
                'overall_coverage': self.test_results.get('unit', {}).get('coverage', 0)
            }
        }
        
        # Validate quality gates
        gates_passed, violations = self.validate_quality_gates()
        report['quality_gates_passed'] = gates_passed
        report['quality_gate_violations'] = violations
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Test report saved to {output_file}")
        
        return report
    
    def _parse_coverage_from_output(self, output: str) -> float:
        """Parse coverage percentage from pytest output."""
        lines = output.split('\n')
        for line in lines:
            if 'TOTAL' in line and '%' in line:
                try:
                    # Extract percentage from line like "TOTAL    1234    567    85%"
                    parts = line.split()
                    for part in parts:
                        if part.endswith('%'):
                            return float(part[:-1])
                except:
                    continue
        return 0.0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Spatial-Omics GFM tests')
    
    parser.add_argument(
        '--test-type',
        choices=['unit', 'integration', 'performance', 'slow', 'all'],
        default='all',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--report',
        type=Path,
        help='Output file for test report'
    )
    
    parser.add_argument(
        '--lint',
        action='store_true',
        help='Run linting'
    )
    
    parser.add_argument(
        '--type-check',
        action='store_true',
        help='Run type checking'
    )
    
    parser.add_argument(
        '--quality-gates',
        action='store_true',
        help='Validate quality gates'
    )
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Initialize test runner
    runner = TestRunner(project_root)
    
    try:
        # Run tests based on type
        if args.test_type == 'unit':
            runner.run_unit_tests(args.verbose)
        elif args.test_type == 'integration':
            runner.run_integration_tests(args.verbose)
        elif args.test_type == 'performance':
            runner.run_performance_tests(args.verbose)
        elif args.test_type == 'slow':
            runner.run_slow_tests(args.verbose)
        elif args.test_type == 'all':
            runner.run_all_tests(args.verbose)
        
        # Run linting if requested
        if args.lint:
            runner.run_linting()
        
        # Run type checking if requested
        if args.type_check:
            runner.run_type_checking()
        
        # Generate report
        report = runner.generate_report(args.report)
        
        # Validate quality gates if requested
        if args.quality_gates:
            gates_passed, violations = runner.validate_quality_gates()
            if not gates_passed:
                logger.error("Quality gates failed!")
                for violation in violations:
                    logger.error(f"  - {violation}")
                sys.exit(1)
        
        # Exit with appropriate code
        if report['summary']['passed_tests'] < report['summary']['total_tests']:
            sys.exit(1)
        
        logger.info("All tests completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()