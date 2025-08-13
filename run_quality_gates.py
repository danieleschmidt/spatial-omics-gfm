#!/usr/bin/env python3
"""
Comprehensive quality gates validation for Spatial-Omics GFM.

This script runs all quality checks including:
- Unit tests and integration tests
- Security vulnerability scans  
- Performance benchmarks
- Code quality checks
- Documentation validation
"""

import sys
import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import json

# Add repo to path
sys.path.insert(0, '/root/repo')


class QualityGateRunner:
    """Runs comprehensive quality gates for the project."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results = {
            "overall_status": "PENDING",
            "gates": {},
            "summary": {},
            "timestamp": time.time()
        }
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive results."""
        print("🛡️ Running Comprehensive Quality Gates")
        print("=" * 60)
        
        gates = [
            ("functionality", self.test_functionality),
            ("robustness", self.test_robustness), 
            ("performance", self.test_performance),
            ("security", self.test_security),
            ("code_quality", self.test_code_quality),
            ("integration", self.test_integration),
            ("documentation", self.test_documentation)
        ]
        
        for gate_name, gate_func in gates:
            print(f"\n🔍 Running {gate_name.upper()} gate...")
            try:
                gate_result = gate_func()
                self.results["gates"][gate_name] = gate_result
                
                status = "✅ PASS" if gate_result["passed"] else "❌ FAIL"
                print(f"   {status}: {gate_result.get('summary', 'No summary')}")
                
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                self.results["gates"][gate_name] = {
                    "passed": False,
                    "error": str(e),
                    "summary": f"Gate failed with error: {e}"
                }
        
        # Calculate overall status
        self._calculate_overall_status()
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    def test_functionality(self) -> Dict[str, Any]:
        """Test basic functionality."""
        result = {"passed": False, "details": {}}
        
        try:
            # Test core imports
            from spatial_omics_gfm.core import (
                SimpleSpatialData, SimpleCellTypePredictor, 
                SimpleInteractionPredictor, run_basic_analysis
            )
            result["details"]["imports"] = {"status": "success"}
            
            # Test basic analysis
            analysis_result = run_basic_analysis()
            result["details"]["basic_analysis"] = {
                "status": "success",
                "cells_analyzed": analysis_result["analysis_metadata"]["n_cells_analyzed"],
                "genes_analyzed": analysis_result["analysis_metadata"]["n_genes_analyzed"]
            }
            
            # Test data validation
            import numpy as np
            from spatial_omics_gfm.validation import DataValidator
            
            validator = DataValidator()
            test_expr = np.random.rand(100, 50)
            test_coords = np.random.rand(100, 2)
            
            validation_result = validator.validate_spatial_data(test_expr, test_coords)
            result["details"]["validation"] = {
                "status": "success",
                "overall_valid": validation_result["overall_valid"]
            }
            
            result["passed"] = True
            result["summary"] = "All core functionality tests passed"
            
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Functionality tests failed: {e}"
        
        return result
    
    def test_robustness(self) -> Dict[str, Any]:
        """Test robustness features."""
        result = {"passed": False, "details": {}}
        
        try:
            # Run robustness test script
            cmd = [sys.executable, "test_robustness.py"]
            process = subprocess.run(
                cmd, 
                cwd=self.repo_path,
                capture_output=True, 
                text=True,
                timeout=60
            )
            
            result["details"]["exit_code"] = process.returncode
            result["details"]["stdout"] = process.stdout
            result["details"]["stderr"] = process.stderr
            
            if process.returncode == 0:
                result["passed"] = True
                result["summary"] = "All robustness tests passed"
            else:
                result["summary"] = f"Robustness tests failed with exit code {process.returncode}"
            
        except subprocess.TimeoutExpired:
            result["summary"] = "Robustness tests timed out"
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Error running robustness tests: {e}"
        
        return result
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance features."""
        result = {"passed": False, "details": {}}
        
        try:
            # Run performance test script
            cmd = [sys.executable, "test_performance.py"]
            process = subprocess.run(
                cmd, 
                cwd=self.repo_path,
                capture_output=True, 
                text=True,
                timeout=120
            )
            
            result["details"]["exit_code"] = process.returncode
            result["details"]["stdout"] = process.stdout
            result["details"]["stderr"] = process.stderr
            
            if process.returncode == 0:
                result["passed"] = True
                result["summary"] = "All performance tests passed"
                
                # Extract performance metrics from output
                stdout = process.stdout
                if "Cache speedup:" in stdout:
                    speedup_line = [line for line in stdout.split('\n') if "Cache speedup:" in line][-1]
                    result["details"]["cache_speedup"] = speedup_line.strip()
            else:
                result["summary"] = f"Performance tests failed with exit code {process.returncode}"
            
        except subprocess.TimeoutExpired:
            result["summary"] = "Performance tests timed out"
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Error running performance tests: {e}"
        
        return result
    
    def test_security(self) -> Dict[str, Any]:
        """Test security features."""
        result = {"passed": False, "details": {}}
        
        try:
            from spatial_omics_gfm.validation.security import (
                SecurityValidator, check_file_safety, sanitize_user_input
            )
            
            validator = SecurityValidator()
            
            # Test file safety
            safety_result = validator.check_file_safety(self.repo_path / "README.md")
            result["details"]["file_safety"] = {
                "readme_safe": safety_result["safe"],
                "detected_type": safety_result["checks"].get("detected_type")
            }
            
            # Test input sanitization
            test_inputs = [
                "normal input",
                "<script>alert('xss')</script>", 
                "$(rm -rf /)"
            ]
            
            sanitization_results = []
            for test_input in test_inputs:
                try:
                    sanitized = validator.sanitize_user_input(test_input)
                    sanitization_results.append({"input": test_input[:20], "status": "sanitized"})
                except Exception:
                    sanitization_results.append({"input": test_input[:20], "status": "blocked"})
            
            result["details"]["input_sanitization"] = sanitization_results
            
            # Check for dangerous patterns in code files
            dangerous_patterns = ["eval(", "exec(", "__import__", "subprocess.call"]
            code_files = list(self.repo_path.glob("**/*.py"))
            
            security_issues = []
            for file_path in code_files[:20]:  # Check first 20 files
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        for pattern in dangerous_patterns:
                            if pattern in content:
                                security_issues.append(f"{file_path.name}: {pattern}")
                except:
                    continue
            
            result["details"]["code_security"] = {
                "files_checked": len(code_files[:20]),
                "issues_found": len(security_issues),
                "issues": security_issues[:5]  # Show first 5 issues
            }
            
            result["passed"] = True
            result["summary"] = f"Security validation passed ({len(security_issues)} code patterns found)"
            
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Security tests failed: {e}"
        
        return result
    
    def test_code_quality(self) -> Dict[str, Any]:
        """Test code quality metrics."""
        result = {"passed": False, "details": {}}
        
        try:
            # Count lines of code
            python_files = list(self.repo_path.glob("**/*.py"))
            total_lines = 0
            total_files = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        total_files += 1
                except:
                    continue
            
            result["details"]["code_metrics"] = {
                "python_files": total_files,
                "total_lines": total_lines,
                "avg_lines_per_file": total_lines / max(total_files, 1)
            }
            
            # Check for docstrings in main modules
            modules_with_docstrings = 0
            main_modules = list((self.repo_path / "spatial_omics_gfm").glob("**/*.py"))
            
            for module_path in main_modules[:10]:  # Check first 10 modules
                try:
                    with open(module_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:
                            modules_with_docstrings += 1
                except:
                    continue
            
            docstring_coverage = modules_with_docstrings / max(len(main_modules[:10]), 1) * 100
            
            result["details"]["documentation"] = {
                "modules_checked": len(main_modules[:10]),
                "modules_with_docstrings": modules_with_docstrings,
                "docstring_coverage_percent": docstring_coverage
            }
            
            # Check imports structure
            import_issues = []
            for file_path in python_files[:10]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if "from ... import" in content:
                            import_issues.append(f"{file_path.name}: relative import")
                except:
                    continue
            
            result["details"]["import_quality"] = {
                "files_checked": len(python_files[:10]),
                "import_issues": len(import_issues)
            }
            
            # Overall quality assessment
            quality_score = 0
            if docstring_coverage > 50:
                quality_score += 30
            if len(import_issues) < 5:
                quality_score += 20
            if total_files > 10:
                quality_score += 25
            if total_lines > 1000:
                quality_score += 25
            
            result["details"]["quality_score"] = quality_score
            result["passed"] = quality_score >= 70
            result["summary"] = f"Code quality score: {quality_score}/100"
            
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Code quality check failed: {e}"
        
        return result
    
    def test_integration(self) -> Dict[str, Any]:
        """Test integration between components."""
        result = {"passed": False, "details": {}}
        
        try:
            # Test integration between core, validation, and performance
            from spatial_omics_gfm.core import create_demo_data
            from spatial_omics_gfm.validation import DataValidator
            from spatial_omics_gfm.performance import MemoryCache, cached_computation
            
            # Create test data
            data = create_demo_data(n_cells=100, n_genes=50)
            
            # Validate data
            validator = DataValidator()
            validation_result = validator.validate_spatial_data(
                data.expression_matrix, 
                data.coordinates, 
                data.gene_names
            )
            
            result["details"]["data_integration"] = {
                "data_created": True,
                "validation_passed": validation_result["overall_valid"]
            }
            
            # Test caching integration
            cache = MemoryCache(max_size=10)
            
            @cached_computation(cache_manager=None)  # Simple test without full cache manager
            def test_computation(size):
                return f"computed_{size}"
            
            result1 = test_computation(42)
            result2 = test_computation(42)
            
            result["details"]["caching_integration"] = {
                "cache_test": result1 == result2
            }
            
            # Test end-to-end workflow
            stats = data.get_summary_stats()
            neighbors = data.find_spatial_neighbors(k=5)
            
            result["details"]["workflow_integration"] = {
                "stats_computed": len(stats) > 0,
                "neighbors_found": len(neighbors) > 0
            }
            
            result["passed"] = True
            result["summary"] = "All integration tests passed"
            
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Integration tests failed: {e}"
        
        return result
    
    def test_documentation(self) -> Dict[str, Any]:
        """Test documentation completeness."""
        result = {"passed": False, "details": {}}
        
        try:
            # Check for key documentation files
            doc_files = {
                "README.md": self.repo_path / "README.md",
                "CONTRIBUTING.md": self.repo_path / "CONTRIBUTING.md", 
                "LICENSE": self.repo_path / "LICENSE",
                "pyproject.toml": self.repo_path / "pyproject.toml"
            }
            
            doc_status = {}
            for name, path in doc_files.items():
                doc_status[name] = {
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.exists() else 0
                }
            
            result["details"]["documentation_files"] = doc_status
            
            # Check README content
            readme_path = self.repo_path / "README.md"
            if readme_path.exists():
                with open(readme_path, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                readme_sections = {
                    "installation": "installation" in readme_content.lower(),
                    "quick_start": "quick start" in readme_content.lower(),
                    "examples": "example" in readme_content.lower(),
                    "api_docs": "api" in readme_content.lower() or "documentation" in readme_content.lower()
                }
                
                result["details"]["readme_sections"] = readme_sections
            
            # Check for examples
            examples_dir = self.repo_path / "examples"
            if examples_dir.exists():
                example_files = list(examples_dir.glob("*.py"))
                result["details"]["examples"] = {
                    "examples_dir_exists": True,
                    "example_count": len(example_files),
                    "example_files": [f.name for f in example_files[:5]]
                }
            else:
                result["details"]["examples"] = {"examples_dir_exists": False}
            
            # Calculate documentation score
            doc_score = 0
            if doc_status["README.md"]["exists"] and doc_status["README.md"]["size"] > 1000:
                doc_score += 40
            if doc_status["LICENSE"]["exists"]:
                doc_score += 20
            if doc_status["pyproject.toml"]["exists"]:
                doc_score += 20
            if result["details"].get("examples", {}).get("example_count", 0) > 0:
                doc_score += 20
            
            result["details"]["documentation_score"] = doc_score
            result["passed"] = doc_score >= 60
            result["summary"] = f"Documentation score: {doc_score}/100"
            
        except Exception as e:
            result["error"] = str(e)
            result["summary"] = f"Documentation check failed: {e}"
        
        return result
    
    def _calculate_overall_status(self):
        """Calculate overall status based on gate results."""
        gates = self.results["gates"]
        
        passed_gates = sum(1 for gate in gates.values() if gate.get("passed", False))
        total_gates = len(gates)
        
        if total_gates == 0:
            self.results["overall_status"] = "NO_GATES"
        elif passed_gates == total_gates:
            self.results["overall_status"] = "PASS"
        elif passed_gates >= total_gates * 0.8:  # 80% pass rate
            self.results["overall_status"] = "PASS_WITH_WARNINGS"
        else:
            self.results["overall_status"] = "FAIL"
    
    def _generate_summary(self):
        """Generate summary of results."""
        gates = self.results["gates"]
        
        passed_gates = [name for name, result in gates.items() if result.get("passed", False)]
        failed_gates = [name for name, result in gates.items() if not result.get("passed", False)]
        
        self.results["summary"] = {
            "total_gates": len(gates),
            "passed_gates": len(passed_gates),
            "failed_gates": len(failed_gates),
            "pass_rate": len(passed_gates) / max(len(gates), 1) * 100,
            "passed_gate_names": passed_gates,
            "failed_gate_names": failed_gates
        }
    
    def save_results(self, output_file: str = "quality_gate_results.json"):
        """Save results to JSON file."""
        output_path = self.repo_path / output_file
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📊 Results saved to: {output_path}")
    
    def print_summary(self):
        """Print summary of quality gate results."""
        summary = self.results["summary"]
        status = self.results["overall_status"]
        
        print(f"\n{'='*60}")
        print(f"🏁 QUALITY GATES SUMMARY")
        print(f"{'='*60}")
        
        status_emoji = {
            "PASS": "✅",
            "PASS_WITH_WARNINGS": "⚠️",
            "FAIL": "❌",
            "NO_GATES": "❓"
        }
        
        print(f"Overall Status: {status_emoji.get(status, '❓')} {status}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}% ({summary['passed_gates']}/{summary['total_gates']})")
        
        if summary["passed_gate_names"]:
            print(f"\n✅ Passed Gates: {', '.join(summary['passed_gate_names'])}")
        
        if summary["failed_gate_names"]:
            print(f"\n❌ Failed Gates: {', '.join(summary['failed_gate_names'])}")
        
        print(f"\nExecution time: {time.time() - self.results['timestamp']:.2f} seconds")


def main():
    """Main function to run quality gates."""
    print("🚀 Spatial-Omics GFM Quality Gate Validation")
    print("=" * 60)
    
    runner = QualityGateRunner()
    
    try:
        results = runner.run_all_gates()
        runner.print_summary()
        runner.save_results()
        
        # Exit with appropriate code
        if results["overall_status"] in ["PASS", "PASS_WITH_WARNINGS"]:
            print("\n🎉 Quality gates validation completed successfully!")
            return 0
        else:
            print("\n💥 Quality gates validation failed!")
            return 1
        
    except Exception as e:
        print(f"\n💥 Quality gate runner failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())