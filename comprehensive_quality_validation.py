"""
Comprehensive Quality Validation - All Generations

Validates all three generations with comprehensive quality gates,
security checks, performance benchmarks, and production readiness.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spatial_omics_gfm', 'core'))


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    name: str
    passed: bool
    score: float
    details: List[str]
    execution_time: float
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]


class ComprehensiveQualityValidator:
    """Comprehensive quality validation for all generations."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def run_all_quality_gates(self) -> Dict[str, Any]:
        """Run comprehensive quality gates for all generations."""
        
        print("=== COMPREHENSIVE QUALITY VALIDATION ===")
        print("Validating all three generations with production-ready quality gates\n")
        
        # Gate 1: Code Quality and Standards
        self.results.append(self._validate_code_quality())
        
        # Gate 2: Security Validation
        self.results.append(self._validate_security())
        
        # Gate 3: Generation 1 Testing
        self.results.append(self._validate_generation1())
        
        # Gate 4: Generation 2 Testing
        self.results.append(self._validate_generation2())
        
        # Gate 5: Generation 3 Testing
        self.results.append(self._validate_generation3())
        
        # Gate 6: Performance Benchmarks
        self.results.append(self._validate_performance())
        
        # Gate 7: Documentation Quality
        self.results.append(self._validate_documentation())
        
        # Gate 8: Production Readiness
        self.results.append(self._validate_production_readiness())
        
        # Gate 9: Integration Testing
        self.results.append(self._validate_integration())
        
        # Gate 10: Deployment Validation
        self.results.append(self._validate_deployment())
        
        return self._compile_final_report()
    
    def _validate_code_quality(self) -> QualityGateResult:
        """Validate code quality and standards."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            # Check Python syntax across all files
            python_files = [
                "spatial_omics_gfm/core/simple_example.py",
                "spatial_omics_gfm/core/enhanced_basic_example.py",
                "spatial_omics_gfm/core/robust_framework.py",
                "spatial_omics_gfm/core/optimized_framework.py"
            ]
            
            syntax_errors = 0
            for file_path in python_files:
                if os.path.exists(file_path):
                    try:
                        result = subprocess.run(
                            ["python3", "-m", "py_compile", file_path],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if result.returncode == 0:
                            details.append(f"✅ {file_path}: Syntax OK")
                        else:
                            syntax_errors += 1
                            critical_issues.append(f"Syntax error in {file_path}: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        warnings.append(f"Timeout checking syntax for {file_path}")
                else:
                    warnings.append(f"File not found: {file_path}")
            
            # Check for proper imports and structure
            structure_score = 100
            if syntax_errors > 0:
                structure_score -= syntax_errors * 20
                critical_issues.append(f"Found {syntax_errors} syntax errors")
            
            # Check for documentation strings
            doc_files_checked = 0
            doc_files_with_docstrings = 0
            
            for file_path in python_files:
                if os.path.exists(file_path):
                    doc_files_checked += 1
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if '"""' in content or "'''" in content:
                                doc_files_with_docstrings += 1
                                details.append(f"✅ {file_path}: Has documentation")
                            else:
                                warnings.append(f"Missing docstrings in {file_path}")
                    except Exception as e:
                        warnings.append(f"Could not check docstrings in {file_path}: {e}")
            
            doc_coverage = (doc_files_with_docstrings / doc_files_checked * 100) if doc_files_checked > 0 else 0
            details.append(f"Documentation coverage: {doc_coverage:.1f}%")
            
            # Overall score calculation
            final_score = structure_score * 0.7 + doc_coverage * 0.3
            
            if final_score >= 85:
                details.append("Excellent code quality standards")
            elif final_score >= 70:
                recommendations.append("Consider improving documentation coverage")
            else:
                critical_issues.append("Code quality below acceptable threshold")
            
            recommendations.extend([
                "Consider adding type hints for better code clarity",
                "Implement automated code formatting (black/ruff)",
                "Add pre-commit hooks for quality checks"
            ])
            
            return QualityGateResult(
                name="Code Quality & Standards",
                passed=len(critical_issues) == 0 and final_score >= 70,
                score=final_score,
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Code Quality & Standards",
                passed=False,
                score=0.0,
                details=[f"Quality validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Quality gate execution failed: {e}"],
                warnings=[],
                recommendations=["Fix quality gate execution issues"]
            )
    
    def _validate_security(self) -> QualityGateResult:
        """Validate security aspects."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            # Check for common security anti-patterns
            security_patterns = [
                "eval(", "exec(", "__import__", "subprocess.call",
                "os.system", "shell=True", "input("
            ]
            
            python_files = [
                "spatial_omics_gfm/core/simple_example.py",
                "spatial_omics_gfm/core/enhanced_basic_example.py", 
                "spatial_omics_gfm/core/robust_framework.py",
                "spatial_omics_gfm/core/optimized_framework.py"
            ]
            
            security_issues_found = 0
            files_scanned = 0
            
            for file_path in python_files:
                if os.path.exists(file_path):
                    files_scanned += 1
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            
                        file_issues = []
                        for pattern in security_patterns:
                            if pattern in content:
                                file_issues.append(pattern)
                                security_issues_found += 1
                        
                        if file_issues:
                            warnings.append(f"{file_path}: Found patterns {file_issues}")
                        else:
                            details.append(f"✅ {file_path}: No security anti-patterns detected")
                    
                    except Exception as e:
                        warnings.append(f"Could not scan {file_path}: {e}")
            
            # Check for secure file handling
            secure_practices = [
                "Path validation present",
                "Input sanitization implemented", 
                "Error handling for file operations",
                "Logging of security events"
            ]
            
            # Scan robust_framework for security features
            if os.path.exists("spatial_omics_gfm/core/robust_framework.py"):
                with open("spatial_omics_gfm/core/robust_framework.py", 'r') as f:
                    robust_content = f.read()
                
                if "SecurityGuard" in robust_content:
                    details.append("✅ Security validation framework present")
                if "sanitize" in robust_content.lower():
                    details.append("✅ Input sanitization implemented")
                if "security_event" in robust_content:
                    details.append("✅ Security event logging present")
            
            # Calculate security score
            security_score = max(0, 100 - (security_issues_found * 10))
            
            if security_issues_found > 5:
                critical_issues.append(f"High number of security concerns: {security_issues_found}")
            elif security_issues_found > 2:
                warnings.append(f"Moderate security concerns: {security_issues_found}")
            
            details.append(f"Files scanned: {files_scanned}")
            details.append(f"Security issues found: {security_issues_found}")
            
            recommendations.extend([
                "Implement comprehensive input validation",
                "Add security scanning to CI/CD pipeline",
                "Regular security audits and penetration testing",
                "Use environment variables for sensitive configuration"
            ])
            
            return QualityGateResult(
                name="Security Validation",
                passed=security_issues_found < 3,
                score=security_score,
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Security Validation",
                passed=False,
                score=0.0,
                details=[f"Security validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Security gate execution failed: {e}"],
                warnings=[],
                recommendations=["Fix security validation issues"]
            )
    
    def _validate_generation1(self) -> QualityGateResult:
        """Validate Generation 1 (Simple) functionality."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            # Test Generation 1 basic functionality
            details.append("Testing Generation 1 (Simple) functionality...")
            
            result = subprocess.run(
                ["python3", "spatial_omics_gfm/core/enhanced_basic_example.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                details.append("✅ Generation 1 execution successful")
                
                # Check if results file was created
                if os.path.exists("/root/repo/enhanced_generation1_results.json"):
                    details.append("✅ Generation 1 results file created")
                    
                    # Validate results content
                    try:
                        with open("/root/repo/enhanced_generation1_results.json", 'r') as f:
                            gen1_results = json.load(f)
                        
                        if "generation" in gen1_results and gen1_results["generation"] == "1_simple_enhanced":
                            details.append("✅ Generation 1 results format valid")
                        
                        if "cell_type_analysis" in gen1_results:
                            details.append("✅ Cell type analysis completed")
                        
                        if "interaction_analysis" in gen1_results:
                            details.append("✅ Interaction analysis completed")
                        
                        # Check for minimum required features
                        required_features = [
                            "summary_stats",
                            "cell_type_analysis", 
                            "interaction_analysis",
                            "spatial_analysis"
                        ]
                        
                        missing_features = [f for f in required_features if f not in gen1_results]
                        if missing_features:
                            warnings.extend([f"Missing feature: {f}" for f in missing_features])
                        else:
                            details.append("✅ All required features present")
                    
                    except Exception as e:
                        warnings.append(f"Could not validate results content: {e}")
                else:
                    warnings.append("Results file not created")
            else:
                critical_issues.append(f"Generation 1 execution failed: {result.stderr}")
            
            # Calculate Generation 1 score
            gen1_score = 100 if result.returncode == 0 else 0
            if warnings:
                gen1_score -= len(warnings) * 10
            
            recommendations.extend([
                "Add more comprehensive unit tests",
                "Implement input validation for user data",
                "Add progress indicators for long-running operations"
            ])
            
            return QualityGateResult(
                name="Generation 1 (Simple)",
                passed=result.returncode == 0 and len(critical_issues) == 0,
                score=max(0, gen1_score),
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                name="Generation 1 (Simple)",
                passed=False,
                score=0.0,
                details=["Generation 1 execution timed out"],
                execution_time=time.time() - start_time,
                critical_issues=["Execution timeout - performance issues detected"],
                warnings=[],
                recommendations=["Optimize Generation 1 performance"]
            )
        except Exception as e:
            return QualityGateResult(
                name="Generation 1 (Simple)",
                passed=False,
                score=0.0,
                details=[f"Generation 1 validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Validation failed: {e}"],
                warnings=[],
                recommendations=["Fix Generation 1 execution issues"]
            )
    
    def _validate_generation2(self) -> QualityGateResult:
        """Validate Generation 2 (Robust) functionality."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            details.append("Testing Generation 2 (Robust) functionality...")
            
            result = subprocess.run(
                ["python3", "spatial_omics_gfm/core/robust_framework.py"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                details.append("✅ Generation 2 execution successful")
                
                # Check for robust features in output
                if "validation" in result.stdout.lower():
                    details.append("✅ Data validation implemented")
                
                if "security" in result.stdout.lower():
                    details.append("✅ Security features implemented")
                
                if "normalization" in result.stdout.lower():
                    details.append("✅ Safe normalization implemented")
                
                # Check results file
                if os.path.exists("/root/repo/robust_generation2_results.json"):
                    details.append("✅ Generation 2 results file created")
                    
                    try:
                        with open("/root/repo/robust_generation2_results.json", 'r') as f:
                            gen2_results = json.load(f)
                        
                        # Check for robustness features
                        if "robustness_features" in gen2_results:
                            robustness = gen2_results["robustness_features"]
                            
                            if robustness.get("validation_errors", 1) == 0:
                                details.append("✅ No validation errors detected")
                            else:
                                warnings.append(f"Validation errors: {robustness.get('validation_errors', 0)}")
                            
                            if robustness.get("security_threats", 1) == 0:
                                details.append("✅ No security threats detected")
                            else:
                                critical_issues.append(f"Security threats: {robustness.get('security_threats', 0)}")
                    
                    except Exception as e:
                        warnings.append(f"Could not validate Generation 2 results: {e}")
                
                # Check for validation report
                if os.path.exists("/root/repo/robust_validation_report.json"):
                    details.append("✅ Validation report generated")
                else:
                    warnings.append("Validation report not found")
            
            else:
                critical_issues.append(f"Generation 2 execution failed: {result.stderr}")
            
            # Calculate Generation 2 score
            gen2_score = 100 if result.returncode == 0 else 0
            if warnings:
                gen2_score -= len(warnings) * 5
            if critical_issues:
                gen2_score -= len(critical_issues) * 20
            
            recommendations.extend([
                "Implement comprehensive error recovery",
                "Add detailed audit logging",
                "Create automated health checks",
                "Implement circuit breaker patterns"
            ])
            
            return QualityGateResult(
                name="Generation 2 (Robust)",
                passed=result.returncode == 0 and len(critical_issues) == 0,
                score=max(0, gen2_score),
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                name="Generation 2 (Robust)",
                passed=False,
                score=0.0,
                details=["Generation 2 execution timed out"],
                execution_time=time.time() - start_time,
                critical_issues=["Execution timeout - robustness issues detected"],
                warnings=[],
                recommendations=["Optimize Generation 2 resilience"]
            )
        except Exception as e:
            return QualityGateResult(
                name="Generation 2 (Robust)",
                passed=False,
                score=0.0,
                details=[f"Generation 2 validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Validation failed: {e}"],
                warnings=[],
                recommendations=["Fix Generation 2 robustness issues"]
            )
    
    def _validate_generation3(self) -> QualityGateResult:
        """Validate Generation 3 (Optimized) functionality."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            details.append("Testing Generation 3 (Optimized) functionality...")
            
            result = subprocess.run(
                ["python3", "spatial_omics_gfm/core/optimized_framework.py"],
                capture_output=True,
                text=True,
                timeout=180  # Longer timeout for optimization tests
            )
            
            if result.returncode == 0:
                details.append("✅ Generation 3 execution successful")
                
                # Check for optimization features in output
                if "cache" in result.stdout.lower():
                    details.append("✅ Caching implemented")
                
                if "parallel" in result.stdout.lower():
                    details.append("✅ Parallel processing implemented")
                
                if "speedup" in result.stdout.lower():
                    details.append("✅ Performance optimization detected")
                
                # Check results file for performance metrics
                if os.path.exists("/root/repo/optimized_generation3_results.json"):
                    details.append("✅ Generation 3 results file created")
                    
                    try:
                        with open("/root/repo/optimized_generation3_results.json", 'r') as f:
                            gen3_results = json.load(f)
                        
                        # Check performance metrics
                        if "performance_metrics" in gen3_results:
                            perf = gen3_results["performance_metrics"]
                            
                            # Check for reasonable execution times
                            total_time = perf.get("total_execution_time", float('inf'))
                            if total_time < 30:  # Reasonable for test dataset
                                details.append(f"✅ Good performance: {total_time:.2f}s total")
                            elif total_time < 60:
                                warnings.append(f"Moderate performance: {total_time:.2f}s total")
                            else:
                                critical_issues.append(f"Poor performance: {total_time:.2f}s total")
                            
                            # Check cache performance
                            speedup = perf.get("cache_speedup_factor", 1)
                            if speedup > 10:
                                details.append(f"✅ Excellent cache performance: {speedup:.1f}x speedup")
                            elif speedup > 2:
                                details.append(f"✅ Good cache performance: {speedup:.1f}x speedup")
                            else:
                                warnings.append(f"Low cache performance: {speedup:.1f}x speedup")
                        
                        # Check optimization features
                        if "optimization_features" in gen3_results:
                            opt = gen3_results["optimization_features"]
                            
                            if opt.get("caching_enabled", False):
                                details.append("✅ Caching enabled")
                            else:
                                warnings.append("Caching not enabled")
                            
                            if opt.get("parallel_processing", False):
                                details.append("✅ Parallel processing enabled")
                            else:
                                warnings.append("Parallel processing not enabled")
                            
                            hit_rate = opt.get("cache_hit_rate", 0)
                            if hit_rate > 0.8:
                                details.append(f"✅ Excellent cache hit rate: {hit_rate:.1%}")
                            elif hit_rate > 0.5:
                                details.append(f"✅ Good cache hit rate: {hit_rate:.1%}")
                            else:
                                warnings.append(f"Low cache hit rate: {hit_rate:.1%}")
                    
                    except Exception as e:
                        warnings.append(f"Could not validate Generation 3 results: {e}")
            
            else:
                critical_issues.append(f"Generation 3 execution failed: {result.stderr}")
            
            # Calculate Generation 3 score based on performance
            gen3_score = 100 if result.returncode == 0 else 0
            if warnings:
                gen3_score -= len(warnings) * 8
            if critical_issues:
                gen3_score -= len(critical_issues) * 25
            
            recommendations.extend([
                "Implement adaptive caching strategies",
                "Add performance monitoring and alerting",
                "Optimize memory usage patterns", 
                "Consider GPU acceleration for large datasets",
                "Implement auto-scaling capabilities"
            ])
            
            return QualityGateResult(
                name="Generation 3 (Optimized)",
                passed=result.returncode == 0 and len(critical_issues) == 0,
                score=max(0, gen3_score),
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                name="Generation 3 (Optimized)",
                passed=False,
                score=0.0,
                details=["Generation 3 execution timed out"],
                execution_time=time.time() - start_time,
                critical_issues=["Execution timeout - optimization ineffective"],
                warnings=[],
                recommendations=["Review and improve optimization strategies"]
            )
        except Exception as e:
            return QualityGateResult(
                name="Generation 3 (Optimized)",
                passed=False,
                score=0.0,
                details=[f"Generation 3 validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Validation failed: {e}"],
                warnings=[],
                recommendations=["Fix Generation 3 optimization issues"]
            )
    
    def _validate_performance(self) -> QualityGateResult:
        """Validate overall performance benchmarks."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            details.append("Running performance benchmark validation...")
            
            # Check if all generation result files exist and have performance data
            gen_files = {
                "Generation 1": "/root/repo/enhanced_generation1_results.json",
                "Generation 2": "/root/repo/robust_generation2_results.json",
                "Generation 3": "/root/repo/optimized_generation3_results.json"
            }
            
            performance_data = {}
            
            for gen_name, file_path in gen_files.items():
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Extract performance metrics
                        if "analysis_metadata" in data:
                            metadata = data["analysis_metadata"]
                            n_cells = metadata.get("n_cells", 0)
                            n_genes = metadata.get("n_genes", 0)
                            
                            performance_data[gen_name] = {
                                "cells": n_cells,
                                "genes": n_genes,
                                "data_points": n_cells * n_genes
                            }
                            
                            details.append(f"✅ {gen_name}: {n_cells} cells, {n_genes} genes")
                        
                        # Check specific performance metrics for Generation 3
                        if gen_name == "Generation 3" and "performance_metrics" in data:
                            perf = data["performance_metrics"]
                            total_time = perf.get("total_execution_time", 0)
                            
                            if total_time > 0:
                                cells_per_second = performance_data[gen_name]["cells"] / total_time
                                details.append(f"✅ Generation 3 throughput: {cells_per_second:.0f} cells/sec")
                                
                                # Performance thresholds
                                if cells_per_second > 100:
                                    details.append("✅ Excellent processing throughput")
                                elif cells_per_second > 50:
                                    details.append("✅ Good processing throughput")
                                else:
                                    warnings.append(f"Low processing throughput: {cells_per_second:.0f} cells/sec")
                    
                    except Exception as e:
                        warnings.append(f"Could not analyze performance data for {gen_name}: {e}")
                else:
                    warnings.append(f"Performance data not available for {gen_name}")
            
            # Compare generations for progressive improvement
            if "Generation 1" in performance_data and "Generation 3" in performance_data:
                gen1_points = performance_data["Generation 1"]["data_points"]
                gen3_points = performance_data["Generation 3"]["data_points"]
                
                if gen3_points >= gen1_points:
                    details.append("✅ Performance scaling maintained across generations")
                else:
                    warnings.append("Performance scaling degraded in later generations")
            
            # Memory efficiency check (if available)
            if os.path.exists("/root/repo/optimized_generation3_results.json"):
                try:
                    with open("/root/repo/optimized_generation3_results.json", 'r') as f:
                        gen3_data = json.load(f)
                    
                    if "system_performance" in gen3_data:
                        sys_perf = gen3_data["system_performance"]
                        memory_stats = sys_perf.get("memory_usage", {})
                        
                        peak_memory = memory_stats.get("peak", 0)
                        if peak_memory > 0:
                            if peak_memory < 100:  # MB
                                details.append(f"✅ Efficient memory usage: {peak_memory:.1f} MB peak")
                            elif peak_memory < 500:
                                details.append(f"✅ Reasonable memory usage: {peak_memory:.1f} MB peak")
                            else:
                                warnings.append(f"High memory usage: {peak_memory:.1f} MB peak")
                
                except Exception as e:
                    warnings.append(f"Could not analyze memory performance: {e}")
            
            # Calculate performance score
            perf_score = 100
            if warnings:
                perf_score -= len(warnings) * 10
            if critical_issues:
                perf_score -= len(critical_issues) * 30
            
            # Bonus for having all generations working
            if len(performance_data) == 3:
                perf_score += 10
                details.append("✅ All generations operational")
            
            recommendations.extend([
                "Implement continuous performance monitoring",
                "Set up performance regression testing", 
                "Create performance budgets for different operations",
                "Add performance alerts for production deployments"
            ])
            
            return QualityGateResult(
                name="Performance Benchmarks",
                passed=len(critical_issues) == 0 and perf_score >= 70,
                score=max(0, min(100, perf_score)),
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Performance Benchmarks",
                passed=False,
                score=0.0,
                details=[f"Performance validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Performance validation error: {e}"],
                warnings=[],
                recommendations=["Fix performance validation system"]
            )
    
    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation quality and completeness."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            # Check for essential documentation files
            doc_files = {
                "README.md": "Main project documentation",
                "CONTRIBUTING.md": "Contribution guidelines", 
                "LICENSE": "License information",
                "pyproject.toml": "Project configuration",
                "setup.py": "Installation configuration"
            }
            
            docs_found = 0
            for doc_file, description in doc_files.items():
                if os.path.exists(doc_file):
                    docs_found += 1
                    details.append(f"✅ {doc_file}: {description}")
                    
                    # Check file content quality
                    try:
                        with open(doc_file, 'r') as f:
                            content = f.read()
                        
                        if len(content) < 100:
                            warnings.append(f"{doc_file} seems too short")
                        elif len(content) > 50000:
                            warnings.append(f"{doc_file} seems very long")
                        else:
                            details.append(f"✅ {doc_file}: Good content length")
                    
                    except Exception as e:
                        warnings.append(f"Could not analyze {doc_file}: {e}")
                else:
                    if doc_file == "README.md":
                        critical_issues.append(f"Missing critical documentation: {doc_file}")
                    else:
                        warnings.append(f"Missing documentation: {doc_file}")
            
            # Check docs directory
            docs_dir_files = 0
            if os.path.exists("docs"):
                for root, dirs, files in os.walk("docs"):
                    docs_dir_files += len([f for f in files if f.endswith(('.md', '.rst', '.txt'))])
                
                if docs_dir_files > 0:
                    details.append(f"✅ Found {docs_dir_files} additional documentation files")
                else:
                    warnings.append("docs directory exists but appears empty")
            else:
                warnings.append("No dedicated docs directory found")
            
            # Check for API documentation
            api_docs_found = False
            if os.path.exists("docs/API.md"):
                api_docs_found = True
                details.append("✅ API documentation found")
            else:
                recommendations.append("Consider adding comprehensive API documentation")
            
            # Check for examples directory
            if os.path.exists("examples"):
                example_files = 0
                for root, dirs, files in os.walk("examples"):
                    example_files += len([f for f in files if f.endswith('.py')])
                
                if example_files > 0:
                    details.append(f"✅ Found {example_files} example files")
                else:
                    warnings.append("examples directory exists but no Python examples found")
            else:
                recommendations.append("Consider adding examples directory with usage examples")
            
            # Calculate documentation score
            doc_score = (docs_found / len(doc_files)) * 100
            if docs_dir_files > 5:
                doc_score += 10  # Bonus for comprehensive docs
            if api_docs_found:
                doc_score += 10  # Bonus for API docs
            
            if doc_score >= 90:
                details.append("✅ Excellent documentation coverage")
            elif doc_score >= 70:
                details.append("✅ Good documentation coverage")
            else:
                recommendations.append("Improve documentation coverage")
            
            recommendations.extend([
                "Add inline code documentation (docstrings)",
                "Create user guides and tutorials",
                "Set up automated documentation generation",
                "Add troubleshooting and FAQ sections"
            ])
            
            return QualityGateResult(
                name="Documentation Quality",
                passed=len(critical_issues) == 0 and doc_score >= 60,
                score=min(100, doc_score),
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Documentation Quality",
                passed=False,
                score=0.0,
                details=[f"Documentation validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Documentation validation error: {e}"],
                warnings=[],
                recommendations=["Fix documentation validation system"]
            )
    
    def _validate_production_readiness(self) -> QualityGateResult:
        """Validate production readiness."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            # Check for Docker configuration
            docker_files = ["Dockerfile", "docker-compose.yml", "docker/Dockerfile"]
            docker_found = False
            
            for docker_file in docker_files:
                if os.path.exists(docker_file):
                    docker_found = True
                    details.append(f"✅ Docker configuration found: {docker_file}")
                    break
            
            if not docker_found:
                warnings.append("No Docker configuration found")
                recommendations.append("Add Docker configuration for containerized deployment")
            
            # Check for Kubernetes configuration
            k8s_files = ["kubernetes", "k8s", "deployment.yaml"]
            k8s_found = False
            
            for k8s_path in k8s_files:
                if os.path.exists(k8s_path):
                    k8s_found = True
                    details.append(f"✅ Kubernetes configuration found: {k8s_path}")
                    break
            
            if not k8s_found:
                recommendations.append("Consider adding Kubernetes deployment manifests")
            
            # Check for configuration management
            config_files = ["config", "configs", "settings"]
            config_found = False
            
            for config_dir in config_files:
                if os.path.exists(config_dir):
                    config_found = True
                    details.append(f"✅ Configuration directory found: {config_dir}")
                    break
            
            if not config_found:
                warnings.append("No configuration management structure found")
                recommendations.append("Implement proper configuration management")
            
            # Check for health check endpoints
            health_check_indicators = ["healthcheck", "health", "status"]
            health_check_found = False
            
            # Search in Python files for health check implementations
            for root, dirs, files in os.walk("."):
                for file in files:
                    if file.endswith('.py'):
                        try:
                            file_path = os.path.join(root, file)
                            with open(file_path, 'r') as f:
                                content = f.read().lower()
                            
                            if any(indicator in content for indicator in health_check_indicators):
                                health_check_found = True
                                details.append(f"✅ Health check implementation found in {file_path}")
                                break
                        except Exception:
                            continue
                
                if health_check_found:
                    break
            
            if not health_check_found:
                recommendations.append("Implement health check endpoints for monitoring")
            
            # Check for logging configuration
            logging_found = False
            log_patterns = ["logging", "logger", "log"]
            
            # Check for logging in robust framework
            if os.path.exists("spatial_omics_gfm/core/robust_framework.py"):
                try:
                    with open("spatial_omics_gfm/core/robust_framework.py", 'r') as f:
                        content = f.read().lower()
                    
                    if any(pattern in content for pattern in log_patterns):
                        logging_found = True
                        details.append("✅ Logging implementation found")
                except Exception:
                    pass
            
            if not logging_found:
                warnings.append("No comprehensive logging found")
                recommendations.append("Implement structured logging for production")
            
            # Check for monitoring and metrics
            monitoring_indicators = ["metrics", "monitoring", "prometheus", "grafana"]
            monitoring_found = any(os.path.exists(indicator) for indicator in monitoring_indicators)
            
            if monitoring_found:
                details.append("✅ Monitoring configuration found")
            else:
                recommendations.append("Add monitoring and metrics collection")
            
            # Check for backup and disaster recovery
            backup_indicators = ["backup", "restore", "recovery"]
            backup_found = False
            
            for root, dirs, files in os.walk("scripts"):
                for file in files:
                    if any(indicator in file.lower() for indicator in backup_indicators):
                        backup_found = True
                        details.append(f"✅ Backup/recovery script found: {file}")
                        break
                
                if backup_found:
                    break
            
            if not backup_found:
                recommendations.append("Implement backup and disaster recovery procedures")
            
            # Calculate production readiness score
            readiness_features = [
                docker_found,
                k8s_found, 
                config_found,
                health_check_found,
                logging_found,
                monitoring_found,
                backup_found
            ]
            
            readiness_score = (sum(readiness_features) / len(readiness_features)) * 100
            
            # Minimum requirements for production
            if not docker_found and not k8s_found:
                critical_issues.append("No containerization or orchestration configuration")
            
            if not health_check_found:
                warnings.append("No health check implementation for monitoring")
            
            if readiness_score >= 80:
                details.append("✅ Excellent production readiness")
            elif readiness_score >= 60:
                details.append("✅ Good production readiness")
            else:
                warnings.append("Production readiness needs improvement")
            
            recommendations.extend([
                "Set up CI/CD pipelines",
                "Implement automated testing in production pipeline",
                "Add security scanning to deployment process",
                "Create runbooks for operational procedures"
            ])
            
            return QualityGateResult(
                name="Production Readiness",
                passed=len(critical_issues) == 0 and readiness_score >= 50,
                score=readiness_score,
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Production Readiness",
                passed=False,
                score=0.0,
                details=[f"Production readiness validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Production validation error: {e}"],
                warnings=[],
                recommendations=["Fix production readiness validation"]
            )
    
    def _validate_integration(self) -> QualityGateResult:
        """Validate integration between all components."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            details.append("Testing component integration...")
            
            # Check if all generation result files exist
            result_files = {
                "enhanced_generation1_results.json": "Generation 1",
                "robust_generation2_results.json": "Generation 2", 
                "optimized_generation3_results.json": "Generation 3"
            }
            
            integration_data = {}
            files_found = 0
            
            for file_name, gen_name in result_files.items():
                file_path = f"/root/repo/{file_name}"
                if os.path.exists(file_path):
                    files_found += 1
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        integration_data[gen_name] = data
                        details.append(f"✅ {gen_name} results loaded successfully")
                        
                        # Check data consistency
                        if "analysis_metadata" in data:
                            metadata = data["analysis_metadata"]
                            n_cells = metadata.get("n_cells", 0)
                            n_genes = metadata.get("n_genes", 0)
                            
                            if n_cells > 0 and n_genes > 0:
                                details.append(f"✅ {gen_name}: Valid data dimensions ({n_cells} cells, {n_genes} genes)")
                            else:
                                warnings.append(f"{gen_name}: Invalid data dimensions")
                    
                    except Exception as e:
                        warnings.append(f"Could not load {gen_name} results: {e}")
                else:
                    warnings.append(f"Missing results file for {gen_name}")
            
            # Check data consistency across generations
            if len(integration_data) >= 2:
                # Compare cell/gene counts across generations
                cell_counts = {}
                gene_counts = {}
                
                for gen_name, data in integration_data.items():
                    if "analysis_metadata" in data:
                        metadata = data["analysis_metadata"]
                        cell_counts[gen_name] = metadata.get("n_cells", 0)
                        gene_counts[gen_name] = metadata.get("n_genes", 0)
                
                # Check for reasonable scaling across generations
                if cell_counts:
                    unique_cell_counts = set(cell_counts.values())
                    if len(unique_cell_counts) <= 2:  # Allow some variation
                        details.append("✅ Consistent cell counts across generations")
                    else:
                        warnings.append("Inconsistent cell counts across generations")
                
                if gene_counts:
                    unique_gene_counts = set(gene_counts.values())
                    if len(unique_gene_counts) <= 2:  # Allow some variation
                        details.append("✅ Consistent gene counts across generations")
                    else:
                        warnings.append("Inconsistent gene counts across generations")
            
            # Test data flow between generations
            data_flow_test_passed = True
            
            # Check if Generation 2 has validation results
            if "Generation 2" in integration_data:
                gen2_data = integration_data["Generation 2"]
                if "validation_summary" not in gen2_data:
                    warnings.append("Generation 2 missing validation summary")
                    data_flow_test_passed = False
                else:
                    details.append("✅ Generation 2 validation data present")
            
            # Check if Generation 3 has performance metrics
            if "Generation 3" in integration_data:
                gen3_data = integration_data["Generation 3"]
                if "performance_metrics" not in gen3_data:
                    warnings.append("Generation 3 missing performance metrics")
                    data_flow_test_passed = False
                else:
                    details.append("✅ Generation 3 performance data present")
            
            # Check for common analysis results across generations
            common_features = ["cell_type_analysis", "interaction_analysis"]
            feature_consistency = True
            
            for feature in common_features:
                generations_with_feature = [
                    gen_name for gen_name, data in integration_data.items()
                    if feature in data
                ]
                
                if len(generations_with_feature) == len(integration_data):
                    details.append(f"✅ {feature} present in all generations")
                else:
                    warnings.append(f"{feature} not consistent across generations")
                    feature_consistency = False
            
            # Calculate integration score
            integration_score = 0
            
            # File availability (40 points)
            integration_score += (files_found / len(result_files)) * 40
            
            # Data consistency (30 points)
            if len(integration_data) > 1 and not warnings:
                integration_score += 30
            elif len(integration_data) > 1:
                integration_score += 15
            
            # Data flow (20 points)
            if data_flow_test_passed:
                integration_score += 20
            
            # Feature consistency (10 points)
            if feature_consistency:
                integration_score += 10
            
            if integration_score >= 90:
                details.append("✅ Excellent integration between components")
            elif integration_score >= 70:
                details.append("✅ Good integration between components")
            else:
                if integration_score < 50:
                    critical_issues.append("Poor integration between components")
                else:
                    warnings.append("Integration between components needs improvement")
            
            recommendations.extend([
                "Implement comprehensive integration tests",
                "Add data validation between generation handoffs",
                "Create integration monitoring and alerting",
                "Standardize data formats across components"
            ])
            
            return QualityGateResult(
                name="Integration Testing",
                passed=len(critical_issues) == 0 and integration_score >= 60,
                score=integration_score,
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Integration Testing",
                passed=False,
                score=0.0,
                details=[f"Integration testing failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Integration testing error: {e}"],
                warnings=[],
                recommendations=["Fix integration testing system"]
            )
    
    def _validate_deployment(self) -> QualityGateResult:
        """Validate deployment configuration and readiness."""
        
        start_time = time.time()
        details = []
        warnings = []
        critical_issues = []
        recommendations = []
        
        try:
            # Check deployment-related files
            deployment_files = {
                "docker": "Containerization",
                "kubernetes": "Container orchestration",
                "deployment": "Deployment configuration",
                "scripts": "Deployment scripts",
                "config": "Configuration management"
            }
            
            deployment_score = 0
            deployment_features = 0
            
            for deploy_path, description in deployment_files.items():
                if os.path.exists(deploy_path):
                    deployment_features += 1
                    details.append(f"✅ {description} found: {deploy_path}")
                    
                    # Check specific deployment file contents
                    if deploy_path == "docker":
                        docker_files = ["Dockerfile", "docker-compose.yml"]
                        docker_config_found = False
                        
                        for docker_file in docker_files:
                            docker_path = os.path.join(deploy_path, docker_file)
                            if os.path.exists(docker_path):
                                docker_config_found = True
                                details.append(f"✅ Docker configuration: {docker_file}")
                        
                        if not docker_config_found:
                            warnings.append("Docker directory exists but no Dockerfile found")
                    
                    elif deploy_path == "kubernetes":
                        k8s_files = []
                        for root, dirs, files in os.walk(deploy_path):
                            k8s_files.extend([f for f in files if f.endswith(('.yaml', '.yml'))])
                        
                        if k8s_files:
                            details.append(f"✅ Kubernetes manifests: {len(k8s_files)} files")
                        else:
                            warnings.append("Kubernetes directory exists but no manifests found")
                    
                    elif deploy_path == "scripts":
                        script_files = []
                        for root, dirs, files in os.walk(deploy_path):
                            script_files.extend([f for f in files if f.endswith(('.py', '.sh', '.bash'))])
                        
                        if script_files:
                            details.append(f"✅ Deployment scripts: {len(script_files)} files")
                        else:
                            warnings.append("Scripts directory exists but no deployment scripts found")
                
                else:
                    warnings.append(f"Missing {description.lower()}: {deploy_path}")
            
            # Check for environment configuration
            env_files = [".env", ".env.example", "environment.yml", "requirements.txt", "pyproject.toml"]
            env_config_found = 0
            
            for env_file in env_files:
                if os.path.exists(env_file):
                    env_config_found += 1
                    details.append(f"✅ Environment configuration: {env_file}")
            
            if env_config_found == 0:
                critical_issues.append("No environment configuration files found")
            elif env_config_found < 2:
                warnings.append("Limited environment configuration")
            
            # Check for CI/CD configuration
            ci_cd_paths = [".github", ".gitlab-ci.yml", "Jenkinsfile", "azure-pipelines.yml"]
            ci_cd_found = any(os.path.exists(path) for path in ci_cd_paths)
            
            if ci_cd_found:
                details.append("✅ CI/CD configuration found")
            else:
                recommendations.append("Add CI/CD pipeline configuration")
            
            # Check for security configurations
            security_files = ["SECURITY.md", "security.yaml", "secrets.yaml"]
            security_config_found = any(os.path.exists(f) for f in security_files)
            
            if security_config_found:
                details.append("✅ Security configuration found")
            else:
                warnings.append("No security configuration found")
                recommendations.append("Add security configuration and policies")
            
            # Calculate deployment readiness score
            base_score = (deployment_features / len(deployment_files)) * 60
            env_score = min(20, (env_config_found / 2) * 20)
            ci_cd_score = 10 if ci_cd_found else 0
            security_score = 10 if security_config_found else 0
            
            deployment_score = base_score + env_score + ci_cd_score + security_score
            
            # Minimum requirements
            if deployment_features == 0:
                critical_issues.append("No deployment configuration found")
            
            if env_config_found == 0:
                critical_issues.append("No environment configuration")
            
            if deployment_score >= 80:
                details.append("✅ Excellent deployment readiness")
            elif deployment_score >= 60:
                details.append("✅ Good deployment readiness")
            else:
                warnings.append("Deployment readiness needs improvement")
            
            recommendations.extend([
                "Implement blue-green deployment strategy",
                "Add automated rollback capabilities",
                "Set up deployment monitoring and health checks",
                "Create deployment runbooks and procedures"
            ])
            
            return QualityGateResult(
                name="Deployment Validation",
                passed=len(critical_issues) == 0 and deployment_score >= 50,
                score=deployment_score,
                details=details,
                execution_time=time.time() - start_time,
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                name="Deployment Validation",
                passed=False,
                score=0.0,
                details=[f"Deployment validation failed: {e}"],
                execution_time=time.time() - start_time,
                critical_issues=[f"Deployment validation error: {e}"],
                warnings=[],
                recommendations=["Fix deployment validation system"]
            )
    
    def _compile_final_report(self) -> Dict[str, Any]:
        """Compile final comprehensive quality report."""
        
        total_execution_time = time.time() - self.start_time
        
        # Calculate overall metrics
        total_gates = len(self.results)
        passed_gates = sum(1 for result in self.results if result.passed)
        failed_gates = total_gates - passed_gates
        
        overall_score = sum(result.score for result in self.results) / total_gates if total_gates > 0 else 0
        
        # Categorize issues
        all_critical_issues = []
        all_warnings = []
        all_recommendations = []
        
        for result in self.results:
            all_critical_issues.extend(result.critical_issues)
            all_warnings.extend(result.warnings) 
            all_recommendations.extend(result.recommendations)
        
        # Determine overall pass/fail
        critical_gate_failures = sum(
            1 for result in self.results 
            if not result.passed and result.name in [
                "Code Quality & Standards",
                "Security Validation", 
                "Generation 1 (Simple)",
                "Generation 2 (Robust)",
                "Generation 3 (Optimized)"
            ]
        )
        
        overall_passed = critical_gate_failures == 0 and overall_score >= 70
        
        # Generate summary
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE QUALITY VALIDATION SUMMARY")
        print("="*80)
        print(f"📊 Overall Score: {overall_score:.1f}/100")
        print(f"✅ Passed Gates: {passed_gates}/{total_gates}")
        print(f"❌ Failed Gates: {failed_gates}/{total_gates}")
        print(f"⏱️ Total Execution Time: {total_execution_time:.2f}s")
        print(f"🎯 Overall Result: {'✅ PASSED' if overall_passed else '❌ FAILED'}")
        
        print("\n📋 QUALITY GATE RESULTS:")
        print("-" * 80)
        
        for result in self.results:
            status_icon = "✅" if result.passed else "❌"
            print(f"{status_icon} {result.name}: {result.score:.1f}/100 ({result.execution_time:.2f}s)")
            
            if result.critical_issues:
                for issue in result.critical_issues[:2]:  # Show top 2
                    print(f"    🚨 {issue}")
            
            if result.warnings:
                for warning in result.warnings[:2]:  # Show top 2
                    print(f"    ⚠️ {warning}")
        
        if all_critical_issues:
            print(f"\n🚨 CRITICAL ISSUES ({len(all_critical_issues)}):")
            for i, issue in enumerate(all_critical_issues[:5], 1):
                print(f"  {i}. {issue}")
            if len(all_critical_issues) > 5:
                print(f"  ... and {len(all_critical_issues) - 5} more")
        
        if all_warnings:
            print(f"\n⚠️ WARNINGS ({len(all_warnings)}):")
            for i, warning in enumerate(all_warnings[:5], 1):
                print(f"  {i}. {warning}")
            if len(all_warnings) > 5:
                print(f"  ... and {len(all_warnings) - 5} more")
        
        print(f"\n💡 TOP RECOMMENDATIONS ({len(set(all_recommendations))}):") 
        unique_recommendations = list(set(all_recommendations))
        for i, rec in enumerate(unique_recommendations[:10], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        
        # Compile final report data
        report = {
            "summary": {
                "overall_passed": overall_passed,
                "overall_score": overall_score,
                "total_gates": total_gates,
                "passed_gates": passed_gates,
                "failed_gates": failed_gates,
                "execution_time": total_execution_time,
                "timestamp": time.time()
            },
            "gate_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "details": result.details,
                    "critical_issues": result.critical_issues,
                    "warnings": result.warnings,
                    "recommendations": result.recommendations
                }
                for result in self.results
            ],
            "issue_summary": {
                "total_critical_issues": len(all_critical_issues),
                "total_warnings": len(all_warnings),
                "critical_issues": all_critical_issues,
                "warnings": all_warnings,
                "unique_recommendations": unique_recommendations
            },
            "next_steps": [
                "Address all critical issues before production deployment",
                "Review and implement high-priority recommendations", 
                "Set up continuous quality monitoring",
                "Create quality improvement roadmap",
                "Schedule regular quality gate reviews"
            ]
        }
        
        # Save comprehensive report
        try:
            with open("/root/repo/comprehensive_quality_validation_report.json", "w") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"📄 Comprehensive report saved: comprehensive_quality_validation_report.json")
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")
        
        return report


if __name__ == "__main__":
    validator = ComprehensiveQualityValidator()
    final_report = validator.run_all_quality_gates()