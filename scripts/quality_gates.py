#!/usr/bin/env python3
"""
Quality Gates Script for Spatial-Omics GFM

Runs comprehensive quality checks without requiring full dependency installation.
"""

import ast
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import json


class QualityGates:
    """Comprehensive quality gates for the project."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {}
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🔍 Running Quality Gates for Spatial-Omics GFM")
        print("=" * 50)
        
        gates = [
            ("syntax_check", self.check_syntax),
            ("import_structure", self.check_import_structure),
            ("type_hints", self.check_type_hints),
            ("code_quality", self.check_code_quality),
            ("security_scan", self.check_security),
            ("documentation", self.check_documentation),
            ("test_coverage", self.analyze_test_coverage),
            ("performance", self.check_performance_patterns),
        ]
        
        all_passed = True
        for gate_name, gate_func in gates:
            print(f"\n📋 Running {gate_name.replace('_', ' ').title()}...")
            try:
                result = gate_func()
                self.results[gate_name] = result
                if result.get("passed", False):
                    print(f"✅ {gate_name} PASSED")
                else:
                    print(f"❌ {gate_name} FAILED")
                    all_passed = False
                    if "errors" in result:
                        for error in result["errors"][:3]:  # Show first 3 errors
                            print(f"   • {error}")
                        if len(result["errors"]) > 3:
                            print(f"   • ... and {len(result['errors'])-3} more")
            except Exception as e:
                print(f"❌ {gate_name} ERROR: {e}")
                all_passed = False
                self.results[gate_name] = {"passed": False, "error": str(e)}
        
        self.results["overall_passed"] = all_passed
        
        print("\n" + "=" * 50)
        if all_passed:
            print("🎉 ALL QUALITY GATES PASSED!")
        else:
            print("⚠️  Some quality gates failed. See details above.")
        
        return self.results
    
    def check_syntax(self) -> Dict[str, Any]:
        """Check Python syntax for all files."""
        python_files = list(self.project_root.glob("**/*.py"))
        errors = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                ast.parse(source, filename=str(file_path))
            except SyntaxError as e:
                errors.append(f"{file_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
        
        return {
            "passed": not errors,
            "total_files": len(python_files),
            "errors": errors,
            "metrics": {
                "syntax_errors": len(errors),
                "files_checked": len(python_files)
            }
        }
    
    def check_import_structure(self) -> Dict[str, Any]:
        """Verify import structure and dependencies."""
        issues = []
        
        # Check main __init__.py
        init_file = self.project_root / "spatial_omics_gfm" / "__init__.py"
        if not init_file.exists():
            issues.append("Missing main __init__.py file")
        else:
            try:
                with open(init_file) as f:
                    content = f.read()
                    
                # Check for required exports
                required_exports = [
                    "SpatialGraphTransformer", "VisiumDataset", "CellTypeClassifier",
                    "InteractionPredictor", "PathwayAnalyzer", "FineTuner"
                ]
                
                for export in required_exports:
                    if export not in content:
                        issues.append(f"Missing export: {export} in __init__.py")
            except Exception as e:
                issues.append(f"Error reading __init__.py: {e}")
        
        # Check module structure
        required_modules = [
            "models", "data", "tasks", "training", 
            "inference", "visualization", "utils"
        ]
        
        for module in required_modules:
            module_path = self.project_root / "spatial_omics_gfm" / module
            if not module_path.exists():
                issues.append(f"Missing module directory: {module}")
            else:
                init_path = module_path / "__init__.py"
                if not init_path.exists():
                    issues.append(f"Missing __init__.py in {module}")
        
        return {
            "passed": not issues,
            "errors": issues,
            "metrics": {
                "modules_checked": len(required_modules),
                "structure_issues": len(issues)
            }
        }
    
    def check_type_hints(self) -> Dict[str, Any]:
        """Check type hint coverage."""
        python_files = [
            f for f in self.project_root.glob("spatial_omics_gfm/**/*.py")
            if not f.name.startswith("__")
        ]
        
        total_functions = 0
        typed_functions = 0
        issues = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check if function has type hints
                        has_return_annotation = node.returns is not None
                        has_arg_annotations = any(
                            arg.annotation is not None 
                            for arg in node.args.args 
                            if arg.arg != 'self'
                        )
                        
                        if has_return_annotation or has_arg_annotations:
                            typed_functions += 1
                        else:
                            issues.append(
                                f"{file_path.relative_to(self.project_root)}:"
                                f"{node.lineno}: Function '{node.name}' lacks type hints"
                            )
                            
            except Exception as e:
                issues.append(f"Error analyzing {file_path}: {e}")
        
        coverage = (typed_functions / total_functions * 100) if total_functions > 0 else 0
        
        return {
            "passed": coverage >= 85,  # 85% type hint coverage required
            "errors": issues if coverage < 85 else [],
            "metrics": {
                "total_functions": total_functions,
                "typed_functions": typed_functions,
                "coverage_percentage": round(coverage, 2)
            }
        }
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        python_files = list(self.project_root.glob("spatial_omics_gfm/**/*.py"))
        issues = []
        metrics = {
            "total_lines": 0,
            "total_files": len(python_files),
            "avg_complexity": 0,
            "long_functions": 0,
            "long_files": 0
        }
        
        total_complexity = 0
        function_count = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                file_lines = len(lines)
                metrics["total_lines"] += file_lines
                
                # Check for very long files
                if file_lines > 1000:
                    metrics["long_files"] += 1
                    issues.append(f"{file_path.relative_to(self.project_root)}: File too long ({file_lines} lines)")
                
                # Parse and analyze functions
                source = ''.join(lines)
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        function_count += 1
                        
                        # Estimate complexity (simplified)
                        complexity = self._estimate_complexity(node)
                        total_complexity += complexity
                        
                        # Check function length
                        func_lines = getattr(node, 'end_lineno', node.lineno) - node.lineno + 1
                        if func_lines > 100:
                            metrics["long_functions"] += 1
                            issues.append(
                                f"{file_path.relative_to(self.project_root)}:"
                                f"{node.lineno}: Function '{node.name}' too long ({func_lines} lines)"
                            )
                        
                        if complexity > 10:
                            issues.append(
                                f"{file_path.relative_to(self.project_root)}:"
                                f"{node.lineno}: Function '{node.name}' too complex (complexity: {complexity})"
                            )
                            
            except Exception as e:
                issues.append(f"Error analyzing {file_path}: {e}")
        
        if function_count > 0:
            metrics["avg_complexity"] = round(total_complexity / function_count, 2)
        
        return {
            "passed": not issues,
            "errors": issues,
            "metrics": metrics
        }
    
    def _estimate_complexity(self, node: ast.AST) -> int:
        """Estimate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def check_security(self) -> Dict[str, Any]:
        """Basic security checks."""
        issues = []
        python_files = list(self.project_root.glob("**/*.py"))
        
        # Security patterns to check
        security_patterns = {
            r'eval\s*\(': "Use of eval() function",
            r'exec\s*\(': "Use of exec() function", 
            r'__import__\s*\(': "Dynamic import usage",
            r'pickle\.loads\s*\(': "Unsafe pickle.loads usage",
            r'subprocess\.call\s*\([^)]*shell\s*=\s*True': "Shell=True in subprocess",
            r'os\.system\s*\(': "Use of os.system()",
            r'input\s*\([^)]*\)\s*\)': "Raw input() usage",
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in security_patterns.items():
                    matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append(
                            f"{file_path.relative_to(self.project_root)}:"
                            f"{line_num}: {description}"
                        )
            except Exception as e:
                issues.append(f"Error scanning {file_path}: {e}")
        
        return {
            "passed": not issues,
            "errors": issues,
            "metrics": {
                "files_scanned": len(python_files),
                "security_issues": len(issues)
            }
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation coverage."""
        issues = []
        doc_files = [
            "README.md", "CONTRIBUTING.md", "LICENSE",
            "docs/ROADMAP.md"
        ]
        
        missing_docs = []
        for doc_file in doc_files:
            if not (self.project_root / doc_file).exists():
                missing_docs.append(doc_file)
        
        # Check docstring coverage in main modules
        main_modules = [
            "spatial_omics_gfm/models/graph_transformer.py",
            "spatial_omics_gfm/data/visium.py",
            "spatial_omics_gfm/tasks/cell_typing.py",
        ]
        
        undocumented_functions = 0
        total_functions = 0
        
        for module_path in main_modules:
            full_path = self.project_root / module_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        source = f.read()
                    
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if not ast.get_docstring(node):
                                undocumented_functions += 1
                                if len(issues) < 10:  # Limit output
                                    issues.append(
                                        f"{module_path}:{node.lineno}: "
                                        f"{type(node).__name__} '{node.name}' lacks docstring"
                                    )
                                    
                except Exception as e:
                    issues.append(f"Error checking docs in {module_path}: {e}")
        
        if missing_docs:
            issues.extend([f"Missing documentation: {doc}" for doc in missing_docs])
        
        doc_coverage = (
            (total_functions - undocumented_functions) / total_functions * 100 
            if total_functions > 0 else 100
        )
        
        return {
            "passed": not missing_docs and doc_coverage >= 70,
            "errors": issues,
            "metrics": {
                "missing_docs": len(missing_docs),
                "doc_coverage": round(doc_coverage, 2),
                "undocumented_functions": undocumented_functions,
                "total_functions": total_functions
            }
        }
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test structure and coverage."""
        test_dir = self.project_root / "tests"
        issues = []
        
        if not test_dir.exists():
            issues.append("Missing tests directory")
            return {"passed": False, "errors": issues, "metrics": {}}
        
        test_files = list(test_dir.glob("test_*.py"))
        src_files = list((self.project_root / "spatial_omics_gfm").glob("**/*.py"))
        
        # Check for test files corresponding to source modules
        core_modules = ["models", "data", "tasks", "training", "inference"]
        missing_tests = []
        
        for module in core_modules:
            test_file = test_dir / f"test_{module}.py"
            if not test_file.exists():
                missing_tests.append(f"test_{module}.py")
        
        # Analyze test file structure
        test_functions = 0
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                tree = ast.parse(source)
                
                for node in ast.walk(tree):
                    if (isinstance(node, ast.FunctionDef) and 
                        node.name.startswith('test_')):
                        test_functions += 1
            except Exception as e:
                issues.append(f"Error analyzing {test_file}: {e}")
        
        if missing_tests:
            issues.extend([f"Missing test file: {test}" for test in missing_tests])
        
        return {
            "passed": not missing_tests and test_functions >= 20,
            "errors": issues,
            "metrics": {
                "test_files": len(test_files),
                "test_functions": test_functions,
                "missing_test_files": len(missing_tests),
                "source_files": len(src_files)
            }
        }
    
    def check_performance_patterns(self) -> Dict[str, Any]:
        """Check for common performance anti-patterns."""
        issues = []
        python_files = list(self.project_root.glob("spatial_omics_gfm/**/*.py"))
        
        # Performance anti-patterns
        patterns = {
            r'\.append\s*\([^)]+\)\s*\n\s*for\s+': "List append in loop - consider list comprehension",
            r'len\s*\([^)]+\)\s*==\s*0': "Use 'not sequence' instead of 'not sequence'",
            r'\.keys\(\)\s*\)\s*:\s*\n.*\[[^]]+\]': "Inefficient dict iteration pattern",
        }
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, description in patterns.items():
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append(
                            f"{file_path.relative_to(self.project_root)}:"
                            f"{line_num}: {description}"
                        )
            except Exception as e:
                issues.append(f"Error checking performance in {file_path}: {e}")
        
        return {
            "passed": not issues,
            "errors": issues,
            "metrics": {
                "files_checked": len(python_files),
                "performance_issues": len(issues)
            }
        }
    
    def save_report(self, output_file: str = "quality_report.json"):
        """Save detailed quality report."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Quality report saved to {output_file}")


def main():
    """Run quality gates."""
    gates = QualityGates()
    results = gates.run_all_gates()
    gates.save_report()
    
    # Exit with error code if gates failed
    sys.exit(0 if results.get("overall_passed", False) else 1)


if __name__ == "__main__":
    main()