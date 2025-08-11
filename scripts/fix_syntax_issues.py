#!/usr/bin/env python3
"""
Fix syntax issues in the codebase.
"""

import re
from pathlib import Path


def fix_file_syntax_issues(file_path: Path) -> bool:
    """Fix common syntax issues in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix line continuation issues with newlines
        content = re.sub(r'\\n\s+', '\n        ', content)
        
        # Remove unnecessary eval usage
        content = re.sub(r'eval\([\'"]([^\'"]+)[\'"]\)', r'\1', content)
        
        # Fix len() == 0 patterns
        content = re.sub(r'len\(([^)]+)\)\s*==\s*0', r'not \1', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False
    
    return False


def main():
    """Fix syntax issues in the project."""
    project_root = Path(".")
    files_fixed = 0
    
    # Fix Python files
    for py_file in project_root.glob("**/*.py"):
        if fix_file_syntax_issues(py_file):
            files_fixed += 1
            print(f"Fixed: {py_file}")
    
    print(f"Fixed {files_fixed} files")


if __name__ == "__main__":
    main()