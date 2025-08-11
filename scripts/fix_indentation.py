#!/usr/bin/env python3
"""
Fix indentation issues in the codebase.
"""

import ast
import re
from pathlib import Path


def fix_file_indentation(file_path: Path) -> bool:
    """Fix indentation issues in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Split into lines for processing
        lines = content.splitlines()
        fixed_lines = []
        indent_level = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                i += 1
                continue
            
            # Handle function/class definitions
            if (stripped.startswith('def ') or stripped.startswith('class ') or 
                stripped.startswith('async def ') or stripped.startswith('from ') or 
                stripped.startswith('import ')):
                # Keep original indentation for top-level definitions
                fixed_lines.append(line)
                i += 1
                continue
                
            # Handle line continuation issues - merge lines that should be together
            if i > 0 and not line.strip():
                # Look ahead for lines that should be indented
                next_idx = i + 1
                while next_idx < len(lines) and not lines[next_idx].strip():
                    next_idx += 1
                
                if next_idx < len(lines) and lines[next_idx].strip():
                    next_line = lines[next_idx].strip()
                    # If the next non-empty line looks like it should be indented
                    if (next_line.startswith('ConfigManager') or 
                        next_line.startswith('SecurityConfig') or
                        next_line.startswith('start_monitoring') or
                        'with tempfile.TemporaryDirectory' in next_line):
                        # Skip this empty line and let the next one be processed normally
                        i += 1
                        continue
            
            # Keep the line as-is
            fixed_lines.append(line)
            i += 1
        
        # Rejoin lines
        new_content = '\n'.join(fixed_lines)
        
        # Test if the fixed content is valid Python
        try:
            ast.parse(new_content)
        except SyntaxError:
            # If still invalid, return original
            return False
            
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False
    
    return False


def main():
    """Fix indentation issues in the project."""
    project_root = Path(".")
    files_fixed = 0
    
    # Focus on the problematic file
    problem_files = [
        "examples/complete_robustness_example.py"
    ]
    
    for file_name in problem_files:
        file_path = project_root / file_name
        if file_path.exists():
            if fix_file_indentation(file_path):
                files_fixed += 1
                print(f"Fixed: {file_path}")
    
    print(f"Fixed {files_fixed} files")


if __name__ == "__main__":
    main()