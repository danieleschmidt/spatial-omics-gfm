"""
Security validation and sanitization utilities.

Provides security checks for file operations, user inputs,
and data processing to prevent malicious attacks.
"""

import os
import re
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
import subprocess

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Exception for security-related validation failures."""
    pass


class SecurityValidator:
    """Security validation and sanitization utilities."""
    
    def __init__(self):
        # Dangerous file patterns and extensions
        self.dangerous_extensions = {
            '.exe', '.bat', '.cmd', '.com', '.scr', '.pif', '.vbs', '.js', '.jar',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.psm1', '.psd1',
            '.app', '.deb', '.rpm', '.dmg', '.pkg', '.msi', '.dll', '.so'
        }
        
        # Dangerous path patterns
        self.dangerous_paths = [
            r'\.\./',           # Directory traversal
            r'\.\.\\'
            r'/etc/',           # System directories
            r'/bin/',
            r'/sbin/',
            r'/usr/bin/',
            r'/usr/sbin/',
            r'/var/',
            r'/tmp/',
            r'/dev/',
            r'/proc/',
            r'C:\\Windows\\',
            r'C:\\Program Files\\',
            r'\\\\',            # UNC paths
        ]
        
        # Dangerous command patterns
        self.command_injection_patterns = [
            r'[;&|`$(){}[\]<>]',  # Shell metacharacters
            r'\\x[0-9a-fA-F]{2}', # Hex encoding
            r'%[0-9a-fA-F]{2}',   # URL encoding
            r'\$\{.*\}',          # Variable substitution
            r'`.*`',              # Command substitution
            r'\$\(.*\)',          # Command substitution
        ]
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"'.*'",
            r'";.*',
            r'--.*',
            r'/\*.*\*/',
            r'\bUNION\b',
            r'\bSELECT\b',
            r'\bINSERT\b',
            r'\bDELETE\b',
            r'\bDROP\b',
            r'\bALTER\b',
        ]
    
    def check_file_safety(
        self,
        file_path: Union[str, Path],
        check_content: bool = True,
        max_file_size: int = 100 * 1024 * 1024  # 100MB default
    ) -> Dict[str, Any]:
        """
        Check if a file is safe to process.
        
        Args:
            file_path: Path to file to check
            check_content: Whether to scan file content for threats
            max_file_size: Maximum allowed file size in bytes
            
        Returns:
            Dictionary with safety check results
        """
        results = {
            "safe": True,
            "warnings": [],
            "errors": [],
            "checks": {}
        }
        
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                results["errors"].append(f"File does not exist: {path}")
                results["safe"] = False
                return results
            
            # Check file extension
            file_ext = path.suffix.lower()
            if file_ext in self.dangerous_extensions:
                results["errors"].append(f"Dangerous file extension: {file_ext}")
                results["safe"] = False
            
            results["checks"]["extension"] = file_ext
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > max_file_size:
                results["errors"].append(f"File too large: {file_size} > {max_file_size}")
                results["safe"] = False
            
            results["checks"]["size_bytes"] = file_size
            
            # Check file permissions
            if path.is_file():
                # Check if file is executable
                if os.access(path, os.X_OK):
                    results["warnings"].append("File has execute permissions")
            
            # Check path for dangerous patterns
            path_str = str(path.resolve())
            for pattern in self.dangerous_paths:
                if re.search(pattern, path_str, re.IGNORECASE):
                    results["errors"].append(f"Dangerous path pattern: {pattern}")
                    results["safe"] = False
            
            # Content-based checks
            if check_content and path.is_file() and file_size < 10 * 1024 * 1024:  # Only for files < 10MB
                try:
                    # Try to read as text (first 1KB)
                    with open(path, 'rb') as f:
                        header = f.read(1024)
                    
                    # Check if it's a text file
                    try:
                        text_header = header.decode('utf-8', errors='ignore')
                        results["checks"]["contains_text"] = True
                        
                        # Check for suspicious patterns in text files
                        if self._contains_suspicious_content(text_header):
                            results["warnings"].append("File contains potentially suspicious content")
                        
                    except UnicodeDecodeError:
                        results["checks"]["contains_text"] = False
                    
                    # Check file signature/magic bytes
                    file_type = self._identify_file_type(header)
                    results["checks"]["detected_type"] = file_type
                    
                    # Warn about mismatched extensions
                    if file_type and file_ext:
                        expected_ext = self._get_expected_extension(file_type)
                        if expected_ext and file_ext != expected_ext:
                            results["warnings"].append(f"Extension mismatch: {file_ext} vs expected {expected_ext}")
                
                except Exception as e:
                    results["warnings"].append(f"Could not check file content: {e}")
            
            # Compute file hash for integrity checking
            if path.is_file() and file_size < 50 * 1024 * 1024:  # Only for files < 50MB
                try:
                    file_hash = self._compute_file_hash(path)
                    results["checks"]["sha256"] = file_hash
                except Exception as e:
                    results["warnings"].append(f"Could not compute file hash: {e}")
        
        except Exception as e:
            results["errors"].append(f"File safety check failed: {e}")
            results["safe"] = False
        
        return results
    
    def sanitize_user_input(
        self,
        user_input: str,
        max_length: int = 1000,
        allow_html: bool = False,
        allow_sql: bool = False
    ) -> str:
        """
        Sanitize user input to prevent injection attacks.
        
        Args:
            user_input: Raw user input string
            max_length: Maximum allowed length
            allow_html: Whether to allow HTML tags
            allow_sql: Whether to allow SQL keywords
            
        Returns:
            Sanitized input string
            
        Raises:
            SecurityError: If input contains dangerous patterns
        """
        if not isinstance(user_input, str):
            raise SecurityError("Input must be a string")
        
        # Length check
        if len(user_input) > max_length:
            raise SecurityError(f"Input too long: {len(user_input)} > {max_length}")
        
        sanitized = user_input
        
        # Check for command injection patterns
        for pattern in self.command_injection_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                raise SecurityError(f"Potentially dangerous pattern detected: {pattern}")
        
        # Check for SQL injection patterns (if not allowed)
        if not allow_sql:
            for pattern in self.sql_injection_patterns:
                if re.search(pattern, sanitized, re.IGNORECASE):
                    raise SecurityError(f"Potentially dangerous SQL pattern: {pattern}")
        
        # HTML/XML sanitization (if not allowed)
        if not allow_html:
            # Remove HTML tags
            sanitized = re.sub(r'<[^>]*>', '', sanitized)
            
            # Escape remaining HTML entities
            html_escapes = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#x27;',
            }
            
            for char, escape in html_escapes.items():
                sanitized = sanitized.replace(char, escape)
        
        # Remove null bytes and other control characters
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Normalize whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        return sanitized
    
    def validate_permissions(
        self,
        file_path: Union[str, Path],
        required_permissions: str = "r"
    ) -> Dict[str, Any]:
        """
        Validate file/directory permissions.
        
        Args:
            file_path: Path to check
            required_permissions: Required permissions string (r/w/x combinations)
            
        Returns:
            Permission validation results
        """
        results = {
            "valid": True,
            "permissions": {},
            "errors": []
        }
        
        try:
            path = Path(file_path)
            
            if not path.exists():
                results["errors"].append(f"Path does not exist: {path}")
                results["valid"] = False
                return results
            
            # Check actual permissions
            permissions = {
                "readable": os.access(path, os.R_OK),
                "writable": os.access(path, os.W_OK),
                "executable": os.access(path, os.X_OK)
            }
            
            results["permissions"] = permissions
            
            # Check required permissions
            if 'r' in required_permissions and not permissions["readable"]:
                results["errors"].append("Missing read permission")
                results["valid"] = False
            
            if 'w' in required_permissions and not permissions["writable"]:
                results["errors"].append("Missing write permission")
                results["valid"] = False
            
            if 'x' in required_permissions and not permissions["executable"]:
                results["errors"].append("Missing execute permission")
                results["valid"] = False
            
            # Get detailed permission info (Unix-like systems)
            try:
                stat_info = path.stat()
                results["permissions"]["mode"] = oct(stat_info.st_mode)[-3:]
                results["permissions"]["owner_uid"] = stat_info.st_uid
                results["permissions"]["group_gid"] = stat_info.st_gid
            except Exception:
                pass  # Permission details not available on all systems
        
        except Exception as e:
            results["errors"].append(f"Permission check failed: {e}")
            results["valid"] = False
        
        return results
    
    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        suspicious_patterns = [
            r'<script[^>]*>',           # JavaScript
            r'javascript:',             # JavaScript URLs
            r'eval\s*\(',              # eval() calls
            r'exec\s*\(',              # exec() calls
            r'__import__',             # Python imports
            r'subprocess',             # Process execution
            r'os\.system',             # System calls
            r'shell=True',             # Shell execution
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False
    
    def _identify_file_type(self, header: bytes) -> Optional[str]:
        """Identify file type from magic bytes."""
        magic_signatures = {
            b'\x89PNG\r\n\x1a\n': 'PNG',
            b'\xff\xd8\xff': 'JPEG',
            b'GIF87a': 'GIF87a',
            b'GIF89a': 'GIF89a',
            b'%PDF': 'PDF',
            b'PK\x03\x04': 'ZIP',
            b'\x1f\x8b': 'GZIP',
            b'BZh': 'BZIP2',
            b'\x7fELF': 'ELF',
            b'MZ': 'PE/DOS',
            b'\xca\xfe\xba\xbe': 'Java Class',
            b'\x89HDF\r\n\x1a\n': 'HDF5',
        }
        
        for signature, file_type in magic_signatures.items():
            if header.startswith(signature):
                return file_type
        
        # Check for text files
        try:
            header.decode('utf-8')
            return 'TEXT'
        except UnicodeDecodeError:
            pass
        
        return 'UNKNOWN'
    
    def _get_expected_extension(self, file_type: str) -> Optional[str]:
        """Get expected file extension for detected file type."""
        type_extensions = {
            'PNG': '.png',
            'JPEG': '.jpg',
            'GIF87a': '.gif',
            'GIF89a': '.gif',
            'PDF': '.pdf',
            'ZIP': '.zip',
            'GZIP': '.gz',
            'BZIP2': '.bz2',
            'HDF5': '.h5',
            'TEXT': '.txt',
        }
        
        return type_extensions.get(file_type)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()


# Convenience functions
def check_file_safety(file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Convenience function for file safety checking."""
    validator = SecurityValidator()
    return validator.check_file_safety(file_path, **kwargs)


def sanitize_user_input(user_input: str, **kwargs) -> str:
    """Convenience function for input sanitization."""
    validator = SecurityValidator()
    return validator.sanitize_user_input(user_input, **kwargs)


def validate_permissions(file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Convenience function for permission validation."""
    validator = SecurityValidator()
    return validator.validate_permissions(file_path, **kwargs)