#!/usr/bin/env python3
"""
Health Check Script for Spatial-Omics GFM Container
Comprehensive health monitoring for production deployments
"""
import json
import os
import sys
import time
from typing import Dict, Any, Optional
import urllib.request
import urllib.error


class HealthChecker:
    """Production health checker with comprehensive diagnostics"""
    
    def __init__(self):
        self.port = os.getenv('PORT', '8000')
        self.host = os.getenv('HOST', '127.0.0.1')
        self.timeout = int(os.getenv('HEALTH_CHECK_TIMEOUT', '10'))
        self.checks = {
            'api_endpoint': True,
            'memory_usage': True,
            'disk_space': True,
            'dependencies': True
        }
    
    def check_api_endpoint(self) -> Dict[str, Any]:
        """Check if API endpoint is responding"""
        try:
            url = f"http://{self.host}:{self.port}/health"
            
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    return {
                        'status': 'healthy',
                        'response_time': data.get('response_time', 0),
                        'timestamp': data.get('timestamp', time.time())
                    }
                else:
                    return {
                        'status': 'unhealthy',
                        'error': f'HTTP {response.status}',
                        'response_time': None
                    }
                    
        except urllib.error.URLError as e:
            return {
                'status': 'unhealthy',
                'error': f'Connection error: {str(e)}',
                'response_time': None
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': f'Unexpected error: {str(e)}',
                'response_time': None
            }
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            # Read memory info from /proc/meminfo
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            mem_total = 0
            mem_available = 0
            
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) * 1024  # Convert kB to bytes
                elif line.startswith('MemAvailable:'):
                    mem_available = int(line.split()[1]) * 1024
            
            if mem_total > 0:
                mem_used = mem_total - mem_available
                usage_percent = (mem_used / mem_total) * 100
                
                status = 'healthy' if usage_percent < 90 else 'unhealthy'
                
                return {
                    'status': status,
                    'usage_percent': round(usage_percent, 2),
                    'used_bytes': mem_used,
                    'total_bytes': mem_total,
                    'available_bytes': mem_available
                }
            else:
                return {
                    'status': 'unknown',
                    'error': 'Could not determine memory usage'
                }
                
        except Exception as e:
            return {
                'status': 'unknown',
                'error': f'Memory check failed: {str(e)}'
            }
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            import shutil
            
            # Check disk usage for the current directory
            total, used, free = shutil.disk_usage('/app')
            
            usage_percent = (used / total) * 100
            status = 'healthy' if usage_percent < 85 else 'unhealthy'
            
            return {
                'status': status,
                'usage_percent': round(usage_percent, 2),
                'used_bytes': used,
                'total_bytes': total,
                'free_bytes': free
            }
            
        except Exception as e:
            return {
                'status': 'unknown',
                'error': f'Disk check failed: {str(e)}'
            }
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies"""
        try:
            dependencies = {
                'torch': None,
                'numpy': None,
                'spatial_omics_gfm': None
            }
            
            # Check if critical packages can be imported
            for package in dependencies.keys():
                try:
                    if package == 'torch':
                        import torch
                        dependencies[package] = {
                            'version': torch.__version__,
                            'cuda_available': torch.cuda.is_available(),
                            'status': 'available'
                        }
                    elif package == 'numpy':
                        import numpy
                        dependencies[package] = {
                            'version': numpy.__version__,
                            'status': 'available'
                        }
                    elif package == 'spatial_omics_gfm':
                        import spatial_omics_gfm
                        dependencies[package] = {
                            'version': getattr(spatial_omics_gfm, '__version__', 'unknown'),
                            'status': 'available'
                        }
                except ImportError as e:
                    dependencies[package] = {
                        'status': 'unavailable',
                        'error': str(e)
                    }
            
            # Determine overall status
            failed_deps = [
                name for name, info in dependencies.items()
                if info and info.get('status') != 'available'
            ]
            
            status = 'healthy' if not failed_deps else 'unhealthy'
            
            return {
                'status': status,
                'dependencies': dependencies,
                'failed_dependencies': failed_deps
            }
            
        except Exception as e:
            return {
                'status': 'unknown',
                'error': f'Dependencies check failed: {str(e)}'
            }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'checks': {}
        }
        
        check_methods = {
            'api_endpoint': self.check_api_endpoint,
            'memory_usage': self.check_memory_usage,
            'disk_space': self.check_disk_space,
            'dependencies': self.check_dependencies
        }
        
        failed_checks = []
        
        for check_name, check_method in check_methods.items():
            if self.checks.get(check_name, True):
                try:
                    result = check_method()
                    results['checks'][check_name] = result
                    
                    if result.get('status') == 'unhealthy':
                        failed_checks.append(check_name)
                        
                except Exception as e:
                    results['checks'][check_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    failed_checks.append(check_name)
        
        # Determine overall status
        if failed_checks:
            results['overall_status'] = 'unhealthy'
            results['failed_checks'] = failed_checks
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print health check results in a readable format"""
        print(f"Health Check Results - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        print("-" * 50)
        
        for check_name, check_result in results.get('checks', {}).items():
            status = check_result.get('status', 'unknown').upper()
            print(f"{check_name.replace('_', ' ').title()}: {status}")
            
            if check_result.get('error'):
                print(f"  Error: {check_result['error']}")
            
            if check_name == 'memory_usage' and 'usage_percent' in check_result:
                print(f"  Usage: {check_result['usage_percent']:.1f}%")
            
            if check_name == 'disk_space' and 'usage_percent' in check_result:
                print(f"  Usage: {check_result['usage_percent']:.1f}%")
            
            if check_name == 'api_endpoint' and 'response_time' in check_result:
                if check_result['response_time']:
                    print(f"  Response Time: {check_result['response_time']:.3f}s")
        
        if 'failed_checks' in results:
            print(f"\nFailed Checks: {', '.join(results['failed_checks'])}")


def main():
    """Main health check execution"""
    # Parse command line arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    json_output = '--json' in sys.argv
    
    # Initialize health checker
    checker = HealthChecker()
    
    # Run health checks
    try:
        results = checker.run_all_checks()
        
        if json_output:
            # Output JSON for programmatic consumption
            print(json.dumps(results, indent=2))
        elif verbose:
            # Output detailed results
            checker.print_results(results)
        else:
            # Simple output for Docker health check
            if results['overall_status'] == 'healthy':
                print("HEALTHY")
            else:
                print("UNHEALTHY")
                if verbose:
                    failed = results.get('failed_checks', [])
                    print(f"Failed checks: {', '.join(failed)}")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_status'] == 'healthy' else 1
        sys.exit(exit_code)
        
    except Exception as e:
        if json_output:
            error_result = {
                'timestamp': time.time(),
                'overall_status': 'error',
                'error': str(e)
            }
            print(json.dumps(error_result, indent=2))
        else:
            print(f"UNHEALTHY - Health check error: {e}")
        
        sys.exit(1)


if __name__ == '__main__':
    main()