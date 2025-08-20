#!/usr/bin/env python3
"""
Health Monitoring Script for Spatial-Omics GFM
"""
import psutil
import json
import time
from pathlib import Path


def check_system_health():
    """Check system health metrics"""
    health = {
        "timestamp": time.time(),
        "memory": {
            "total_gb": psutil.virtual_memory().total / (1024**3),
            "available_gb": psutil.virtual_memory().available / (1024**3),
            "percent_used": psutil.virtual_memory().percent
        },
        "cpu": {
            "percent_used": psutil.cpu_percent(interval=1),
            "core_count": psutil.cpu_count()
        },
        "disk": {
            "total_gb": psutil.disk_usage('/').total / (1024**3),
            "free_gb": psutil.disk_usage('/').free / (1024**3),
            "percent_used": psutil.disk_usage('/').percent
        }
    }
    
    # Health status
    health["status"] = "healthy"
    if health["memory"]["percent_used"] > 90:
        health["status"] = "warning"
    if health["disk"]["percent_used"] > 90:
        health["status"] = "critical"
    
    return health


if __name__ == "__main__":
    health = check_system_health()
    print(json.dumps(health, indent=2))
