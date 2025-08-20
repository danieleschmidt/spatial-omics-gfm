#!/usr/bin/env python3
"""
Automated Backup Script for Spatial-Omics GFM
"""
import json
import shutil
import tarfile
import datetime
from pathlib import Path


def create_backup(backup_dir: Path = None):
    """Create comprehensive backup of the project"""
    if backup_dir is None:
        backup_dir = Path.cwd() / "backups"
    
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"spatial_omics_gfm_backup_{timestamp}.tar.gz"
    backup_path = backup_dir / backup_name
    
    # Files to backup
    backup_items = [
        "spatial_omics_gfm/",
        "examples/", 
        "tests/",
        "docs/",
        "README.md",
        "pyproject.toml",
        "requirements*.txt"
    ]
    
    with tarfile.open(backup_path, "w:gz") as tar:
        for item in backup_items:
            item_path = Path(item)
            if item_path.exists():
                tar.add(item_path, arcname=item)
    
    backup_info = {
        "timestamp": timestamp,
        "backup_path": str(backup_path),
        "size_mb": backup_path.stat().st_size / (1024 * 1024),
        "items_backed_up": len(backup_items)
    }
    
    # Save backup info
    info_path = backup_dir / f"backup_info_{timestamp}.json"
    with open(info_path, 'w') as f:
        json.dump(backup_info, f, indent=2)
    
    return backup_info


if __name__ == "__main__":
    backup_info = create_backup()
    print(f"Backup created: {backup_info['backup_path']}")
    print(f"Size: {backup_info['size_mb']:.2f} MB")
