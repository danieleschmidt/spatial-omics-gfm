#!/usr/bin/env python3
"""
Automated Restore Script for Spatial-Omics GFM
"""
import json
import tarfile
import argparse
from pathlib import Path


def restore_backup(backup_path: Path, restore_dir: Path = None):
    """Restore project from backup"""
    if restore_dir is None:
        restore_dir = Path.cwd()
    
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_path}")
    
    # Extract backup
    with tarfile.open(backup_path, "r:gz") as tar:
        tar.extractall(restore_dir)
    
    restore_info = {
        "backup_path": str(backup_path),
        "restore_dir": str(restore_dir),
        "size_mb": backup_path.stat().st_size / (1024 * 1024)
    }
    
    return restore_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore from backup")
    parser.add_argument("backup_path", type=Path, help="Path to backup file")
    parser.add_argument("--restore-dir", type=Path, help="Restore directory")
    
    args = parser.parse_args()
    
    restore_info = restore_backup(args.backup_path, args.restore_dir)
    print(f"Restored from: {restore_info['backup_path']}")
    print(f"Size: {restore_info['size_mb']:.2f} MB")
