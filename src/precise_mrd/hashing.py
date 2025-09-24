"""
Input hashing and results lockfile module.

This module provides:
- Input file hashing for auditability
- Configuration fingerprinting
- Results lockfile generation
- Version tracking
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml


class HashingManager:
    """Manage input hashing and results lockfile generation."""
    
    def __init__(self, lockfile_path: str = "results/lockfile.json"):
        """Initialize hashing manager."""
        self.lockfile_path = Path(lockfile_path)
        self.lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    
    def hash_file(self, filepath: str, algorithm: str = "sha256") -> str:
        """Calculate hash of a file."""
        hash_func = hashlib.new(algorithm)
        
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except FileNotFoundError:
            return f"FILE_NOT_FOUND:{filepath}"
        except Exception as e:
            return f"ERROR:{str(e)}"
    
    def hash_directory(self, dirpath: str, pattern: str = "*", algorithm: str = "sha256") -> Dict[str, str]:
        """Calculate hashes for all files in a directory matching pattern."""
        dir_path = Path(dirpath)
        file_hashes = {}
        
        if not dir_path.exists():
            return {"ERROR": f"Directory not found: {dirpath}"}
        
        for filepath in dir_path.glob(pattern):
            if filepath.is_file():
                relative_path = str(filepath.relative_to(dir_path))
                file_hashes[relative_path] = self.hash_file(str(filepath), algorithm)
        
        return file_hashes
    
    def hash_config(self, config: Dict[str, Any], algorithm: str = "sha256") -> str:
        """Calculate hash of configuration dictionary."""
        # Convert config to sorted JSON string for consistent hashing
        config_str = json.dumps(config, sort_keys=True, separators=(',', ':'))
        hash_func = hashlib.new(algorithm)
        hash_func.update(config_str.encode('utf-8'))
        return hash_func.hexdigest()
    
    def get_git_info(self, repo_path: str = ".") -> Dict[str, str]:
        """Get current git repository information."""
        git_info = {}
        
        try:
            # Get current commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info["commit_hash"] = result.stdout.strip()
            
            # Get current branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info["branch"] = result.stdout.strip()
            
            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            git_info["has_uncommitted_changes"] = bool(result.stdout.strip())
            git_info["uncommitted_files"] = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get last commit info
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H %s %an %ad", "--date=iso"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            if result.stdout.strip():
                parts = result.stdout.strip().split(' ', 3)
                if len(parts) >= 4:
                    git_info["last_commit"] = {
                        "hash": parts[0],
                        "message": parts[1],
                        "author": parts[2],
                        "date": parts[3] if len(parts) > 3 else ""
                    }
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_info["error"] = "Git information unavailable"
        
        return git_info
    
    def get_environment_info(self) -> Dict[str, str]:
        """Get environment and version information."""
        env_info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": os.sys.platform,
            "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
            "user": os.environ.get("USER", "unknown"),
            "working_directory": str(Path.cwd())
        }
        
        # Get package versions
        try:
            import numpy
            env_info["numpy_version"] = numpy.__version__
        except ImportError:
            pass
        
        try:
            import pandas
            env_info["pandas_version"] = pandas.__version__
        except ImportError:
            pass
        
        try:
            import scipy
            env_info["scipy_version"] = scipy.__version__
        except ImportError:
            pass
        
        try:
            import matplotlib
            env_info["matplotlib_version"] = matplotlib.__version__
        except ImportError:
            pass
        
        return env_info
    
    def create_lockfile(
        self,
        run_id: str,
        config: Dict[str, Any],
        input_files: Optional[List[str]] = None,
        input_directories: Optional[List[str]] = None,
        results_summary: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive lockfile for a run."""
        
        lockfile_data = {
            "run_id": run_id,
            "creation_timestamp": datetime.now().isoformat(),
            "environment": self.get_environment_info(),
            "git": self.get_git_info(),
            "config_hash": self.hash_config(config),
            "config": config,
            "input_hashes": {},
            "results_summary": results_summary or {}
        }
        
        # Hash input files
        if input_files:
            for filepath in input_files:
                lockfile_data["input_hashes"][filepath] = self.hash_file(filepath)
        
        # Hash input directories
        if input_directories:
            for dirpath in input_directories:
                dir_hashes = self.hash_directory(dirpath)
                lockfile_data["input_hashes"][f"directory:{dirpath}"] = dir_hashes
        
        return lockfile_data
    
    def save_lockfile(self, lockfile_data: Dict[str, Any]) -> str:
        """Save lockfile to disk."""
        with open(self.lockfile_path, 'w') as f:
            json.dump(lockfile_data, f, indent=2, default=str)
        
        return str(self.lockfile_path)
    
    def load_lockfile(self, lockfile_path: Optional[str] = None) -> Dict[str, Any]:
        """Load lockfile from disk."""
        path = Path(lockfile_path) if lockfile_path else self.lockfile_path
        
        if not path.exists():
            raise FileNotFoundError(f"Lockfile not found: {path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def verify_reproducibility(
        self,
        current_config: Dict[str, Any],
        current_inputs: Optional[List[str]] = None,
        reference_lockfile: Optional[str] = None
    ) -> Dict[str, Any]:
        """Verify if current run would be reproducible against a reference."""
        
        if reference_lockfile:
            reference_data = self.load_lockfile(reference_lockfile)
        else:
            reference_data = self.load_lockfile()
        
        verification_result = {
            "reproducible": True,
            "differences": [],
            "warnings": []
        }
        
        # Check config hash
        current_config_hash = self.hash_config(current_config)
        reference_config_hash = reference_data.get("config_hash")
        
        if current_config_hash != reference_config_hash:
            verification_result["reproducible"] = False
            verification_result["differences"].append("Configuration has changed")
        
        # Check git status
        current_git = self.get_git_info()
        reference_git = reference_data.get("git", {})
        
        if current_git.get("commit_hash") != reference_git.get("commit_hash"):
            verification_result["differences"].append("Different git commit")
        
        if current_git.get("has_uncommitted_changes"):
            verification_result["warnings"].append("Uncommitted changes present")
        
        # Check input files if provided
        if current_inputs:
            reference_hashes = reference_data.get("input_hashes", {})
            for filepath in current_inputs:
                current_hash = self.hash_file(filepath)
                reference_hash = reference_hashes.get(filepath)
                
                if current_hash != reference_hash:
                    verification_result["reproducible"] = False
                    verification_result["differences"].append(f"Input file changed: {filepath}")
        
        # Check environment differences
        current_env = self.get_environment_info()
        reference_env = reference_data.get("environment", {})
        
        for key in ["python_version", "numpy_version", "pandas_version"]:
            if key in reference_env and current_env.get(key) != reference_env.get(key):
                verification_result["warnings"].append(f"Different {key}: {current_env.get(key)} vs {reference_env.get(key)}")
        
        return verification_result
    
    def generate_run_fingerprint(
        self,
        config: Dict[str, Any],
        input_files: Optional[List[str]] = None
    ) -> str:
        """Generate a unique fingerprint for a run configuration."""
        
        fingerprint_data = {
            "config": config,
            "git_hash": self.get_git_info().get("commit_hash", "unknown"),
            "input_hashes": {}
        }
        
        if input_files:
            for filepath in input_files:
                fingerprint_data["input_hashes"][filepath] = self.hash_file(filepath)
        
        return self.hash_config(fingerprint_data)
    
    def cleanup_old_lockfiles(self, keep_recent: int = 10) -> List[str]:
        """Clean up old lockfiles, keeping only the most recent ones."""
        
        lockfile_dir = self.lockfile_path.parent
        pattern = f"{self.lockfile_path.stem}_*.json"
        
        # Find all lockfiles with timestamps
        lockfiles = []
        for filepath in lockfile_dir.glob(pattern):
            try:
                # Extract timestamp from filename
                timestamp_str = filepath.stem.split('_', 1)[1]
                timestamp = datetime.fromisoformat(timestamp_str.replace('_', ':'))
                lockfiles.append((timestamp, filepath))
            except (ValueError, IndexError):
                continue
        
        # Sort by timestamp and remove old files
        lockfiles.sort(reverse=True)  # Most recent first
        removed_files = []
        
        for timestamp, filepath in lockfiles[keep_recent:]:
            try:
                filepath.unlink()
                removed_files.append(str(filepath))
            except OSError:
                pass
        
        return removed_files