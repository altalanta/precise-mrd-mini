#!/usr/bin/env python3
"""
DVC Workflow Scripts for Precise MRD Pipeline

This script provides convenient DVC-based workflows for running the pipeline
with proper data versioning and experiment tracking.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_dvc_command(cmd, stage_name=None):
    """Run a DVC command and handle errors"""
    if stage_name:
        print(f"Running DVC stage: {stage_name}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {stage_name or cmd} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {stage_name or cmd} failed:")
        print(f"Error: {e.stderr}")
        return None

def setup():
    """Initialize DVC and add initial data"""
    print("🚀 Setting up DVC for Precise MRD...")

    # Initialize DVC if not already done
    run_dvc_command("dvc init --no-scm", "DVC initialization")

    # Add key directories to DVC
    run_dvc_command("dvc add data", "Data versioning")
    run_dvc_command("dvc add configs", "Config versioning")
    run_dvc_command("dvc add reports", "Reports versioning")

    print("✅ DVC setup complete!")

def run_smoke():
    """Run the smoke test pipeline"""
    print("🧪 Running smoke test pipeline...")
    run_dvc_command("dvc repro smoke", "Smoke test")

def run_determinism():
    """Run determinism verification"""
    print("🔄 Running determinism verification...")
    run_dvc_command("dvc repro determinism", "Determinism check")

def run_evaluation():
    """Run all evaluation stages"""
    print("📊 Running evaluation pipeline...")

    stages = ["eval-lob", "eval-lod", "eval-loq", "eval-contamination", "eval-stratified"]
    for stage in stages:
        run_dvc_command(f"dvc repro {stage}", stage)

def run_validation():
    """Run artifact validation"""
    print("✅ Running artifact validation...")
    run_dvc_command("dvc repro validate-artifacts", "Artifact validation")

def run_docs():
    """Build documentation"""
    print("📚 Building documentation...")
    run_dvc_command("dvc repro docs", "Documentation build")

def run_all():
    """Run complete pipeline"""
    print("🚀 Running complete pipeline...")
    run_dvc_command("dvc repro", "Complete pipeline")

def status():
    """Show DVC status"""
    print("📊 DVC Status:")
    run_dvc_command("dvc status", "Status check")

def diff():
    """Show DVC diff"""
    print("🔍 DVC Changes:")
    run_dvc_command("dvc diff", "Diff check")

def push():
    """Push DVC data to remote"""
    print("📤 Pushing data to remote...")
    run_dvc_command("dvc push", "Data push")

def pull():
    """Pull DVC data from remote"""
    print("📥 Pulling data from remote...")
    run_dvc_command("dvc pull", "Data pull")

def experiment(name):
    """Run an experiment with given name"""
    print(f"🧪 Starting experiment: {name}")
    run_dvc_command(f"dvc exp run --name {name}", f"Experiment {name}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="DVC workflows for Precise MRD")
    parser.add_argument("command", choices=[
        "setup", "smoke", "determinism", "evaluation", "validation",
        "docs", "all", "status", "diff", "push", "pull", "experiment"
    ])
    parser.add_argument("--name", help="Experiment name (for experiment command)")

    args = parser.parse_args()

    if args.command == "experiment":
        if not args.name:
            print("❌ Experiment name required. Use: --name experiment_name")
            sys.exit(1)
        experiment(args.name)
    else:
        globals()[args.command.replace("-", "_")]()

if __name__ == "__main__":
    main()

