#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import shutil

def get_installed_version(package):
    """Get the installed version of a package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Version: "):
                return line.split(": ")[1].strip()
    except subprocess.CalledProcessError:
        return None
    return None

def get_latest_version(package):
    """Get the latest version available on PyPI."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", package],
            capture_output=True,
            text=True,
            check=True
        )
        # The first version listed is the latest
        for line in result.stdout.split("\n"):
            if "Available versions:" in line:
                versions = line.split("Available versions:")[1].strip()
                if versions:
                    return versions.split(",")[0].strip()
    except subprocess.CalledProcessError:
        return None
    return None

def backup_requirements(requirements_path):
    """Create a backup of the requirements file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = requirements_path.parent / f"requirements.txt.backup_{timestamp}"
    shutil.copy2(requirements_path, backup_path)
    print(f"Created backup at {backup_path}")
    return backup_path

def update_dependencies(requirements_path, dry_run=False):
    """Update all dependencies to their latest versions."""
    if not requirements_path.exists():
        print(f"Error: {requirements_path} not found")
        return False

    # Read current requirements
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Parse package names and constraints
    updates = []
    for req in requirements:
        package = req.split(">=")[0].split("==")[0].split("<")[0].split(">")[0].strip()
        latest_version = get_latest_version(package)
        
        if latest_version:
            updates.append(f"{package}>={latest_version}")
            print(f"Found update for {package}: {latest_version}")
        else:
            updates.append(req)
            print(f"Keeping original requirement for {package}")

    if dry_run:
        print("\nDry run - would update to:")
        for update in updates:
            print(update)
        return True

    # Backup original requirements
    backup_path = backup_requirements(requirements_path)

    # Write updated requirements
    with open(requirements_path, "w") as f:
        for update in updates:
            f.write(f"{update}\n")

    print("\nRequirements updated successfully!")
    return True

def install_requirements(requirements_path):
    """Install the updated requirements."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            check=True
        )
        print("\nAll dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError installing dependencies: {e}")
        return False

def run_tests():
    """Run the project's tests."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pytest"],
            check=True
        )
        print("\nAll tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed: {e}")
        return False

def main():
    requirements_path = Path("requirements.txt")
    
    # Parse command line arguments
    dry_run = "--dry-run" in sys.argv
    skip_install = "--skip-install" in sys.argv
    skip_tests = "--skip-tests" in sys.argv

    print("Checking for updates...")
    if not update_dependencies(requirements_path, dry_run):
        return 1

    if dry_run:
        return 0

    if not skip_install:
        print("\nInstalling updated dependencies...")
        if not install_requirements(requirements_path):
            return 1

    if not skip_tests:
        print("\nRunning tests...")
        if not run_tests():
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main()) 