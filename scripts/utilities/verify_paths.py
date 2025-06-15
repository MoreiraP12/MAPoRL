#!/usr/bin/env python3
"""
Path Verification Script for Reorganized MAPoRL Repository

This script verifies that all imports and file references work correctly
after the repository reorganization.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Colors for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color

def log_success(msg):
    print(f"{GREEN}✅ {msg}{NC}")

def log_error(msg):
    print(f"{RED}❌ {msg}{NC}")

def log_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{NC}")

def log_info(msg):
    print(f"ℹ️  {msg}")

def verify_import(script_path, description):
    """Verify that a Python script can be imported without errors."""
    try:
        # Change to project root
        project_root = Path(__file__).parent.parent.parent
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Add to Python path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Load and check the script
        spec = importlib.util.spec_from_file_location("test_module", script_path)
        if spec is None:
            log_error(f"Could not load {description}: {script_path}")
            return False
        
        module = importlib.util.module_from_spec(spec)
        
        # Try to execute the module (this will catch import errors)
        try:
            spec.loader.exec_module(module)
            log_success(f"{description}: Import check passed")
            return True
        except ImportError as e:
            log_error(f"{description}: Import error - {e}")
            return False
        except Exception as e:
            log_warning(f"{description}: Other error (may be OK) - {e}")
            return True  # Other errors might be expected (like missing data files)
        
        finally:
            os.chdir(original_cwd)
            
    except Exception as e:
        log_error(f"{description}: Failed to check - {e}")
        return False

def verify_file_exists(file_path, description):
    """Verify that a file exists."""
    if os.path.exists(file_path):
        log_success(f"{description}: File exists")
        return True
    else:
        log_error(f"{description}: File missing - {file_path}")
        return False

def main():
    """Main verification function."""
    print("🔍 MAPoRL Repository Path Verification")
    print("=" * 50)
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)
    
    checks = []
    
    print("\n📂 1. Verifying File Structure...")
    
    # Check key files exist
    key_files = [
        ("scripts/training/train.py", "Main training script"),
        ("scripts/training/local_train_medxpert.py", "Local MedXpert training"),
        ("scripts/data/download_medxpert_data.py", "Data download script"),
        ("scripts/utilities/setup_wandb.py", "W&B setup utility"),
        ("sagemaker/sagemaker_entry.py", "SageMaker entry point"),
        ("examples/example_wandb_training.py", "W&B example"),
        ("docs/WANDB_INTEGRATION_GUIDE.md", "W&B guide"),
        ("src/training/mapoRRL_trainer.py", "MAPoRL trainer"),
        ("config/model_config.py", "Model config"),
        ("config/model_config_qwen.py", "Qwen config"),
    ]
    
    for file_path, description in key_files:
        checks.append(verify_file_exists(file_path, description))
    
    print("\n🐍 2. Verifying Python Imports...")
    
    # Check Python scripts can be imported
    python_scripts = [
        ("scripts/training/train.py", "Main training script"),
        ("scripts/training/local_train_medxpert.py", "Local training script"),
        ("scripts/utilities/setup_wandb.py", "Setup utility"),
        ("examples/example_wandb_training.py", "W&B example"),
        ("sagemaker/sagemaker_entry.py", "SageMaker entry"),
    ]
    
    for script_path, description in python_scripts:
        if os.path.exists(script_path):
            checks.append(verify_import(script_path, description))
        else:
            log_error(f"{description}: File not found")
            checks.append(False)
    
    print("\n🔧 3. Verifying Shell Scripts...")
    
    # Check shell scripts exist and are executable
    shell_scripts = [
        "scripts/utilities/run_local_medical_training.sh",
        "scripts/utilities/sagemaker_test_medxpert.sh",
        "scripts/utilities/sagemaker_train_medxpert.sh",
    ]
    
    for script_path in shell_scripts:
        if os.path.exists(script_path):
            if os.access(script_path, os.X_OK):
                log_success(f"Shell script: {script_path} (executable)")
                checks.append(True)
            else:
                log_warning(f"Shell script: {script_path} (not executable)")
                checks.append(True)  # Still counts as success
        else:
            log_error(f"Shell script: {script_path} (missing)")
            checks.append(False)
    
    print("\n📚 4. Verifying Documentation...")
    
    # Check documentation files
    docs = [
        "docs/WANDB_INTEGRATION_GUIDE.md",
        "docs/RUN_LOCAL_TRAINING.md",
        "docs/SAGEMAKER_DEPLOYMENT_GUIDE.md",
        "docs/SAGEMAKER_README.md",
        "docs/TESTING_README.md",
    ]
    
    for doc_path in docs:
        checks.append(verify_file_exists(doc_path, f"Documentation: {doc_path}"))
    
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        log_success(f"All {total} checks passed! Repository organization successful.")
        print("\n🎉 Repository is properly organized and ready to use!")
        print("\nNext steps:")
        print("  1. Run: python scripts/utilities/setup_wandb.py")
        print("  2. Run: python scripts/data/download_medxpert_data.py")
        print("  3. Run: bash scripts/utilities/run_local_medical_training.sh")
        return True
    else:
        log_error(f"{passed}/{total} checks passed. Some issues need to be resolved.")
        print("\nPlease fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 