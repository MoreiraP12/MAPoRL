#!/usr/bin/env python3
"""
W&B Setup and Test Utility for MAPoRL Medical Training

This script helps you set up and test your Weights & Biases integration
for the MAPoRL medical multi-agent training pipeline.
"""

import os
import sys
import json
import wandb
import argparse
from datetime import datetime
from typing import Dict, Any

def check_wandb_installation():
    """Check if W&B is properly installed."""
    try:
        import wandb
        print(f"✅ W&B installed (version: {wandb.__version__})")
        return True
    except ImportError:
        print("❌ W&B not installed. Install with: pip install wandb")
        return False

def check_wandb_login():
    """Check if user is logged into W&B."""
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Logged in as: {user.username}")
        return True
    except Exception as e:
        print(f"❌ Not logged in to W&B: {e}")
        print("   Login with: wandb login")
        return False

def test_wandb_logging():
    """Test basic W&B logging functionality."""
    try:
        # Initialize a test run
        wandb.init(
            project="maporl-test",
            name=f"test-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=["test"],
            mode="online"  # Force online mode for testing
        )
        
        # Log some test data
        for i in range(10):
            wandb.log({
                "test_metric": i * 0.1,
                "test_loss": 1.0 - i * 0.05,
                "step": i
            })
        
        # Create a test table
        table = wandb.Table(
            columns=["epoch", "accuracy", "loss"],
            data=[[i, i * 0.1, 1.0 - i * 0.05] for i in range(5)]
        )
        wandb.log({"test_table": table})
        
        # Create a test artifact
        artifact = wandb.Artifact("test-config", type="config")
        config_data = {"test": True, "timestamp": datetime.now().isoformat()}
        
        with open("test_config.json", "w") as f:
            json.dump(config_data, f)
        
        artifact.add_file("test_config.json")
        wandb.log_artifact(artifact)
        
        # Clean up
        os.remove("test_config.json")
        
        run_url = wandb.run.url
        wandb.finish()
        
        print(f"✅ W&B logging test successful!")
        print(f"🔗 Test run URL: {run_url}")
        return True
        
    except Exception as e:
        print(f"❌ W&B logging test failed: {e}")
        return False

def setup_environment_variables():
    """Help user set up environment variables."""
    print("\n🔧 Setting up environment variables...")
    
    # Get current values
    current_vars = {
        "WANDB_PROJECT": os.getenv("WANDB_PROJECT"),
        "WANDB_ENTITY": os.getenv("WANDB_ENTITY"),
        "WANDB_DIR": os.getenv("WANDB_DIR"),
        "WANDB_API_KEY": os.getenv("WANDB_API_KEY", "***" if os.getenv("WANDB_API_KEY") else None)
    }
    
    print("\nCurrent environment variables:")
    for var, value in current_vars.items():
        status = "✅" if value else "❌"
        print(f"  {status} {var}: {value or 'Not set'}")
    
    print("\nRecommended settings for MAPoRL:")
    print("export WANDB_PROJECT='maporl-medical'")
    print("export WANDB_ENTITY='your-username'")
    print("export WANDB_DIR='./outputs/wandb'")
    print("export WANDB_API_KEY='your-api-key'")
    
    return all(current_vars.values())

def create_wandb_config_template():
    """Create a template configuration file for W&B settings."""
    config = {
        "wandb": {
            "project": "maporl-medical",
            "entity": "your-username",
            "tags": ["medical", "multi-agent", "maporl"],
            "group": "experiments",
            "notes": "MAPoRL medical multi-agent training",
            "log_model": "checkpoint",
            "watch": "all"
        },
        "training": {
            "learning_rate": 1e-5,
            "batch_size": 2,
            "num_epochs": 10,
            "max_rounds_per_episode": 3,
            "gamma": 0.99,
            "clip_ratio": 0.2
        },
        "medical": {
            "safety_penalty_weight": 2.0,
            "collaboration_bonus_weight": 1.5,
            "medical_relevance_weight": 1.2
        }
    }
    
    config_file = "wandb_config_template.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ W&B config template created: {config_file}")
    print("   Edit this file with your settings and use with your training scripts")
    
    return config_file

def check_project_structure():
    """Check if the project structure is set up correctly for W&B integration."""
    required_files = [
        "train.py",
        "local_train_medxpert.py",
        "requirements.txt",
        "src/training/mapoRRL_trainer.py"
    ]
    
    optional_files = [
        "sagemaker_entry.py",
        "WANDB_INTEGRATION_GUIDE.md",
        "example_wandb_training.py"
    ]
    
    print("\n📁 Checking project structure...")
    
    missing_required = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (required)")
            missing_required.append(file)
    
    for file in optional_files:
        if os.path.exists(file):
            print(f"  ✅ {file} (optional)")
        else:
            print(f"  ⚠️  {file} (optional)")
    
    if missing_required:
        print(f"\n❌ Missing required files: {missing_required}")
        return False
    else:
        print("\n✅ Project structure looks good!")
        return True

def run_integration_check():
    """Run comprehensive integration check."""
    print("🔍 Running W&B integration check for MAPoRL...")
    print("=" * 60)
    
    checks = []
    
    # Check installation
    print("\n1. Checking W&B installation...")
    checks.append(check_wandb_installation())
    
    # Check login
    print("\n2. Checking W&B login...")
    checks.append(check_wandb_login())
    
    # Check environment variables
    print("\n3. Checking environment variables...")
    checks.append(setup_environment_variables())
    
    # Check project structure
    print("\n4. Checking project structure...")
    checks.append(check_project_structure())
    
    # Test logging (only if previous checks pass)
    if all(checks):
        print("\n5. Testing W&B logging...")
        checks.append(test_wandb_logging())
    else:
        print("\n5. Skipping W&B logging test (fix previous issues first)")
        checks.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION CHECK SUMMARY:")
    print("=" * 60)
    
    check_names = [
        "W&B Installation",
        "W&B Login",
        "Environment Variables",
        "Project Structure",
        "W&B Logging Test"
    ]
    
    for name, passed in zip(check_names, checks):
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} {name}")
    
    if all(checks):
        print("\n🎉 All checks passed! Your W&B integration is ready.")
        print("\nNext steps:")
        print("  1. Run: python example_wandb_training.py")
        print("  2. Run: python train.py --train_data your_data.jsonl")
        print("  3. Run: python local_train_medxpert.py --data-dir data")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help:")
        print("  - W&B Documentation: https://docs.wandb.ai/")
        print("  - MAPoRL W&B Guide: ./WANDB_INTEGRATION_GUIDE.md")
    
    return all(checks)

def main():
    parser = argparse.ArgumentParser(description="W&B Setup and Test Utility for MAPoRL")
    parser.add_argument("--check", action="store_true", help="Run integration check")
    parser.add_argument("--test", action="store_true", help="Test W&B logging only")
    parser.add_argument("--setup", action="store_true", help="Setup environment variables")
    parser.add_argument("--template", action="store_true", help="Create config template")
    parser.add_argument("--all", action="store_true", help="Run all setup and checks")
    
    args = parser.parse_args()
    
    if args.all or not any([args.check, args.test, args.setup, args.template]):
        # Default: run full integration check
        run_integration_check()
        create_wandb_config_template()
    else:
        if args.check:
            run_integration_check()
        if args.test:
            test_wandb_logging()
        if args.setup:
            setup_environment_variables()
        if args.template:
            create_wandb_config_template()

if __name__ == "__main__":
    print("🏥 MAPoRL W&B Setup Utility")
    print("Helping you set up Weights & Biases for medical multi-agent training")
    print("=" * 70)
    
    main() 