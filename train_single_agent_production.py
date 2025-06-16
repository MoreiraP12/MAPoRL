#!/usr/bin/env python3
"""
Production script to train a single agent at a time
Usage: python train_single_agent_production.py --agent planner
"""

import os
import sys
import argparse
import torch

# Add root directory to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import the main trainer class
from scripts.training.local_train_medxpert import LocalMedicalTrainer

def main():
    parser = argparse.ArgumentParser(description="Train a single medical agent")
    parser.add_argument("--agent", type=str, required=True, 
                       choices=["planner", "researcher", "analyst", "reporter"],
                       help="Which agent to train")
    parser.add_argument("--data-dir", type=str, default="data", 
                       help="Directory containing JSONL data files")
    parser.add_argument("--output-dir", type=str, default="outputs", 
                       help="Output directory for models and results")
    parser.add_argument("--disable-wandb", action="store_true", 
                       help="Disable W&B logging")
    
    args = parser.parse_args()
    
    print(f"🏥 Training {args.agent.upper()} Agent")
    print("=" * 50)
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🖥️ Available GPUs: {torch.cuda.device_count()}")
    print("=" * 50)
    
    # Setup W&B configuration
    wandb_config = {
        "project": "maporl-medxpert-single",
        "name": f"{args.agent}-agent-solo",
        "group": "single-agent-training",
        "tags": ["medxpert", "qwen", "single-agent", args.agent],
        "notes": f"Single agent training for {args.agent}",
        "disable_wandb": args.disable_wandb,
    }
    
    # Create trainer instance
    trainer = LocalMedicalTrainer(args.data_dir, args.output_dir, wandb_config)
    
    try:
        # Load datasets
        train_file = os.path.join(args.data_dir, "medxpert_train.jsonl")
        val_file = os.path.join(args.data_dir, "medxpert_validation.jsonl")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data not found at {train_file}")
        
        train_dataset = trainer.load_jsonl_data(train_file)
        eval_dataset = trainer.load_jsonl_data(val_file) if os.path.exists(val_file) else train_dataset.train_test_split(test_size=0.2)["test"]
        
        # Train the specified agent
        print(f"\n🚀 Starting {args.agent} agent training...")
        result = trainer.train_agent(args.agent, train_dataset, eval_dataset)
        
        print(f"\n✅ {args.agent} Agent Training Complete!")
        print(f"📊 Final loss: {result.get('train_loss', 'N/A')}")
        print(f"⏱️ Training time: {result.get('training_time', 'N/A')} seconds")
        print(f"📁 Model saved to: {args.output_dir}/models/{args.agent}")
        
        # Finish W&B run
        if not args.disable_wandb:
            import wandb
            wandb.finish()
            
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        if not args.disable_wandb:
            import wandb
            wandb.finish(exit_code=1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if not args.disable_wandb:
            import wandb
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main() 