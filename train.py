"""
Training script for Medical Multi-Agent Pipeline with MAPoRL.
"""

import os
import sys
import logging
import argparse
import json
import wandb
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.maporl_trainer import create_trainer, load_medical_dataset, MAPoRLConfig
from src.config.model_config import MULTI_AGENT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Medical Multi-Agent Pipeline with MAPoRL")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--eval_data", type=str, help="Path to evaluation data")
    parser.add_argument("--max_train_samples", type=int, help="Maximum training samples")
    parser.add_argument("--max_eval_samples", type=int, help="Maximum evaluation samples")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_rounds", type=int, default=3, help="Max rounds per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip_ratio", type=float, default=0.2, help="PPO clip ratio")
    
    # Reward system arguments
    parser.add_argument("--safety_penalty_weight", type=float, default=2.0, help="Safety penalty weight")
    parser.add_argument("--collaboration_bonus_weight", type=float, default=1.5, help="Collaboration bonus weight")
    parser.add_argument("--medical_relevance_weight", type=float, default=1.2, help="Medical relevance weight")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training")
    parser.add_argument("--save_every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    
    # Logging arguments
    parser.add_argument("--wandb_project", type=str, default="maporl-medical", help="W&B project name")
    parser.add_argument("--experiment_name", type=str, help="Experiment name")
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=["multi-agent", "medical", "maporl"], help="W&B tags")
    parser.add_argument("--wandb_group", type=str, help="W&B group name")
    parser.add_argument("--wandb_notes", type=str, help="W&B run notes")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging")
    
    return parser.parse_args()

def create_config(args) -> MAPoRLConfig:
    """Create training configuration from arguments."""
    return MAPoRLConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_rounds_per_episode=args.max_rounds,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        save_every=args.save_every,
        eval_every=args.eval_every,
        safety_penalty_weight=args.safety_penalty_weight,
        collaboration_bonus_weight=args.collaboration_bonus_weight,
        medical_relevance_weight=args.medical_relevance_weight
    )

def setup_wandb(args, config):
    """Setup Weights & Biases logging."""
    if args.disable_wandb:
        logger.info("W&B logging disabled")
        return
    
    try:
        # Initialize wandb
        wandb_config = {
            **vars(args),
            "model_config": {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "max_rounds_per_episode": config.max_rounds_per_episode,
                "gamma": config.gamma,
                "clip_ratio": config.clip_ratio,
                "safety_penalty_weight": config.safety_penalty_weight,
                "collaboration_bonus_weight": config.collaboration_bonus_weight,
                "medical_relevance_weight": config.medical_relevance_weight
            }
        }
        
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=wandb_config,
            resume="allow"
        )
        
        # Log system information
        wandb.log({
            "system/gpu_count": len(args.device.split(',')) if ',' in args.device else 1,
            "system/device": args.device,
        })
        
        logger.info(f"W&B initialized: {wandb.run.url}")
        
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        logger.info("Continuing without W&B logging")

def setup_output_dir(output_dir: str, experiment_name: str = None):
    """Setup output directory."""
    if experiment_name:
        output_dir = os.path.join(output_dir, experiment_name)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    return output_dir

def main():
    """Main training function."""
    args = parse_args()
    
    logger.info("Starting MAPoRL Medical Multi-Agent Training")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Setup output directory
        output_dir = setup_output_dir(args.output_dir, args.experiment_name)
        logger.info(f"Output directory: {output_dir}")
        
        # Create training configuration
        config = create_config(args)
        logger.info(f"Training configuration: {config}")
        
        # Setup W&B logging
        setup_wandb(args, config)
        
        # Load datasets
        logger.info(f"Loading training data from: {args.train_data}")
        train_dataset = load_medical_dataset(args.train_data, args.max_train_samples)
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        # Log data statistics to W&B
        if not args.disable_wandb:
            wandb.log({
                "data/train_samples": len(train_dataset),
                "data/train_data_path": args.train_data,
            })
        
        eval_dataset = None
        if args.eval_data:
            logger.info(f"Loading evaluation data from: {args.eval_data}")
            eval_dataset = load_medical_dataset(args.eval_data, args.max_eval_samples)
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
            
            if not args.disable_wandb:
                wandb.log({
                    "data/eval_samples": len(eval_dataset),
                    "data/eval_data_path": args.eval_data,
                })
        
        # Create trainer
        logger.info("Creating MAPoRL trainer...")
        trainer = create_trainer(
            config=config,
            workflow_config=None,  # Use default
            device=args.device
        )
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        config_dict = {
            "args": vars(args),
            "config": {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "max_rounds_per_episode": config.max_rounds_per_episode,
                "gamma": config.gamma,
                "clip_ratio": config.clip_ratio,
                "safety_penalty_weight": config.safety_penalty_weight,
                "collaboration_bonus_weight": config.collaboration_bonus_weight,
                "medical_relevance_weight": config.medical_relevance_weight
            }
        }
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        
        # Log config as artifact to W&B
        if not args.disable_wandb:
            artifact = wandb.Artifact("training_config", type="config")
            artifact.add_file(config_path)
            wandb.log_artifact(artifact)
        
        # Start training
        logger.info("Starting training...")
        trainer.train(train_dataset, eval_dataset)
        
        logger.info("Training completed successfully!")
        
        # Mark run as finished
        if not args.disable_wandb:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if not args.disable_wandb:
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main() 