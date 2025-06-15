"""
Training script for Medical Multi-Agent Pipeline with MAPoRL.
"""

import os
import sys
import logging
import argparse
import json
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
        
        # Load datasets
        logger.info(f"Loading training data from: {args.train_data}")
        train_dataset = load_medical_dataset(args.train_data, args.max_train_samples)
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
        eval_dataset = None
        if args.eval_data:
            logger.info(f"Loading evaluation data from: {args.eval_data}")
            eval_dataset = load_medical_dataset(args.eval_data, args.max_eval_samples)
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Create trainer
        logger.info("Creating MAPoRL trainer...")
        trainer = create_trainer(
            config=config,
            workflow_config=None,  # Use default
            device=args.device
        )
        
        # Save configuration
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump({
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
            }, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(train_dataset, eval_dataset)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 