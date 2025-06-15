#!/usr/bin/env python3
"""
Example script showing comprehensive W&B integration for MAPoRL training.
This script demonstrates how to properly integrate Weights & Biases logging
into your medical multi-agent training pipeline.
"""

import os
import sys
import json
import torch
import wandb
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WandBMedicalTrainer:
    """
    Example trainer class with comprehensive W&B integration.
    Shows how to log system info, training metrics, medical-specific metrics,
    and create artifacts.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.setup_wandb()
        
    def setup_wandb(self):
        """Initialize W&B with comprehensive configuration."""
        try:
            # Initialize W&B run
            wandb.init(
                project=self.config.get("wandb_project", "maporl-medical"),
                name=self.config.get("wandb_name", f"medical-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                group=self.config.get("wandb_group", "examples"),
                tags=self.config.get("wandb_tags", ["medical", "multi-agent", "example"]),
                notes=self.config.get("wandb_notes", "Example W&B integration for MAPoRL"),
                config=self.config
            )
            
            # Log system information
            self.log_system_info()
            
            logger.info(f"🚀 W&B initialized: {wandb.run.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            raise
    
    def log_system_info(self):
        """Log comprehensive system information."""
        system_info = {
            "system/cuda_available": torch.cuda.is_available(),
            "system/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "system/pytorch_version": torch.__version__,
            "system/python_version": sys.version,
            "system/platform": sys.platform,
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                system_info[f"system/gpu_{i}_name"] = gpu_name
                system_info[f"system/gpu_{i}_memory_gb"] = gpu_memory
        
        wandb.log(system_info)
        logger.info("✅ System information logged to W&B")
    
    def log_dataset_info(self, train_data: List[Dict], eval_data: List[Dict] = None):
        """Log dataset statistics and information."""
        dataset_info = {
            "data/train_samples": len(train_data),
            "data/has_eval_data": eval_data is not None,
        }
        
        if eval_data:
            dataset_info["data/eval_samples"] = len(eval_data)
        
        # Analyze categories if available
        if train_data and "category" in train_data[0]:
            categories = [item["category"] for item in train_data]
            category_counts = {cat: categories.count(cat) for cat in set(categories)}
            dataset_info["data/categories"] = category_counts
            dataset_info["data/num_categories"] = len(category_counts)
        
        # Analyze difficulty levels if available
        if train_data and "difficulty" in train_data[0]:
            difficulties = [item["difficulty"] for item in train_data]
            difficulty_counts = {diff: difficulties.count(diff) for diff in set(difficulties)}
            dataset_info["data/difficulties"] = difficulty_counts
        
        wandb.log(dataset_info)
        
        # Create dataset table for visualization
        if len(train_data) <= 100:  # Only for small datasets
            table_data = []
            for i, item in enumerate(train_data[:50]):  # Show first 50 samples
                table_data.append([
                    i,
                    item.get("question", "")[:100] + "..." if len(item.get("question", "")) > 100 else item.get("question", ""),
                    item.get("category", "unknown"),
                    item.get("difficulty", "unknown"),
                    len(item.get("answer", ""))
                ])
            
            table = wandb.Table(
                columns=["Index", "Question (Preview)", "Category", "Difficulty", "Answer Length"],
                data=table_data
            )
            wandb.log({"data/sample_table": table})
        
        logger.info("✅ Dataset information logged to W&B")
    
    def log_training_metrics(self, epoch: int, agent: str, metrics: Dict[str, float]):
        """Log training metrics for a specific agent."""
        log_data = {}
        
        # Add epoch and agent info
        log_data["epoch"] = epoch
        log_data["agent"] = agent
        
        # Add metrics with proper namespacing
        for metric_name, value in metrics.items():
            log_data[f"{agent}/{metric_name}"] = value
            log_data[f"global/{metric_name}"] = value  # Also log globally for comparison
        
        wandb.log(log_data)
    
    def log_medical_metrics(self, predictions: List[str], ground_truths: List[str], categories: List[str]):
        """Log medical-specific evaluation metrics."""
        from sklearn.metrics import accuracy_score
        import numpy as np
        
        # Calculate basic accuracy (simplified)
        # In practice, you'd use more sophisticated medical evaluation metrics
        
        medical_metrics = {}
        
        # Safety assessment (count unsafe phrases)
        unsafe_phrases = ["definitely", "certainly", "100% sure", "impossible", "never", "always"]
        safe_phrases = ["consider", "may", "could", "suggest", "recommend", "consult"]
        
        safety_scores = []
        relevance_scores = []
        
        medical_terms = [
            "diagnosis", "treatment", "patient", "symptom", "condition",
            "medication", "therapy", "clinical", "medical", "disease"
        ]
        
        for pred, truth, category in zip(predictions, ground_truths, categories):
            pred_lower = pred.lower()
            
            # Safety score
            unsafe_count = sum(1 for phrase in unsafe_phrases if phrase in pred_lower)
            safe_count = sum(1 for phrase in safe_phrases if phrase in pred_lower)
            safety_score = max(0.0, 1.0 - 0.2 * unsafe_count + 0.1 * safe_count)
            safety_scores.append(safety_score)
            
            # Medical relevance score
            medical_count = sum(1 for term in medical_terms if term in pred_lower)
            relevance_score = min(1.0, medical_count / 5.0)  # Normalize to 0-1
            relevance_scores.append(relevance_score)
        
        medical_metrics["medical/average_safety_score"] = np.mean(safety_scores)
        medical_metrics["medical/average_relevance_score"] = np.mean(relevance_scores)
        medical_metrics["medical/num_predictions"] = len(predictions)
        
        # Category-specific metrics
        unique_categories = list(set(categories))
        for category in unique_categories:
            category_indices = [i for i, c in enumerate(categories) if c == category]
            if category_indices:
                category_safety = np.mean([safety_scores[i] for i in category_indices])
                category_relevance = np.mean([relevance_scores[i] for i in category_indices])
                
                medical_metrics[f"medical/safety_{category}"] = category_safety
                medical_metrics[f"medical/relevance_{category}"] = category_relevance
        
        wandb.log(medical_metrics)
        
        # Create metrics distribution table
        metrics_table = wandb.Table(
            columns=["Category", "Safety Score", "Relevance Score", "Prediction Preview"],
            data=[[cat, safety, relevance, pred[:100] + "..." if len(pred) > 100 else pred] 
                  for cat, safety, relevance, pred in zip(categories[:20], safety_scores[:20], relevance_scores[:20], predictions[:20])]
        )
        wandb.log({"medical/metrics_table": metrics_table})
        
        logger.info("✅ Medical metrics logged to W&B")
    
    def log_agent_collaboration(self, collaboration_data: Dict[str, Any]):
        """Log agent collaboration metrics."""
        collab_metrics = {
            "collaboration/total_interactions": collaboration_data.get("total_interactions", 0),
            "collaboration/successful_handoffs": collaboration_data.get("successful_handoffs", 0),
            "collaboration/average_response_time": collaboration_data.get("avg_response_time", 0.0),
            "collaboration/consensus_reached": collaboration_data.get("consensus_reached", 0),
        }
        
        if collaboration_data.get("agent_contributions"):
            for agent, contribution in collaboration_data["agent_contributions"].items():
                collab_metrics[f"collaboration/{agent}_contribution"] = contribution
        
        wandb.log(collab_metrics)
        logger.info("✅ Collaboration metrics logged to W&B")
    
    def create_training_artifacts(self, output_dir: str):
        """Create and log training artifacts."""
        try:
            # Create training results artifact
            results_artifact = wandb.Artifact("training_results", type="results")
            
            # Add configuration file
            config_file = os.path.join(output_dir, "config.json")
            with open(config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            results_artifact.add_file(config_file)
            
            # Add training log (if exists)
            log_file = os.path.join(output_dir, "training.log")
            if os.path.exists(log_file):
                results_artifact.add_file(log_file)
            
            # Add any model files (if they exist)
            model_dir = os.path.join(output_dir, "models")
            if os.path.exists(model_dir):
                results_artifact.add_dir(model_dir, name="models")
            
            wandb.log_artifact(results_artifact)
            logger.info("✅ Training artifacts logged to W&B")
            
        except Exception as e:
            logger.warning(f"Failed to create artifacts: {e}")
    
    def run_example_training(self):
        """Run an example training loop with W&B logging."""
        logger.info("🚀 Starting example training with W&B integration")
        
        # Simulate dataset
        train_data = [
            {
                "question": f"What is the treatment for condition {i}?",
                "answer": f"The treatment for condition {i} includes medication and therapy.",
                "category": ["cardiology", "neurology", "oncology"][i % 3],
                "difficulty": ["easy", "medium", "hard"][i % 3]
            }
            for i in range(100)
        ]
        
        eval_data = train_data[-20:]  # Use last 20 for evaluation
        
        # Log dataset information
        self.log_dataset_info(train_data, eval_data)
        
        # Simulate multi-agent training
        agents = ["planner", "researcher", "analyst", "reporter"]
        
        for epoch in range(self.config.get("num_epochs", 3)):
            logger.info(f"📈 Epoch {epoch + 1}")
            
            epoch_metrics = {}
            
            for agent in agents:
                # Simulate agent training
                agent_metrics = {
                    "loss": max(0.1, 2.0 - epoch * 0.3 + torch.randn(1).item() * 0.1),
                    "accuracy": min(0.95, 0.5 + epoch * 0.15 + torch.randn(1).item() * 0.05),
                    "learning_rate": self.config.get("learning_rate", 1e-5) * (0.9 ** epoch),
                    "training_time": torch.rand(1).item() * 100 + 50  # 50-150 seconds
                }
                
                # Log agent metrics
                self.log_training_metrics(epoch, agent, agent_metrics)
                
                # Store for epoch summary
                for metric, value in agent_metrics.items():
                    if metric not in epoch_metrics:
                        epoch_metrics[metric] = []
                    epoch_metrics[metric].append(value)
                
                logger.info(f"  {agent}: loss={agent_metrics['loss']:.4f}, acc={agent_metrics['accuracy']:.4f}")
            
            # Log epoch summary
            epoch_summary = {
                f"epoch_summary/avg_{metric}": sum(values) / len(values)
                for metric, values in epoch_metrics.items()
            }
            epoch_summary["epoch"] = epoch
            wandb.log(epoch_summary)
            
            # Simulate evaluation every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.run_evaluation(eval_data)
        
        # Simulate collaboration metrics
        collaboration_data = {
            "total_interactions": 150,
            "successful_handoffs": 142,
            "avg_response_time": 1.2,
            "consensus_reached": 95,
            "agent_contributions": {
                "planner": 0.25,
                "researcher": 0.30,
                "analyst": 0.25,
                "reporter": 0.20
            }
        }
        self.log_agent_collaboration(collaboration_data)
        
        # Create artifacts
        output_dir = self.config.get("output_dir", "./outputs")
        os.makedirs(output_dir, exist_ok=True)
        self.create_training_artifacts(output_dir)
        
        logger.info("🎉 Example training completed!")
    
    def run_evaluation(self, eval_data: List[Dict]):
        """Run evaluation and log results."""
        logger.info("🔍 Running evaluation...")
        
        # Simulate predictions
        predictions = [
            f"Based on the medical evidence, I suggest considering {item['category']}-specific treatment approaches. "
            f"Please consult with a specialist for proper diagnosis and treatment planning."
            for item in eval_data
        ]
        
        ground_truths = [item["answer"] for item in eval_data]
        categories = [item["category"] for item in eval_data]
        
        # Log medical-specific metrics
        self.log_medical_metrics(predictions, ground_truths, categories)
        
        # Log evaluation metrics
        eval_metrics = {
            "eval/num_samples": len(eval_data),
            "eval/avg_prediction_length": sum(len(p) for p in predictions) / len(predictions),
            "eval/categories_evaluated": len(set(categories))
        }
        
        wandb.log(eval_metrics)

def main():
    """Main function demonstrating W&B integration."""
    # Configuration
    config = {
        # Training parameters
        "learning_rate": 1e-5,
        "batch_size": 2,
        "num_epochs": 3,
        "max_rounds_per_episode": 3,
        
        # MAPoRL parameters
        "gamma": 0.99,
        "clip_ratio": 0.2,
        "safety_penalty_weight": 2.0,
        "collaboration_bonus_weight": 1.5,
        "medical_relevance_weight": 1.2,
        
        # W&B configuration
        "wandb_project": "maporl-medical-example",
        "wandb_name": f"example-run-{datetime.now().strftime('%H%M%S')}",
        "wandb_group": "examples",
        "wandb_tags": ["example", "medical", "multi-agent", "maporl"],
        "wandb_notes": "Example script demonstrating comprehensive W&B integration for MAPoRL medical training",
        
        # Output
        "output_dir": "./outputs/example_wandb_run"
    }
    
    try:
        # Initialize trainer with W&B
        trainer = WandBMedicalTrainer(config)
        
        # Run example training
        trainer.run_example_training()
        
        # Finish W&B run
        wandb.finish()
        
        print("\n🎉 Example completed successfully!")
        print(f"🔗 Check your results at: {wandb.run.url if wandb.run else 'N/A'}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        wandb.finish(exit_code=1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    print("🏥 MAPoRL Medical Training - W&B Integration Example")
    print("=" * 60)
    print("This script demonstrates comprehensive W&B logging for")
    print("medical multi-agent training with MAPoRL.")
    print("=" * 60)
    
    main() 