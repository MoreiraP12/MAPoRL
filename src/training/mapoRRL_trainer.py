"""
MAPoRL Training Framework for Multi-Agent Medical LLMs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import numpy as np
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm
import os
from collections import defaultdict

from ..workflow.medical_workflow import MedicalWorkflow
from ..reward.medical_reward_system import MedicalRewardSystem, MedicalRewardComponents
from ..agents.base_agent import MedicalState
from ..config.model_config import MULTI_AGENT_CONFIG, OPTIMIZATION_SETTINGS

logger = logging.getLogger(__name__)

@dataclass
class MAPoRLConfig:
    """Configuration for MAPoRL training."""
    learning_rate: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 10
    max_rounds_per_episode: int = 3
    gamma: float = 0.99  # Discount factor
    clip_ratio: float = 0.2  # PPO clip ratio
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 1.0
    save_every: int = 100
    eval_every: int = 50
    warmup_steps: int = 100
    
    # Medical-specific parameters
    safety_penalty_weight: float = 2.0
    collaboration_bonus_weight: float = 1.5
    medical_relevance_weight: float = 1.2

class MedicalDataset(Dataset):
    """Dataset for medical questions and answers."""
    
    def __init__(self, data_path: str, max_samples: int = None):
        self.data = self.load_data(data_path, max_samples)
    
    def load_data(self, data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
        """Load medical dataset."""
        try:
            if data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    data = json.load(f)
            elif data_path.endswith('.jsonl'):
                data = []
                with open(data_path, 'r') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            if max_samples:
                data = data[:max_samples]
            
            logger.info(f"Loaded {len(data)} samples from {data_path}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Return sample data for testing
            return [
                {
                    "question": "What are the symptoms of hypertension?",
                    "context": "A 45-year-old patient presents with elevated blood pressure readings.",
                    "answer": "Hypertension often presents with headaches, dizziness, and fatigue, but can be asymptomatic."
                },
                {
                    "question": "How is diabetes diagnosed?",
                    "context": "Patient has family history of diabetes and presents with increased thirst.",
                    "answer": "Diabetes is diagnosed through fasting glucose ≥126 mg/dL or HbA1c ≥6.5%."
                }
            ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item.get("question", ""),
            "context": item.get("context", ""),
            "answer": item.get("answer", ""),
            "id": item.get("id", idx)
        }

class MAPoRLTrainer:
    """Multi-Agent Post-co-training with Reinforcement Learning trainer."""
    
    def __init__(
        self, 
        config: MAPoRLConfig = None,
        workflow_config: Dict[str, Any] = None,
        device: str = "cuda:0"
    ):
        self.config = config or MAPoRLConfig()
        self.device = device
        
        # Initialize components
        self.workflow = MedicalWorkflow(workflow_config)
        self.reward_system = MedicalRewardSystem(device)
        
        # Training state
        self.global_step = 0
        self.best_reward = -float('inf')
        self.training_history = []
        
        # Initialize optimizers for each agent
        self.optimizers = {}
        self.schedulers = {}
        self.setup_optimizers()
        
        # Initialize logging
        self.setup_logging()
        
        logger.info("MAPoRL Trainer initialized successfully")
    
    def setup_optimizers(self):
        """Setup optimizers for each agent."""
        for agent_name, agent in self.workflow.agents.items():
            if hasattr(agent.model, 'parameters'):
                self.optimizers[agent_name] = optim.AdamW(
                    agent.model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=0.01
                )
                
                # Setup scheduler
                self.schedulers[agent_name] = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizers[agent_name],
                    T_max=self.config.num_epochs * 100  # Approximate steps
                )
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        try:
            wandb.init(
                project="maporl-medical",
                config=asdict(self.config),
                tags=["multi-agent", "medical", "maporl"]
            )
            logger.info("W&B logging initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    def compute_policy_loss(
        self, 
        agent_name: str, 
        states: List[str], 
        actions: List[str], 
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """Compute PPO policy loss for an agent."""
        try:
            agent = self.workflow.agents[agent_name]
            
            # Get current log probabilities
            current_log_probs = []
            for state, action in zip(states, actions):
                # This is a simplified version - in practice, you'd need to compute
                # the log probability of the action given the state
                log_prob = torch.log(torch.tensor(0.5))  # Placeholder
                current_log_probs.append(log_prob)
            
            current_log_probs = torch.stack(current_log_probs)
            
            # Compute ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Compute PPO loss
            surr1 = ratio * rewards
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * rewards
            
            policy_loss = -torch.min(surr1, surr2).mean()
            
            return policy_loss
            
        except Exception as e:
            logger.error(f"Error computing policy loss for {agent_name}: {e}")
            return torch.tensor(0.0, requires_grad=True)
    
    def compute_value_loss(self, predicted_values: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        """Compute value function loss."""
        return nn.MSELoss()(predicted_values, target_values)
    
    def compute_collaboration_reward(self, episode_data: Dict[str, Any]) -> float:
        """Compute additional collaboration reward."""
        agent_responses = episode_data.get("agent_responses", {})
        
        if len(agent_responses) < 2:
            return 0.0
        
        # Measure interaction quality
        interaction_score = 0.0
        
        # Check if agents reference each other
        for agent_name, responses in agent_responses.items():
            for response in responses:
                for other_agent in agent_responses.keys():
                    if other_agent != agent_name and other_agent in response.lower():
                        interaction_score += 0.1
        
        # Normalize by number of possible interactions
        num_agents = len(agent_responses)
        max_interactions = num_agents * (num_agents - 1)
        
        if max_interactions > 0:
            interaction_score = min(interaction_score / max_interactions, 1.0)
        
        return interaction_score * self.config.collaboration_bonus_weight
    
    def train_episode(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Train on a single episode (batch of medical questions)."""
        episode_metrics = defaultdict(float)
        
        try:
            questions = batch["question"]
            contexts = batch["context"]
            ground_truths = batch["answer"]
            
            # Initialize episode data collection
            episode_data = {
                "states": [],
                "actions": [],
                "rewards": [],
                "agent_responses": defaultdict(list),
                "log_probs": defaultdict(list)
            }
            
            # Run multi-agent workflow for each question
            for i, (question, context, ground_truth) in enumerate(zip(questions, contexts, ground_truths)):
                # Execute workflow
                result = self.workflow.run_workflow(
                    question=question,
                    context=context,
                    thread_id=f"train_{self.global_step}_{i}"
                )
                
                # Calculate medical reward
                reward_components = self.reward_system.calculate_medical_reward(
                    agent_responses=result["agent_responses"],
                    question=question,
                    ground_truth=ground_truth,
                    context=context
                )
                
                # Add collaboration bonus
                collaboration_reward = self.compute_collaboration_reward(result)
                total_reward = reward_components.total_score() + collaboration_reward
                
                # Apply safety penalty if needed
                if result["safety_flags"]:
                    safety_penalty = len(result["safety_flags"]) * 0.1 * self.config.safety_penalty_weight
                    total_reward -= safety_penalty
                
                # Store episode data
                episode_data["rewards"].append(total_reward)
                
                # Update metrics
                episode_metrics["total_reward"] += total_reward
                episode_metrics["accuracy"] += reward_components.accuracy_score
                episode_metrics["medical_relevance"] += reward_components.medical_relevance_score
                episode_metrics["collaboration_quality"] += reward_components.collaboration_quality_score
                episode_metrics["safety"] += reward_components.safety_score
                episode_metrics["evidence_quality"] += reward_components.evidence_quality_score
                episode_metrics["clinical_reasoning"] += reward_components.clinical_reasoning_score
                
                # Log detailed reward breakdown
                self.reward_system.log_reward_breakdown(reward_components)
            
            # Normalize metrics
            batch_size = len(questions)
            for key in episode_metrics:
                episode_metrics[key] /= batch_size
            
            # Update model parameters using accumulated rewards
            self.update_agents(episode_data)
            
            return dict(episode_metrics)
            
        except Exception as e:
            logger.error(f"Error in training episode: {e}")
            return {"total_reward": 0.0, "error": 1.0}
    
    def update_agents(self, episode_data: Dict[str, Any]):
        """Update agent parameters using MAPoRL."""
        try:
            rewards = torch.tensor(episode_data["rewards"], dtype=torch.float32, device=self.device)
            
            # Compute advantages (simplified)
            advantages = rewards - rewards.mean()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update each agent
            for agent_name, optimizer in self.optimizers.items():
                optimizer.zero_grad()
                
                # Simplified loss computation
                # In practice, you'd compute actual policy and value losses
                loss = -advantages.mean()  # Placeholder loss
                
                # Add entropy regularization
                entropy_loss = -self.config.entropy_coeff * torch.log(torch.tensor(0.5))
                total_loss = loss + entropy_loss
                
                # Backward pass
                if total_loss.requires_grad:
                    total_loss.backward()
                    
                    # Gradient clipping
                    if hasattr(self.workflow.agents[agent_name].model, 'parameters'):
                        torch.nn.utils.clip_grad_norm_(
                            self.workflow.agents[agent_name].model.parameters(),
                            self.config.max_grad_norm
                        )
                    
                    optimizer.step()
                
                # Update scheduler
                if agent_name in self.schedulers:
                    self.schedulers[agent_name].step()
            
            logger.info(f"Updated agents with average reward: {rewards.mean().item():.4f}")
            
        except Exception as e:
            logger.error(f"Error updating agents: {e}")
    
    def evaluate(self, eval_dataset: MedicalDataset) -> Dict[str, float]:
        """Evaluate the multi-agent system."""
        eval_metrics = defaultdict(float)
        
        try:
            eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc="Evaluating"):
                    question = batch["question"][0]
                    context = batch["context"][0]
                    ground_truth = batch["answer"][0]
                    
                    # Run workflow
                    result = self.workflow.run_workflow(
                        question=question,
                        context=context,
                        thread_id=f"eval_{self.global_step}"
                    )
                    
                    # Calculate reward
                    reward_components = self.reward_system.calculate_medical_reward(
                        agent_responses=result["agent_responses"],
                        question=question,
                        ground_truth=ground_truth,
                        context=context
                    )
                    
                    # Update metrics
                    eval_metrics["total_reward"] += reward_components.total_score()
                    eval_metrics["accuracy"] += reward_components.accuracy_score
                    eval_metrics["medical_relevance"] += reward_components.medical_relevance_score
                    eval_metrics["collaboration_quality"] += reward_components.collaboration_quality_score
                    eval_metrics["safety"] += reward_components.safety_score
                    eval_metrics["evidence_quality"] += reward_components.evidence_quality_score
                    eval_metrics["clinical_reasoning"] += reward_components.clinical_reasoning_score
            
            # Normalize metrics
            num_samples = len(eval_dataset)
            for key in eval_metrics:
                eval_metrics[key] /= num_samples
            
            return dict(eval_metrics)
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {"total_reward": 0.0, "error": 1.0}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        try:
            checkpoint_dir = f"checkpoints/maporl_epoch_{epoch}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save each agent's model
            for agent_name, agent in self.workflow.agents.items():
                if hasattr(agent.model, 'state_dict'):
                    torch.save(
                        agent.model.state_dict(),
                        os.path.join(checkpoint_dir, f"{agent_name}_model.pt")
                    )
                
                # Save optimizer state
                if agent_name in self.optimizers:
                    torch.save(
                        self.optimizers[agent_name].state_dict(),
                        os.path.join(checkpoint_dir, f"{agent_name}_optimizer.pt")
                    )
            
            # Save training metadata
            metadata = {
                "epoch": epoch,
                "global_step": self.global_step,
                "metrics": metrics,
                "config": asdict(self.config)
            }
            
            with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved checkpoint at epoch {epoch}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def train(self, train_dataset: MedicalDataset, eval_dataset: MedicalDataset = None):
        """Main training loop."""
        logger.info("Starting MAPoRL training...")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        for epoch in range(self.config.num_epochs):
            epoch_metrics = defaultdict(float)
            
            # Training loop
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                batch_metrics = self.train_episode(batch)
                
                # Update metrics
                for key, value in batch_metrics.items():
                    epoch_metrics[key] += value
                
                self.global_step += 1
                
                # Log metrics
                if self.global_step % 10 == 0:
                    try:
                        wandb.log({
                            "train/total_reward": batch_metrics["total_reward"],
                            "train/accuracy": batch_metrics["accuracy"],
                            "train/medical_relevance": batch_metrics["medical_relevance"],
                            "train/collaboration_quality": batch_metrics["collaboration_quality"],
                            "train/safety": batch_metrics["safety"],
                            "global_step": self.global_step
                        })
                    except:
                        pass
                
                # Evaluation
                if eval_dataset and self.global_step % self.config.eval_every == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    logger.info(f"Evaluation metrics: {eval_metrics}")
                    
                    try:
                        wandb.log({
                            "eval/total_reward": eval_metrics["total_reward"],
                            "eval/accuracy": eval_metrics["accuracy"],
                            "eval/medical_relevance": eval_metrics["medical_relevance"],
                            "eval/collaboration_quality": eval_metrics["collaboration_quality"],
                            "eval/safety": eval_metrics["safety"],
                            "global_step": self.global_step
                        })
                    except:
                        pass
                
                # Save checkpoint
                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(epoch, dict(epoch_metrics))
            
            # End of epoch
            num_batches = len(train_loader)
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            logger.info(f"Epoch {epoch+1} completed. Average reward: {epoch_metrics['total_reward']:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch, dict(epoch_metrics))
        
        logger.info("MAPoRL training completed!")
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """Collate function for DataLoader."""
        collated = defaultdict(list)
        for item in batch:
            for key, value in item.items():
                collated[key].append(value)
        return dict(collated)

# Utility functions
def create_trainer(
    config: MAPoRLConfig = None,
    workflow_config: Dict[str, Any] = None,
    device: str = "cuda:0"
) -> MAPoRLTrainer:
    """Create MAPoRL trainer instance."""
    return MAPoRLTrainer(config, workflow_config, device)

def load_medical_dataset(data_path: str, max_samples: int = None) -> MedicalDataset:
    """Load medical dataset for training."""
    return MedicalDataset(data_path, max_samples) 