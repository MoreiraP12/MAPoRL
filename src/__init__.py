"""
Medical Multi-Agent Pipeline with MAPoRL Training.
"""

__version__ = "0.1.0"
__author__ = "MAPoRL Medical Team"
__description__ = "Multi-Agent Post-co-training for Collaborative Medical LLMs with Reinforcement Learning"

from .workflow.medical_workflow import create_medical_workflow, MedicalWorkflow
from .reward.medical_reward_system import create_medical_reward_system, MedicalRewardSystem
from .training.mapoRRL_trainer import create_trainer, MAPoRLTrainer, MAPoRLConfig

__all__ = [
    "create_medical_workflow",
    "MedicalWorkflow", 
    "create_medical_reward_system",
    "MedicalRewardSystem",
    "create_trainer",
    "MAPoRLTrainer",
    "MAPoRLConfig"
] 