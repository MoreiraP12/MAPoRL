"""
Model configuration for multi-agent medical pipeline.
Optimized for 4 A10G GPUs with small but effective models.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for individual agent models."""
    model_name: str
    max_length: int
    temperature: float
    top_p: float
    device_map: str
    load_in_8bit: bool
    trust_remote_code: bool

@dataclass
class MultiAgentConfig:
    """Configuration for the multi-agent system."""
    agents: Dict[str, ModelConfig]
    max_rounds: int
    memory_length: int
    reward_weights: Dict[str, float]

# Small models optimized for medical tasks and GPU constraints
MEDICAL_AGENT_CONFIGS = {
    "planner": ModelConfig(
        model_name="microsoft/DialoGPT-small",  # 117M parameters
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        device_map="cuda:0",
        load_in_8bit=True,
        trust_remote_code=True
    ),
    "researcher": ModelConfig(
        model_name="distilbert-base-uncased",  # 66M parameters, good for retrieval
        max_length=512,
        temperature=0.5,
        top_p=0.8,
        device_map="cuda:1",
        load_in_8bit=True,
        trust_remote_code=True
    ),
    "analyst": ModelConfig(
        model_name="microsoft/DialoGPT-small",  # 117M parameters
        max_length=512,
        temperature=0.6,
        top_p=0.9,
        device_map="cuda:2",
        load_in_8bit=True,
        trust_remote_code=True
    ),
    "reporter": ModelConfig(
        model_name="microsoft/DialoGPT-small",  # 117M parameters
        max_length=512,
        temperature=0.4,
        top_p=0.8,
        device_map="cuda:3",
        load_in_8bit=True,
        trust_remote_code=True
    )
}

# Alternative: Use tiny medical-focused models if available
ALTERNATIVE_CONFIGS = {
    "planner": ModelConfig(
        model_name="emilyalsentzer/Bio_ClinicalBERT",  # Medical domain
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        device_map="cuda:0",
        load_in_8bit=True,
        trust_remote_code=True
    ),
    "researcher": ModelConfig(
        model_name="dmis-lab/biobert-base-cased-v1.1",  # Medical BERT
        max_length=512,
        temperature=0.5,
        top_p=0.8,
        device_map="cuda:1",
        load_in_8bit=True,
        trust_remote_code=True
    ),
    "analyst": ModelConfig(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=512,
        temperature=0.6,
        top_p=0.9,
        device_map="cuda:2",
        load_in_8bit=True,
        trust_remote_code=True
    ),
    "reporter": ModelConfig(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        max_length=512,
        temperature=0.4,
        top_p=0.8,
        device_map="cuda:3",
        load_in_8bit=True,
        trust_remote_code=True
    )
}

MULTI_AGENT_CONFIG = MultiAgentConfig(
    agents=MEDICAL_AGENT_CONFIGS,
    max_rounds=5,
    memory_length=2048,
    reward_weights={
        "accuracy": 0.4,
        "medical_relevance": 0.3,
        "collaboration_quality": 0.2,
        "safety": 0.1
    }
)

# GPU memory optimization settings
OPTIMIZATION_SETTINGS = {
    "gradient_checkpointing": True,
    "fp16": True,
    "dataloader_pin_memory": True,
    "dataloader_num_workers": 4,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 4,
    "gradient_accumulation_steps": 8,
} 