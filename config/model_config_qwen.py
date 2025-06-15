"""
Model configuration for Qwen3 0.6B models on 4x A10G GPUs (SageMaker).
Optimized for MedXpert benchmark training.
"""

import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

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
    torch_dtype: str = "float16"

@dataclass
class MultiAgentConfig:
    """Configuration for the multi-agent system."""
    agents: Dict[str, ModelConfig]
    max_rounds: int
    memory_length: int
    reward_weights: Dict[str, float]

# Qwen3 0.6B models optimized for medical tasks and GPU constraints
QWEN_MEDICAL_AGENT_CONFIGS = {
    "planner": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        max_length=1024,
        temperature=0.7,
        top_p=0.9,
        device_map="cuda:0",
        load_in_8bit=True,
        trust_remote_code=True,
        torch_dtype="float16"
    ),
    "researcher": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        max_length=1024,
        temperature=0.5,
        top_p=0.8,
        device_map="cuda:1", 
        load_in_8bit=True,
        trust_remote_code=True,
        torch_dtype="float16"
    ),
    "analyst": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        max_length=1024,
        temperature=0.6,
        top_p=0.9,
        device_map="cuda:2",
        load_in_8bit=True,
        trust_remote_code=True,
        torch_dtype="float16"
    ),
    "reporter": ModelConfig(
        model_name="Qwen/Qwen3-0.6B",
        max_length=1024,
        temperature=0.4,
        top_p=0.8,
        device_map="cuda:3",
        load_in_8bit=True,
        trust_remote_code=True,
        torch_dtype="float16"
    )
}

# MedXpert-specific configuration
MEDXPERT_CONFIG = MultiAgentConfig(
    agents=QWEN_MEDICAL_AGENT_CONFIGS,
    max_rounds=4,  # Increased for better collaboration
    memory_length=2048,
    reward_weights={
        "accuracy": 0.35,  # Higher weight for accuracy on MedXpert
        "medical_relevance": 0.25,  # Critical for medical benchmarks
        "collaboration_quality": 0.15,
        "safety": 0.15,  # Important for medical applications
        "evidence_quality": 0.05,
        "clinical_reasoning": 0.05
    }
)

# SageMaker optimization settings for 4x A10G GPUs
SAGEMAKER_OPTIMIZATION_SETTINGS = {
    "gradient_checkpointing": True,
    "fp16": True,
    "bf16": False,  # A10G works better with fp16
    "dataloader_pin_memory": True,
    "dataloader_num_workers": 8,  # 2 per GPU
    "per_device_train_batch_size": 4,  # Increased for 0.6B models
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,
    "lr_scheduler_type": "cosine",
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
}

# MedXpert-specific training parameters
MEDXPERT_TRAINING_CONFIG = {
    "learning_rate": 3e-5,  # Higher LR for smaller models
    "batch_size": 8,  # Total batch size across GPUs
    "num_epochs": 15,  # More epochs for MedXpert
    "max_rounds_per_episode": 4,
    "gamma": 0.95,  # Slightly lower discount for medical reasoning
    "clip_ratio": 0.2,
    "value_loss_coeff": 0.5,
    "entropy_coeff": 0.02,  # Higher entropy for exploration
    "save_every": 50,
    "eval_every": 25,
    "warmup_steps": 200,
    
    # Medical-specific parameters
    "safety_penalty_weight": 3.0,  # Higher safety penalty
    "collaboration_bonus_weight": 2.0,
    "medical_relevance_weight": 1.5,
    "medxpert_accuracy_weight": 2.0,  # Bonus for MedXpert accuracy
}

# Environment variables for SageMaker
SAGEMAKER_ENV_CONFIG = {
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "TOKENIZERS_PARALLELISM": "false",
    "TRANSFORMERS_CACHE": "/opt/ml/model/transformers_cache",
    "HF_HOME": "/opt/ml/model/huggingface_cache",
    "WANDB_DIR": "/opt/ml/output/data",
    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
}

# Data paths for SageMaker
SAGEMAKER_DATA_CONFIG = {
    "train_data_path": "/opt/ml/input/data/train",
    "eval_data_path": "/opt/ml/input/data/eval", 
    "output_path": "/opt/ml/output/data",
    "model_path": "/opt/ml/model",
    "checkpoints_path": "/opt/ml/checkpoints",
}

# MedXpert dataset configuration
MEDXPERT_DATA_CONFIG = {
    "dataset_name": "medxpert",
    "train_file": "medxpert_train.jsonl",
    "eval_file": "medxpert_eval.jsonl",
    "max_train_samples": 5000,  # Subset for demo
    "max_eval_samples": 500,
    "data_format": "jsonl",
    "question_field": "question",
    "context_field": "context", 
    "answer_field": "answer",
    "id_field": "id"
}

def get_qwen_config():
    """
    Optimized Qwen3-0.6B configuration for SageMaker A10G GPUs
    """
    return {
        # Model Configuration
        "model_name": "Qwen/Qwen3-0.6B",  # Using official Qwen3 0.6B model
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
        
        # Quantization for A10G memory constraints
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
        ),
        
        # Generation parameters optimized for medical reasoning
        "generation_config": {
            "max_new_tokens": 512,
            "temperature": 0.6,  # Balanced for medical accuracy
            "top_p": 0.95,
            "top_k": 20,
            "do_sample": True,
            "pad_token_id": None,  # Will be set dynamically
            "eos_token_id": None,  # Will be set dynamically
        },
        
        # Memory optimization
        "gradient_checkpointing": True,
        "use_cache": False,  # Saves memory during training
    }

# Agent-specific configurations
AGENT_CONFIGS = {
    "planner": {
        "model_name": "Qwen/Qwen3-0.6B",
        "device": "cuda:0",
        "max_tokens": 256,
        "temperature": 0.7,
    },
    "researcher": {
        "model_name": "Qwen/Qwen3-0.6B",
        "device": "cuda:1", 
        "max_tokens": 512,
        "temperature": 0.6,
    },
    "analyst": {
        "model_name": "Qwen/Qwen3-0.6B",
        "device": "cuda:2",
        "max_tokens": 512,
        "temperature": 0.5,
    },
    "reporter": {
        "model_name": "Qwen/Qwen3-0.6B",
        "device": "cuda:3",
        "max_tokens": 1024,
        "temperature": 0.4,
    }
}

def load_qwen_model(agent_type="base", device="cuda:0"):
    """
    Load Qwen3-0.6B model with optimizations for specific agent
    """
    config = get_qwen_config()
    agent_config = AGENT_CONFIGS.get(agent_type, AGENT_CONFIGS["planner"])
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            agent_config["model_name"],
            trust_remote_code=True,
            padding_side="left"  # Important for batch inference
        )
        
        # Set special tokens if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            agent_config["model_name"],
            torch_dtype=config["torch_dtype"],
            device_map={"": device},
            quantization_config=config["quantization_config"],
            trust_remote_code=True,
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            
        logger.info(f"✅ Loaded {agent_type} agent with Qwen3-0.6B on {device}")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to load Qwen3-0.6B for {agent_type}: {e}")
        raise 