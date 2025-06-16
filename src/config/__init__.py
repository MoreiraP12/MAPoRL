"""
Configuration Package.
"""

from .model_config import (
    ModelConfig,
    MultiAgentConfig,
    MEDICAL_AGENT_CONFIGS,
    ALTERNATIVE_CONFIGS,
    QWEN_AGENT_CONFIGS,
    MULTI_AGENT_CONFIG,
    ALT_MULTI_AGENT_CONFIG,
    QWEN_MULTI_AGENT_CONFIG,
    OPTIMIZATION_SETTINGS
)

__all__ = [
    "ModelConfig",
    "MultiAgentConfig", 
    "MEDICAL_AGENT_CONFIGS",
    "ALTERNATIVE_CONFIGS",
    "QWEN_AGENT_CONFIGS",
    "MULTI_AGENT_CONFIG",
    "ALT_MULTI_AGENT_CONFIG",
    "QWEN_MULTI_AGENT_CONFIG",
    "OPTIMIZATION_SETTINGS"
] 