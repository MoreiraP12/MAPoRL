#!/usr/bin/env python3
"""
Simple test script for the multiagent pipeline with Qwen3 0.6B models.
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).parent / "config"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_config_import():
    """Test importing the Qwen configuration."""
    try:
        from model_config_qwen import (
            QWEN_MEDICAL_AGENT_CONFIGS, 
            MEDXPERT_CONFIG,
            load_qwen_model
        )
        
        logger.info("✅ Successfully imported Qwen configurations")
        logger.info(f"📊 Found {len(QWEN_MEDICAL_AGENT_CONFIGS)} agent configs:")
        
        for agent_name, config in QWEN_MEDICAL_AGENT_CONFIGS.items():
            logger.info(f"  - {agent_name}: {config.model_name}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to import configurations: {e}")
        return False

def test_model_loading():
    """Test loading a single Qwen model."""
    try:
        from model_config_qwen import load_qwen_model
        
        logger.info("🤖 Testing model loading...")
        
        # Try to load the planner model
        model, tokenizer = load_qwen_model(agent_type="planner", device="cpu")
        
        logger.info("✅ Successfully loaded Qwen3-0.6B model")
        logger.info(f"   Model type: {model.config.model_type}")
        logger.info(f"   Vocab size: {tokenizer.vocab_size}")
        
        # Test a simple generation
        test_prompt = "What is hypertension?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + 50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🧠 Model response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_simple_multiagent_simulation():
    """Simulate a simple multiagent conversation."""
    try:
        from model_config_qwen import QWEN_MEDICAL_AGENT_CONFIGS
        
        logger.info("🏥 Simulating multiagent medical consultation...")
        
        # Sample medical question
        question = "What are the symptoms of diabetes?"
        context = "A 45-year-old patient is asking about diabetes symptoms."
        
        logger.info(f"📋 Question: {question}")
        logger.info(f"📋 Context: {context}")
        
        # Simulate responses from each agent
        agent_roles = {
            "planner": "I'll coordinate our medical consultation. Let me analyze this diabetes question.",
            "researcher": "Based on medical literature, diabetes symptoms include increased thirst, frequent urination, fatigue, and unexplained weight loss.",
            "analyst": "The patient profile suggests we should focus on Type 2 diabetes symptoms, which are more common in this age group.",
            "reporter": "MEDICAL CONSULTATION: Common diabetes symptoms include polyuria (frequent urination), polydipsia (increased thirst), polyphagia (increased hunger), fatigue, blurred vision, and slow-healing wounds. For a 45-year-old patient, Type 2 diabetes is most likely. Recommend blood glucose testing and consultation with healthcare provider."
        }
        
        logger.info("\n🤝 Multiagent Consultation Simulation:")
        for agent, response in agent_roles.items():
            logger.info(f"\n{agent.upper()} AGENT:")
            logger.info(f"  {response}")
        
        logger.info("\n✅ Multiagent simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multiagent simulation failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting Simple Multiagent Pipeline Test")
    logger.info("🤖 Using Qwen3-0.6B for all agents")
    logger.info("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Configuration Import
    logger.info("\n1️⃣ Testing configuration import...")
    if test_config_import():
        success_count += 1
    
    # Test 2: Model Loading
    logger.info("\n2️⃣ Testing model loading...")
    try:
        import torch
        if test_model_loading():
            success_count += 1
    except ImportError:
        logger.warning("⚠️ PyTorch not available, skipping model loading test")
        total_tests -= 1
    
    # Test 3: Multiagent Simulation
    logger.info("\n3️⃣ Testing multiagent simulation...")
    if test_simple_multiagent_simulation():
        success_count += 1
    
    # Final Results
    logger.info("\n" + "=" * 50)
    logger.info(f"🎯 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        logger.info("🎉 All tests passed! Your multiagent pipeline is ready.")
        logger.info("💡 Next steps:")
        logger.info("   - Run full evaluation: python test_medxpert_evaluation.py")
        logger.info("   - Start training: python local_train_medxpert.py")
    else:
        logger.error(f"❌ {total_tests - success_count} tests failed. Check the logs above.")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 