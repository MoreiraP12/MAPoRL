#!/usr/bin/env python3
"""
Basic test script for the multiagent pipeline with Qwen3 0.6B models (no quantization).
"""

import os
import sys
import logging
from pathlib import Path

# Add config to path
sys.path.append(str(Path(__file__).parent / "config"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_config():
    """Test the basic configuration setup."""
    try:
        from model_config_qwen import (
            QWEN_MEDICAL_AGENT_CONFIGS, 
            MEDXPERT_CONFIG
        )
        
        logger.info("✅ Configuration loaded successfully")
        logger.info(f"📊 Agents configured: {list(QWEN_MEDICAL_AGENT_CONFIGS.keys())}")
        logger.info(f"🎯 Max rounds: {MEDXPERT_CONFIG.max_rounds}")
        logger.info(f"🧠 Memory length: {MEDXPERT_CONFIG.memory_length}")
        
        # Check reward weights
        logger.info("🎪 Reward weights:")
        for metric, weight in MEDXPERT_CONFIG.reward_weights.items():
            logger.info(f"  - {metric}: {weight}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False

def test_basic_model_config():
    """Test basic model configuration without loading."""
    try:
        from model_config_qwen import get_qwen_config, AGENT_CONFIGS
        
        logger.info("🤖 Testing model configurations...")
        
        qwen_config = get_qwen_config()
        logger.info(f"✅ Base Qwen config loaded")
        logger.info(f"   Torch dtype: {qwen_config.get('torch_dtype', 'default')}")
        
        logger.info("🎭 Agent-specific configurations:")
        for agent_type, config in AGENT_CONFIGS.items():
            logger.info(f"   {agent_type}:")
            logger.info(f"     - Model: {config['model_name']}")
            logger.info(f"     - Device: {config['device']}")
            logger.info(f"     - Max tokens: {config['max_tokens']}")
            logger.info(f"     - Temperature: {config['temperature']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model config test failed: {e}")
        return False

def test_multiagent_workflow_simulation():
    """Simulate the multiagent workflow logic."""
    try:
        logger.info("🏥 Testing multiagent workflow simulation...")
        
        # Simulate a medical consultation workflow
        sample_case = {
            "question": "A 60-year-old patient with chest pain. What should be the immediate evaluation?",
            "context": "Patient presents to ED with 2-hour history of substernal chest pain, diaphoretic, with history of hypertension and diabetes."
        }
        
        logger.info(f"📋 Test Case: {sample_case['question']}")
        logger.info(f"📋 Context: {sample_case['context']}")
        
        # Simulate workflow steps
        workflow_steps = [
            {
                "agent": "planner",
                "step": "Initial Assessment",
                "action": "Analyze symptoms and prioritize immediate actions - chest pain protocol"
            },
            {
                "agent": "researcher", 
                "step": "Evidence Gathering",
                "action": "Review guidelines for acute chest pain evaluation - STEMI/NSTEMI protocols"
            },
            {
                "agent": "analyst",
                "step": "Clinical Analysis", 
                "action": "Risk stratification using HEART score, consider differential diagnoses"
            },
            {
                "agent": "reporter",
                "step": "Final Recommendation",
                "action": "Immediate 12-lead ECG, IV access, cardiac monitors, troponin levels, aspirin if no contraindications"
            }
        ]
        
        logger.info("\n🔄 Workflow Simulation:")
        for i, step in enumerate(workflow_steps, 1):
            logger.info(f"\n  Step {i} - {step['agent'].upper()} ({step['step']}):")
            logger.info(f"    Action: {step['action']}")
        
        # Simulate reward calculation
        logger.info("\n📊 Simulated Performance Metrics:")
        metrics = {
            "accuracy": 0.85,
            "medical_relevance": 0.92,
            "collaboration_quality": 0.78,
            "safety": 0.95,
            "evidence_quality": 0.83,
            "clinical_reasoning": 0.88
        }
        
        for metric, score in metrics.items():
            logger.info(f"  - {metric}: {score:.2f}")
        
        overall_score = sum(metrics.values()) / len(metrics)
        logger.info(f"  🎯 Overall Score: {overall_score:.2f}")
        
        logger.info("\n✅ Workflow simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Workflow simulation failed: {e}")
        return False

def test_dependencies():
    """Test required dependencies."""
    missing_deps = []
    
    # Check core dependencies
    deps = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("numpy", "NumPy")
    ]
    
    for module, name in deps:
        try:
            __import__(module)
            logger.info(f"✅ {name} available")
        except ImportError:
            logger.warning(f"⚠️ {name} not available")
            missing_deps.append(name)
    
    if missing_deps:
        logger.warning(f"Missing dependencies: {', '.join(missing_deps)}")
        logger.info("💡 Install with: pip install torch transformers datasets numpy")
    
    return len(missing_deps) == 0

def main():
    """Main test function."""
    logger.info("🚀 Basic Multiagent Pipeline Test")
    logger.info("🤖 Target: Qwen3-0.6B for all agents")
    logger.info("📝 Mode: Configuration and workflow testing")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration Test", test_basic_config),
        ("Model Config Test", test_basic_model_config), 
        ("Workflow Simulation", test_multiagent_workflow_simulation),
        ("Dependencies Check", test_dependencies)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"💥 {test_name} CRASHED: {e}")
    
    # Final Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"🎯 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✨ Your multiagent pipeline configuration is working!")
        logger.info("\n💡 Next Steps:")
        logger.info("   1. Install missing dependencies if any")
        logger.info("   2. Test with actual models: python local_train_medxpert.py --test-only")
        logger.info("   3. Run full evaluation: python test_medxpert_evaluation.py")
        logger.info("   4. Start training: python local_train_medxpert.py")
    elif passed > 0:
        logger.warning(f"⚠️ Partial success: {passed}/{total} tests passed")
        logger.info("💡 Fix the failing tests above to proceed")
    else:
        logger.error("❌ All tests failed - check your setup")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 