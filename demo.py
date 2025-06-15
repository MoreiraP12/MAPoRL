"""
Demo script for Medical Multi-Agent Pipeline with MAPoRL.
"""

import os
import sys
import logging
import json
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.workflow.medical_workflow import create_medical_workflow
from src.reward.medical_reward_system import create_medical_reward_system
from src.training.maporl_trainer import create_trainer, load_medical_dataset, MAPoRLConfig
from src.config.model_config import MULTI_AGENT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_workflow():
    """Demonstrate the medical workflow with sample questions."""
    logger.info("=== Medical Multi-Agent Workflow Demo ===")
    
    # Create workflow
    workflow = create_medical_workflow()
    
    # Sample medical questions
    sample_questions = [
        {
            "question": "What are the first-line treatments for hypertension in adults?",
            "context": "A 50-year-old patient presents with consistently elevated blood pressure readings above 140/90 mmHg."
        },
        {
            "question": "How should type 2 diabetes be managed in elderly patients?",
            "context": "An 75-year-old patient with newly diagnosed type 2 diabetes and multiple comorbidities."
        },
        {
            "question": "What are the diagnostic criteria for pneumonia?",
            "context": "A patient presents with fever, cough, and shortness of breath."
        }
    ]
    
    # Process each question
    for i, sample in enumerate(sample_questions):
        logger.info(f"\n--- Processing Question {i+1} ---")
        logger.info(f"Question: {sample['question']}")
        logger.info(f"Context: {sample['context']}")
        
        try:
            # Run workflow
            result = workflow.run_workflow(
                question=sample["question"],
                context=sample["context"],
                thread_id=f"demo_{i}"
            )
            
            # Display results
            logger.info(f"Workflow Status: {result['workflow_status']}")
            logger.info(f"Rounds Completed: {result['rounds_completed']}")
            logger.info(f"Confidence Scores: {result['confidence_scores']}")
            logger.info(f"Medical Entities: {result['medical_entities']}")
            logger.info(f"Safety Flags: {result['safety_flags']}")
            
            # Display agent responses
            logger.info("\n--- Agent Responses ---")
            for agent_name, responses in result["agent_responses"].items():
                logger.info(f"\n{agent_name.upper()}:")
                for j, response in enumerate(responses):
                    logger.info(f"  Response {j+1}: {response[:200]}...")
            
            # Display final answer
            logger.info(f"\n--- Final Answer ---")
            logger.info(f"{result['final_answer'][:500]}...")
            
        except Exception as e:
            logger.error(f"Error processing question {i+1}: {e}")
    
    logger.info("\n=== Workflow Demo Completed ===")

def demo_reward_system():
    """Demonstrate the medical reward system."""
    logger.info("\n=== Medical Reward System Demo ===")
    
    # Create reward system
    reward_system = create_medical_reward_system()
    
    # Sample agent responses
    sample_responses = {
        "planner": [
            "MEDICAL INVESTIGATION PLAN: This is a diagnosis type question about hypertension. Key concepts include blood pressure, cardiovascular risk factors. Recommended approach: 1. Analyze symptoms and patient history 2. Consider differential diagnoses 3. Identify required diagnostic tests 4. Assess risk factors"
        ],
        "researcher": [
            "MEDICAL RESEARCH FINDINGS: Evidence-based information on hypertension management. Clinical Guidelines: JNC 8, AHA/ACC 2017. First-line treatment includes ACE inhibitors, ARBs, thiazide diuretics, CCBs. Evidence Level: Level 1 systematic reviews support these recommendations."
        ],
        "analyst": [
            "CLINICAL ANALYSIS: Differential analysis shows primary hypertension likely based on age and risk factors. No red flags identified. Clinical confidence: high for evidence-based treatment approach. Appropriate monitoring includes regular BP checks, kidney function assessment."
        ],
        "reporter": [
            "MEDICAL CONSULTATION REPORT: Based on collaborative analysis, first-line treatments for hypertension include ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers. Treatment should be individualized based on patient factors. Regular monitoring recommended."
        ]
    }
    
    question = "What are the first-line treatments for hypertension in adults?"
    ground_truth = "First-line treatments for hypertension include ACE inhibitors, ARBs, thiazide diuretics, and calcium channel blockers."
    
    # Calculate reward
    reward_components = reward_system.calculate_medical_reward(
        agent_responses=sample_responses,
        question=question,
        ground_truth=ground_truth
    )
    
    # Display reward breakdown
    reward_system.log_reward_breakdown(reward_components)
    
    # Calculate total score
    total_score = reward_components.total_score()
    logger.info(f"Total Reward Score: {total_score:.4f}")
    
    logger.info("=== Reward System Demo Completed ===")

def demo_training_setup():
    """Demonstrate the training setup (without actual training)."""
    logger.info("\n=== MAPoRL Training Setup Demo ===")
    
    try:
        # Create sample training data
        sample_data = [
            {
                "question": "What are the symptoms of diabetes?",
                "context": "A patient presents with increased thirst and frequent urination.",
                "answer": "Common symptoms include polyuria, polydipsia, polyphagia, and weight loss."
            },
            {
                "question": "How is pneumonia treated?",
                "context": "A patient diagnosed with community-acquired pneumonia.",
                "answer": "Treatment typically involves appropriate antibiotics based on severity and risk factors."
            }
        ]
        
        # Save sample data
        os.makedirs("data", exist_ok=True)
        with open("data/sample_medical_data.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        
        # Create dataset
        dataset = load_medical_dataset("data/sample_medical_data.json")
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Create trainer config
        config = MAPoRLConfig(
            learning_rate=1e-5,
            batch_size=1,  # Small batch for demo
            num_epochs=1,
            max_rounds_per_episode=2
        )
        
        # Create trainer
        trainer = create_trainer(config=config)
        logger.info("Created MAPoRL trainer")
        
        # Display configuration
        logger.info(f"Training Configuration:")
        logger.info(f"  Learning Rate: {config.learning_rate}")
        logger.info(f"  Batch Size: {config.batch_size}")
        logger.info(f"  Number of Epochs: {config.num_epochs}")
        logger.info(f"  Max Rounds per Episode: {config.max_rounds_per_episode}")
        
        logger.info("=== Training Setup Demo Completed ===")
        
    except Exception as e:
        logger.error(f"Error in training setup demo: {e}")

def main():
    """Main demo function."""
    logger.info("Starting Medical Multi-Agent Pipeline Demo")
    
    try:
        # Demo 1: Workflow
        demo_workflow()
        
        # Demo 2: Reward System
        demo_reward_system()
        
        # Demo 3: Training Setup
        demo_training_setup()
        
        logger.info("\n=== All Demos Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main() 