#!/usr/bin/env python3
"""
MedXpert Benchmark Evaluation for Multi-Agent Medical System
Tests the collaborative performance of medical agents on real medical questions.
"""

import asyncio
import json
import logging
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import our modules
from src.workflow.medical_workflow import MedicalWorkflow
from src.agents.base_agent import MedicalState
from src.reward.medical_reward_system import MedicalRewardSystem
from config.model_config import MEDICAL_AGENT_CONFIGS, MULTI_AGENT_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MedXpertSample:
    """Single MedXpert benchmark sample."""
    id: str
    question: str
    context: str
    answer: str
    category: str = "general"
    difficulty: str = "medium"
    medical_specialty: str = "general_medicine"

@dataclass
class EvaluationResult:
    """Evaluation result for a single sample."""
    sample_id: str
    question: str
    ground_truth: str
    agent_response: str
    multi_agent_response: str
    collaboration_rounds: int
    processing_time: float
    accuracy_score: float
    medical_relevance_score: float
    safety_score: float
    collaboration_quality: float
    overall_score: float
    agent_metadata: Dict[str, Any]
    biomcp_used: bool

class MedXpertEvaluator:
    """Evaluator for MedXpert benchmark using multi-agent system."""
    
    def __init__(self, config_path: str = None):
        """Initialize the evaluator."""
        self.workflow = None
        self.reward_system = None
        self.results = []
        self.config_path = config_path
        
        # Initialize components
        self._setup_workflow()
        self._setup_reward_system()
        
        logger.info("🏥 MedXpert Multi-Agent Evaluator initialized")
    
    def _setup_workflow(self):
        """Setup the medical workflow."""
        try:
            self.workflow = MedicalWorkflow(MULTI_AGENT_CONFIG)
            logger.info("✅ Medical workflow initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize workflow: {e}")
            raise
    
    def _setup_reward_system(self):
        """Setup the reward system for evaluation."""
        try:
            self.reward_system = MedicalRewardSystem()
            logger.info("✅ Medical reward system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize reward system: {e}")
            raise
    
    def create_sample_medxpert_data(self) -> List[MedXpertSample]:
        """Create sample MedXpert data for testing."""
        samples = [
            MedXpertSample(
                id="medx_001",
                question="A 65-year-old patient presents with chest pain and shortness of breath. The chest pain is substernal, radiates to the left arm, and started 2 hours ago. What is the most appropriate initial diagnostic test?",
                context="Patient has a history of hypertension and diabetes mellitus. Vital signs: BP 160/90 mmHg, HR 95 bpm, RR 22/min, O2 sat 92% on room air. Patient appears diaphoretic and anxious.",
                answer="The most appropriate initial diagnostic test is a 12-lead electrocardiogram (ECG) to evaluate for acute coronary syndrome, specifically ST-elevation myocardial infarction (STEMI) or non-ST-elevation myocardial infarction (NSTEMI). This should be obtained within 10 minutes of presentation. Additional immediate tests should include chest X-ray and cardiac biomarkers (troponin).",
                category="cardiology",
                difficulty="medium",
                medical_specialty="emergency_medicine"
            ),
            MedXpertSample(
                id="medx_002",
                question="What are the first-line pharmacological treatments for newly diagnosed type 2 diabetes mellitus in adults without contraindications?",
                context="A 55-year-old obese patient (BMI 32) with newly diagnosed type 2 diabetes. HbA1c is 8.5%. No history of cardiovascular disease, kidney disease, or other contraindications to standard medications. Patient is motivated for lifestyle changes.",
                answer="The first-line pharmacological treatment for type 2 diabetes is metformin, typically starting at 500-850 mg twice daily with meals, titrated based on tolerance and glycemic response. This should be combined with intensive lifestyle modifications including dietary changes and regular physical activity. Target HbA1c should be individualized but generally <7% for most adults.",
                category="endocrinology",
                difficulty="easy",
                medical_specialty="primary_care"
            ),
            MedXpertSample(
                id="medx_003",
                question="How should community-acquired pneumonia be managed in a previously healthy 35-year-old adult presenting to the emergency department?",
                context="Patient presents with 3-day history of fever, productive cough with purulent sputum, and pleuritic chest pain. Vital signs: temperature 38.8°C, HR 110 bpm, RR 24/min, BP 120/80 mmHg, O2 sat 94% on room air. Chest X-ray shows right lower lobe consolidation.",
                answer="For a previously healthy adult with community-acquired pneumonia, empirical antibiotic therapy should be initiated. Recommended first-line treatment includes amoxicillin 1g three times daily or doxycycline 100mg twice daily for outpatient management. However, given the patient's presentation in the ED with tachycardia and hypoxemia, consider hospitalization criteria using CURB-65 or PSI scores. If hospitalized, use combination therapy with beta-lactam plus macrolide or respiratory fluoroquinolone.",
                category="pulmonology",
                difficulty="medium",
                medical_specialty="emergency_medicine"
            ),
            MedXpertSample(
                id="medx_004",
                question="A patient with a BRCA1 mutation asks about cancer screening recommendations. What should be advised?",
                context="A 30-year-old woman with a confirmed pathogenic BRCA1 mutation. Family history includes mother with breast cancer at age 45 and maternal grandmother with ovarian cancer at age 60. Patient is currently healthy with no symptoms.",
                answer="For a BRCA1 mutation carrier, enhanced screening is recommended: 1) Breast surveillance: Annual breast MRI starting at age 25-30, clinical breast exam every 6 months; 2) Consider risk-reducing bilateral mastectomy; 3) Ovarian cancer screening: Risk-reducing bilateral salpingo-oophorectomy between ages 35-40 or after childbearing is complete; 4) Genetic counseling for family planning; 5) Consider chemoprevention options. Screening should be coordinated with a high-risk clinic or genetic counselor.",
                category="oncology",
                difficulty="hard",
                medical_specialty="genetic_counseling"
            ),
            MedXpertSample(
                id="medx_005",
                question="What is the appropriate management for a patient presenting with acute severe hypertension (BP 220/120 mmHg) and signs of end-organ damage?",
                context="A 50-year-old patient presents with severe headache, altered mental status, and blood pressure of 220/120 mmHg. Fundoscopic exam shows papilledema and flame-shaped hemorrhages. Patient has a history of poorly controlled hypertension but no recent medication changes.",
                answer="This represents hypertensive emergency requiring immediate but controlled blood pressure reduction. Goals: 1) Reduce BP by 10-20% in the first hour, then gradually to <160/100 mmHg over the next 6 hours; 2) Use IV antihypertensives such as nicardipine (preferred) or clevidipine; 3) Avoid rapid or excessive BP reduction to prevent cerebral, coronary, or renal ischemia; 4) Continuous cardiac monitoring and frequent neurologic assessments; 5) Investigate and treat underlying causes; 6) ICU admission for close monitoring.",
                category="cardiology",
                difficulty="hard",
                medical_specialty="emergency_medicine"
            )
        ]
        
        logger.info(f"📊 Created {len(samples)} MedXpert sample questions")
        return samples
    
    async def evaluate_single_sample(self, sample: MedXpertSample) -> EvaluationResult:
        """Evaluate a single MedXpert sample using the multi-agent system."""
        logger.info(f"🔍 Evaluating sample: {sample.id}")
        start_time = time.time()
        
        try:
            # Create medical state
            initial_state = MedicalState(
                question=sample.question,
                context=sample.context,
                agent_responses={},
                medical_entities=[],
                safety_flags=[],
                confidence_scores={},
                workflow_rounds=0
            )
            
            # Process with multi-agent workflow
            final_state = await self.workflow.process_medical_question(initial_state)
            processing_time = time.time() - start_time
            
            # Extract final response (from reporter agent)
            multi_agent_response = ""
            if "reporter" in final_state.agent_responses:
                multi_agent_response = final_state.agent_responses["reporter"][-1]
            else:
                # Fallback: combine all responses
                all_responses = []
                for agent_responses in final_state.agent_responses.values():
                    all_responses.extend(agent_responses)
                multi_agent_response = "\n\n".join(all_responses)
            
            # Get single agent response for comparison (just use the first agent)
            single_agent_response = ""
            if final_state.agent_responses:
                first_agent_responses = list(final_state.agent_responses.values())[0]
                if first_agent_responses:
                    single_agent_response = first_agent_responses[0]
            
            # Evaluate response quality
            evaluation_metrics = self.reward_system.calculate_rewards(
                final_state, sample.answer
            )
            
            # Check if BioMCP was used
            biomcp_used = any(
                metadata.get("biomcp_available", False) 
                for metadata in getattr(final_state, 'agent_metadata', {}).values()
            )
            
            # Create evaluation result
            result = EvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                ground_truth=sample.answer,
                agent_response=single_agent_response,
                multi_agent_response=multi_agent_response,
                collaboration_rounds=final_state.workflow_rounds,
                processing_time=processing_time,
                accuracy_score=evaluation_metrics.get("accuracy", 0.0),
                medical_relevance_score=evaluation_metrics.get("medical_relevance", 0.0),
                safety_score=evaluation_metrics.get("safety", 0.0),
                collaboration_quality=evaluation_metrics.get("collaboration_quality", 0.0),
                overall_score=evaluation_metrics.get("overall_score", 0.0),
                agent_metadata=getattr(final_state, 'agent_metadata', {}),
                biomcp_used=biomcp_used
            )
            
            logger.info(f"✅ Sample {sample.id} completed - Score: {result.overall_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error evaluating sample {sample.id}: {e}")
            
            # Create error result
            return EvaluationResult(
                sample_id=sample.id,
                question=sample.question,
                ground_truth=sample.answer,
                agent_response=f"Error: {str(e)}",
                multi_agent_response=f"Error: {str(e)}",
                collaboration_rounds=0,
                processing_time=time.time() - start_time,
                accuracy_score=0.0,
                medical_relevance_score=0.0,
                safety_score=0.0,
                collaboration_quality=0.0,
                overall_score=0.0,
                agent_metadata={"error": str(e)},
                biomcp_used=False
            )
    
    async def evaluate_all_samples(self, samples: List[MedXpertSample]) -> List[EvaluationResult]:
        """Evaluate all samples."""
        logger.info(f"🚀 Starting evaluation of {len(samples)} samples")
        
        results = []
        for i, sample in enumerate(samples, 1):
            logger.info(f"📝 Processing sample {i}/{len(samples)}: {sample.id}")
            
            result = await self.evaluate_single_sample(sample)
            results.append(result)
            
            # Brief pause between samples
            await asyncio.sleep(1)
        
        logger.info(f"✅ Completed evaluation of all {len(samples)} samples")
        return results
    
    def analyze_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze evaluation results and generate summary statistics."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Calculate aggregate metrics
        total_samples = len(results)
        successful_samples = len([r for r in results if r.overall_score > 0])
        
        # Score aggregations
        accuracy_scores = [r.accuracy_score for r in results if r.accuracy_score > 0]
        medical_relevance_scores = [r.medical_relevance_score for r in results if r.medical_relevance_score > 0]
        safety_scores = [r.safety_score for r in results if r.safety_score > 0]
        collaboration_scores = [r.collaboration_quality for r in results if r.collaboration_quality > 0]
        overall_scores = [r.overall_score for r in results if r.overall_score > 0]
        
        # Processing time stats
        processing_times = [r.processing_time for r in results]
        
        # BioMCP usage
        biomcp_usage = sum(1 for r in results if r.biomcp_used)
        
        # Collaboration rounds
        collaboration_rounds = [r.collaboration_rounds for r in results]
        
        analysis = {
            "summary": {
                "total_samples": total_samples,
                "successful_evaluations": successful_samples,
                "success_rate": successful_samples / total_samples if total_samples > 0 else 0,
                "biomcp_usage_rate": biomcp_usage / total_samples if total_samples > 0 else 0
            },
            "performance_metrics": {
                "accuracy": {
                    "mean": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
                    "min": min(accuracy_scores) if accuracy_scores else 0,
                    "max": max(accuracy_scores) if accuracy_scores else 0,
                    "count": len(accuracy_scores)
                },
                "medical_relevance": {
                    "mean": sum(medical_relevance_scores) / len(medical_relevance_scores) if medical_relevance_scores else 0,
                    "min": min(medical_relevance_scores) if medical_relevance_scores else 0,
                    "max": max(medical_relevance_scores) if medical_relevance_scores else 0,
                    "count": len(medical_relevance_scores)
                },
                "safety": {
                    "mean": sum(safety_scores) / len(safety_scores) if safety_scores else 0,
                    "min": min(safety_scores) if safety_scores else 0,
                    "max": max(safety_scores) if safety_scores else 0,
                    "count": len(safety_scores)
                },
                "collaboration": {
                    "mean": sum(collaboration_scores) / len(collaboration_scores) if collaboration_scores else 0,
                    "min": min(collaboration_scores) if collaboration_scores else 0,
                    "max": max(collaboration_scores) if collaboration_scores else 0,
                    "count": len(collaboration_scores)
                },
                "overall": {
                    "mean": sum(overall_scores) / len(overall_scores) if overall_scores else 0,
                    "min": min(overall_scores) if overall_scores else 0,
                    "max": max(overall_scores) if overall_scores else 0,
                    "count": len(overall_scores)
                }
            },
            "efficiency_metrics": {
                "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "avg_collaboration_rounds": sum(collaboration_rounds) / len(collaboration_rounds) if collaboration_rounds else 0
            },
            "detailed_results": [asdict(result) for result in results]
        }
        
        return analysis
    
    def save_results(self, results: List[EvaluationResult], analysis: Dict[str, Any], 
                    output_file: str = "medxpert_evaluation_results.json"):
        """Save evaluation results to file."""
        output_data = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_config": "multi_agent_medical_system",
            "benchmark": "MedXpert",
            "analysis": analysis,
            "individual_results": [asdict(result) for result in results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"💾 Results saved to: {output_file}")
    
    def print_summary(self, analysis: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("🏥 MEDXPERT MULTI-AGENT EVALUATION SUMMARY")
        print("="*60)
        
        summary = analysis["summary"]
        metrics = analysis["performance_metrics"]
        efficiency = analysis["efficiency_metrics"]
        
        print(f"\n📊 EVALUATION OVERVIEW:")
        print(f"  Total Samples: {summary['total_samples']}")
        print(f"  Successful Evaluations: {summary['successful_evaluations']}")
        print(f"  Success Rate: {summary['success_rate']:.1%}")
        print(f"  BioMCP Usage Rate: {summary['biomcp_usage_rate']:.1%}")
        
        print(f"\n🎯 PERFORMANCE METRICS:")
        print(f"  Overall Score: {metrics['overall']['mean']:.3f} (±{(metrics['overall']['max'] - metrics['overall']['min'])/2:.3f})")
        print(f"  Accuracy: {metrics['accuracy']['mean']:.3f}")
        print(f"  Medical Relevance: {metrics['medical_relevance']['mean']:.3f}")
        print(f"  Safety Score: {metrics['safety']['mean']:.3f}")
        print(f"  Collaboration Quality: {metrics['collaboration']['mean']:.3f}")
        
        print(f"\n⚡ EFFICIENCY METRICS:")
        print(f"  Avg Processing Time: {efficiency['avg_processing_time']:.2f}s")
        print(f"  Avg Collaboration Rounds: {efficiency['avg_collaboration_rounds']:.1f}")
        
        print(f"\n🏆 BENCHMARK COMPARISON:")
        overall_score = metrics['overall']['mean']
        if overall_score >= 0.8:
            print(f"  Performance: EXCELLENT (>{overall_score:.1%})")
        elif overall_score >= 0.7:
            print(f"  Performance: GOOD ({overall_score:.1%})")
        elif overall_score >= 0.6:
            print(f"  Performance: MODERATE ({overall_score:.1%})")
        else:
            print(f"  Performance: NEEDS IMPROVEMENT ({overall_score:.1%})")
        
        print("\n" + "="*60)

async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Multi-Agent Medical System on MedXpert")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, default="medxpert_evaluation_results.json", 
                       help="Output file for results")
    parser.add_argument("--config", type=str, help="Path to custom config file")
    
    args = parser.parse_args()
    
    print("🏥 Starting MedXpert Multi-Agent Evaluation")
    print(f"📊 Evaluating {args.samples} samples")
    
    # Initialize evaluator
    evaluator = MedXpertEvaluator(config_path=args.config)
    
    # Create sample data
    all_samples = evaluator.create_sample_medxpert_data()
    samples_to_evaluate = all_samples[:args.samples]
    
    # Run evaluation
    results = await evaluator.evaluate_all_samples(samples_to_evaluate)
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Save results
    evaluator.save_results(results, analysis, args.output)
    
    # Print summary
    evaluator.print_summary(analysis)
    
    print(f"\n✅ Evaluation completed! Results saved to: {args.output}")

if __name__ == "__main__":
    asyncio.run(main()) 