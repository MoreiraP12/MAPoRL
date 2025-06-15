"""
Medical Reward System for MAPoRL Training.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Tuple
import numpy as np
import logging
from dataclasses import dataclass
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import json

logger = logging.getLogger(__name__)

@dataclass
class MedicalRewardComponents:
    """Components of the medical reward system."""
    accuracy_score: float = 0.0
    medical_relevance_score: float = 0.0
    collaboration_quality_score: float = 0.0
    safety_score: float = 0.0
    evidence_quality_score: float = 0.0
    clinical_reasoning_score: float = 0.0
    
    def total_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted total score."""
        if weights is None:
            weights = {
                "accuracy": 0.25,
                "medical_relevance": 0.2,
                "collaboration_quality": 0.15,
                "safety": 0.2,
                "evidence_quality": 0.1,
                "clinical_reasoning": 0.1
            }
        
        return (
            self.accuracy_score * weights.get("accuracy", 0.25) +
            self.medical_relevance_score * weights.get("medical_relevance", 0.2) +
            self.collaboration_quality_score * weights.get("collaboration_quality", 0.15) +
            self.safety_score * weights.get("safety", 0.2) +
            self.evidence_quality_score * weights.get("evidence_quality", 0.1) +
            self.clinical_reasoning_score * weights.get("clinical_reasoning", 0.1)
        )

class MedicalRewardSystem:
    """Comprehensive reward system for medical multi-agent training."""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Medical knowledge base for reference
        self.medical_knowledge = {
            "symptoms": [
                "fever", "pain", "nausea", "fatigue", "headache", "cough",
                "shortness of breath", "chest pain", "dizziness", "rash"
            ],
            "diseases": [
                "hypertension", "diabetes", "pneumonia", "influenza", "covid-19",
                "heart disease", "stroke", "cancer", "asthma", "arthritis"
            ],
            "treatments": [
                "medication", "surgery", "therapy", "rest", "exercise",
                "antibiotics", "vaccination", "lifestyle changes"
            ],
            "clinical_guidelines": [
                "evidence-based", "randomized controlled trial", "systematic review",
                "clinical practice guideline", "meta-analysis", "cohort study"
            ]
        }
        
        # Safety patterns to avoid
        self.safety_violations = [
            "definitely", "certainly will cure", "guaranteed", "always works",
            "no side effects", "completely safe", "never fails", "100% effective"
        ]
        
        # Collaboration quality indicators
        self.collaboration_indicators = {
            "positive": [
                "based on", "according to", "building on", "agrees with",
                "supports", "consistent with", "evidence shows", "studies indicate"
            ],
            "negative": [
                "ignores", "contradicts without reason", "dismisses", "repeats exactly"
            ]
        }

    def evaluate_accuracy(self, prediction: str, ground_truth: str) -> float:
        """Evaluate accuracy of medical predictions."""
        if not ground_truth:
            return 0.5  # Neutral score when no ground truth available
        
        try:
            # Use sentence transformer for semantic similarity
            pred_embedding = self.sentence_transformer.encode([prediction])
            truth_embedding = self.sentence_transformer.encode([ground_truth])
            
            similarity = cosine_similarity(pred_embedding, truth_embedding)[0][0]
            
            # Convert similarity to 0-1 range
            accuracy_score = max(0.0, min(1.0, (similarity + 1) / 2))
            
            return accuracy_score
            
        except Exception as e:
            logger.error(f"Error evaluating accuracy: {e}")
            return 0.1

    def evaluate_medical_relevance(self, response: str, question: str) -> float:
        """Evaluate medical relevance of the response."""
        try:
            response_lower = response.lower()
            question_lower = question.lower()
            
            relevance_score = 0.0
            
            # Check for medical terminology
            medical_terms_found = 0
            total_medical_terms = 0
            
            for category, terms in self.medical_knowledge.items():
                total_medical_terms += len(terms)
                for term in terms:
                    if term in response_lower or term in question_lower:
                        medical_terms_found += 1
            
            # Base relevance from medical terminology
            if total_medical_terms > 0:
                relevance_score += 0.3 * (medical_terms_found / total_medical_terms)
            
            # Check for clinical reasoning patterns
            clinical_patterns = [
                "diagnosis", "treatment", "symptoms", "prognosis", "etiology",
                "pathophysiology", "differential", "workup", "monitoring"
            ]
            
            pattern_score = sum(1 for pattern in clinical_patterns if pattern in response_lower)
            relevance_score += 0.4 * min(pattern_score / len(clinical_patterns), 1.0)
            
            # Check for evidence-based language
            evidence_terms = ["evidence", "study", "research", "guideline", "trial"]
            evidence_score = sum(1 for term in evidence_terms if term in response_lower)
            relevance_score += 0.3 * min(evidence_score / len(evidence_terms), 1.0)
            
            return min(relevance_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating medical relevance: {e}")
            return 0.1

    def evaluate_collaboration_quality(self, agent_responses: Dict[str, List[str]]) -> float:
        """Evaluate quality of collaboration between agents."""
        try:
            if len(agent_responses) < 2:
                return 0.0
            
            collaboration_score = 0.0
            
            # Check for reference to other agents' contributions
            reference_count = 0
            total_responses = 0
            
            for agent_name, responses in agent_responses.items():
                for response in responses:
                    total_responses += 1
                    response_lower = response.lower()
                    
                    # Check for positive collaboration indicators
                    for indicator in self.collaboration_indicators["positive"]:
                        if indicator in response_lower:
                            reference_count += 1
                            break
                    
                    # Penalize negative collaboration indicators
                    for indicator in self.collaboration_indicators["negative"]:
                        if indicator in response_lower:
                            reference_count -= 1
                            break
            
            if total_responses > 0:
                collaboration_score += 0.4 * max(0, reference_count / total_responses)
            
            # Check for information building across agents
            all_text = " ".join([
                " ".join(responses) for responses in agent_responses.values()
            ])
            
            # Measure information diversity
            unique_sentences = set()
            for responses in agent_responses.values():
                for response in responses:
                    sentences = response.split('.')
                    unique_sentences.update(s.strip() for s in sentences if len(s.strip()) > 10)
            
            if len(unique_sentences) > 0:
                diversity_score = min(len(unique_sentences) / (total_responses * 2), 1.0)
                collaboration_score += 0.3 * diversity_score
            
            # Check for complementary information
            agent_names = list(agent_responses.keys())
            complementary_score = 0.0
            
            for i, agent1 in enumerate(agent_names):
                for j, agent2 in enumerate(agent_names):
                    if i != j and agent_responses[agent1] and agent_responses[agent2]:
                        text1 = " ".join(agent_responses[agent1])
                        text2 = " ".join(agent_responses[agent2])
                        
                        # Check for complementary content (different focus areas)
                        if self._has_complementary_content(text1, text2):
                            complementary_score += 1
            
            max_pairs = len(agent_names) * (len(agent_names) - 1)
            if max_pairs > 0:
                collaboration_score += 0.3 * (complementary_score / max_pairs)
            
            return min(collaboration_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating collaboration quality: {e}")
            return 0.1

    def _has_complementary_content(self, text1: str, text2: str) -> bool:
        """Check if two texts have complementary content."""
        # Simple heuristic: different focus on medical aspects
        focus_areas = {
            "planning": ["plan", "approach", "strategy", "steps"],
            "research": ["evidence", "study", "research", "literature"],
            "analysis": ["analysis", "reasoning", "differential", "assessment"],
            "reporting": ["summary", "conclusion", "recommendation", "report"]
        }
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        text1_focuses = set()
        text2_focuses = set()
        
        for focus, keywords in focus_areas.items():
            if any(keyword in text1_lower for keyword in keywords):
                text1_focuses.add(focus)
            if any(keyword in text2_lower for keyword in keywords):
                text2_focuses.add(focus)
        
        # Complementary if they have different focuses
        return len(text1_focuses.intersection(text2_focuses)) < len(text1_focuses.union(text2_focuses))

    def evaluate_safety(self, response: str) -> float:
        """Evaluate safety of medical response."""
        try:
            response_lower = response.lower()
            safety_score = 1.0  # Start with perfect safety
            
            # Check for safety violations
            violations = 0
            for violation in self.safety_violations:
                if violation in response_lower:
                    violations += 1
            
            # Penalize safety violations
            safety_score -= 0.2 * violations
            
            # Check for appropriate disclaimers
            disclaimers = [
                "consult", "medical professional", "healthcare provider", 
                "doctor", "physician", "see your doctor"
            ]
            
            has_disclaimer = any(disclaimer in response_lower for disclaimer in disclaimers)
            if not has_disclaimer and len(response.split()) > 20:  # Only for substantial responses
                safety_score -= 0.1
            
            # Check for appropriate uncertainty language
            uncertainty_terms = [
                "may", "might", "could", "possible", "likely", "suggest",
                "appear", "seem", "potential", "consider"
            ]
            
            has_uncertainty = any(term in response_lower for term in uncertainty_terms)
            if not has_uncertainty and len(response.split()) > 20:
                safety_score -= 0.1
            
            return max(safety_score, 0.0)
            
        except Exception as e:
            logger.error(f"Error evaluating safety: {e}")
            return 0.5

    def evaluate_evidence_quality(self, response: str) -> float:
        """Evaluate quality of evidence cited in response."""
        try:
            response_lower = response.lower()
            evidence_score = 0.0
            
            # Check for evidence-based terms
            evidence_terms = {
                "high_quality": ["systematic review", "meta-analysis", "rct", "randomized controlled trial"],
                "medium_quality": ["cohort study", "case-control", "clinical trial"],
                "guidelines": ["guideline", "recommendation", "consensus", "standard of care"],
                "sources": ["pubmed", "cochrane", "nejm", "jama", "bmj"]
            }
            
            for quality_level, terms in evidence_terms.items():
                for term in terms:
                    if term in response_lower:
                        if quality_level == "high_quality":
                            evidence_score += 0.3
                        elif quality_level == "medium_quality":
                            evidence_score += 0.2
                        elif quality_level == "guidelines":
                            evidence_score += 0.25
                        elif quality_level == "sources":
                            evidence_score += 0.15
                        break  # Only count once per quality level
            
            # Check for appropriate citation format
            if any(pattern in response for pattern in ["et al", "2020", "2021", "2022", "2023", "2024"]):
                evidence_score += 0.1
            
            return min(evidence_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating evidence quality: {e}")
            return 0.1

    def evaluate_clinical_reasoning(self, response: str, question: str) -> float:
        """Evaluate clinical reasoning quality."""
        try:
            response_lower = response.lower()
            reasoning_score = 0.0
            
            # Check for clinical reasoning patterns
            reasoning_patterns = {
                "diagnostic": ["differential", "diagnosis", "rule out", "consider", "likely"],
                "systematic": ["first", "second", "then", "next", "finally", "step"],
                "analytical": ["because", "therefore", "due to", "caused by", "leads to"],
                "risk_assessment": ["risk", "benefit", "contraindication", "precaution"]
            }
            
            for pattern_type, patterns in reasoning_patterns.items():
                pattern_count = sum(1 for pattern in patterns if pattern in response_lower)
                if pattern_count > 0:
                    reasoning_score += 0.2 * min(pattern_count / len(patterns), 1.0)
            
            # Check for structured thinking
            structure_indicators = ["plan:", "assessment:", "recommendation:", "conclusion:"]
            structure_score = sum(1 for indicator in structure_indicators if indicator in response_lower)
            reasoning_score += 0.2 * min(structure_score / len(structure_indicators), 1.0)
            
            return min(reasoning_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error evaluating clinical reasoning: {e}")
            return 0.1

    def calculate_medical_reward(
        self, 
        agent_responses: Dict[str, List[str]], 
        question: str, 
        ground_truth: str = "", 
        context: str = "",
        weights: Dict[str, float] = None
    ) -> MedicalRewardComponents:
        """Calculate comprehensive medical reward."""
        try:
            # Combine all responses for evaluation
            all_responses = []
            for responses in agent_responses.values():
                all_responses.extend(responses)
            
            combined_response = " ".join(all_responses)
            
            # Calculate individual reward components
            reward_components = MedicalRewardComponents()
            
            # Accuracy (comparing with ground truth if available)
            if ground_truth:
                reward_components.accuracy_score = self.evaluate_accuracy(combined_response, ground_truth)
            else:
                reward_components.accuracy_score = 0.5  # Neutral when no ground truth
            
            # Medical relevance
            reward_components.medical_relevance_score = self.evaluate_medical_relevance(combined_response, question)
            
            # Collaboration quality
            reward_components.collaboration_quality_score = self.evaluate_collaboration_quality(agent_responses)
            
            # Safety
            reward_components.safety_score = self.evaluate_safety(combined_response)
            
            # Evidence quality
            reward_components.evidence_quality_score = self.evaluate_evidence_quality(combined_response)
            
            # Clinical reasoning
            reward_components.clinical_reasoning_score = self.evaluate_clinical_reasoning(combined_response, question)
            
            return reward_components
            
        except Exception as e:
            logger.error(f"Error calculating medical reward: {e}")
            return MedicalRewardComponents()  # Return zeros

    def create_reward_tensor(self, reward_components: MedicalRewardComponents, weights: Dict[str, float] = None) -> torch.Tensor:
        """Create reward tensor for RL training."""
        total_reward = reward_components.total_score(weights)
        return torch.tensor(total_reward, dtype=torch.float32, device=self.device)

    def log_reward_breakdown(self, reward_components: MedicalRewardComponents, agent_name: str = ""):
        """Log detailed reward breakdown for analysis."""
        logger.info(f"Medical Reward Breakdown {f'for {agent_name}' if agent_name else ''}:")
        logger.info(f"  Accuracy: {reward_components.accuracy_score:.3f}")
        logger.info(f"  Medical Relevance: {reward_components.medical_relevance_score:.3f}")
        logger.info(f"  Collaboration Quality: {reward_components.collaboration_quality_score:.3f}")
        logger.info(f"  Safety: {reward_components.safety_score:.3f}")
        logger.info(f"  Evidence Quality: {reward_components.evidence_quality_score:.3f}")
        logger.info(f"  Clinical Reasoning: {reward_components.clinical_reasoning_score:.3f}")
        logger.info(f"  Total Score: {reward_components.total_score():.3f}")

# Utility functions for reward system
def create_medical_reward_system(device: str = "cuda:0") -> MedicalRewardSystem:
    """Create a medical reward system instance."""
    return MedicalRewardSystem(device)

def evaluate_medical_benchmark(
    responses: Dict[str, List[str]], 
    questions: List[str], 
    ground_truths: List[str] = None,
    reward_system: MedicalRewardSystem = None
) -> List[MedicalRewardComponents]:
    """Evaluate responses on medical benchmark."""
    if reward_system is None:
        reward_system = create_medical_reward_system()
    
    if ground_truths is None:
        ground_truths = [""] * len(questions)
    
    results = []
    for i, question in enumerate(questions):
        ground_truth = ground_truths[i] if i < len(ground_truths) else ""
        reward = reward_system.calculate_medical_reward(responses, question, ground_truth)
        results.append(reward)
    
    return results 