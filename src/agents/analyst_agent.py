"""
Medical Analyst Agent - Provides clinical reasoning and analysis.
"""

from typing import Dict, List, Any, Tuple
import logging
import re
from .base_agent import BaseMedicalAgent, MedicalState

logger = logging.getLogger(__name__)

class MedicalAnalystAgent(BaseMedicalAgent):
    """Agent responsible for clinical reasoning and medical analysis."""
    
    def __init__(self, model_config: dict):
        super().__init__("analyst", model_config)
        self.clinical_reasoning_frameworks = {
            "diagnostic_reasoning": [
                "Pattern recognition",
                "Differential diagnosis",
                "Probabilistic reasoning",
                "Hypothesis testing"
            ],
            "treatment_analysis": [
                "Risk-benefit assessment",
                "Evidence evaluation",
                "Patient-specific factors",
                "Outcome prediction"
            ],
            "safety_analysis": [
                "Contraindications",
                "Drug interactions",
                "Adverse effects",
                "Monitoring requirements"
            ]
        }
        
        self.clinical_red_flags = [
            "chest pain", "shortness of breath", "severe headache",
            "loss of consciousness", "severe bleeding", "high fever",
            "severe abdominal pain", "neurological deficit"
        ]
        
        self.risk_factors = {
            "cardiovascular": ["hypertension", "diabetes", "smoking", "obesity", "family history"],
            "infectious": ["immunocompromised", "recent travel", "exposure", "fever"],
            "oncologic": ["weight loss", "night sweats", "fatigue", "family history"]
        }
    
    def get_role_description(self) -> str:
        """Return description of analyst role."""
        return """You are a Medical Analyst Agent. Your role is to:
1. Provide clinical reasoning and logical analysis
2. Evaluate evidence and synthesize information
3. Identify potential risks and contraindications
4. Apply clinical decision-making frameworks
5. Assess patient-specific factors and considerations

Focus on systematic clinical reasoning, risk assessment, and evidence-based analysis."""

    def identify_clinical_patterns(self, text: str) -> Dict[str, List[str]]:
        """Identify clinical patterns and red flags in the text."""
        patterns = {
            "red_flags": [],
            "risk_factors": [],
            "clinical_signs": []
        }
        
        text_lower = text.lower()
        
        # Check for red flags
        for flag in self.clinical_red_flags:
            if flag in text_lower:
                patterns["red_flags"].append(flag)
        
        # Check for risk factors
        for category, factors in self.risk_factors.items():
            for factor in factors:
                if factor in text_lower:
                    patterns["risk_factors"].append(f"{factor} ({category})")
        
        # Extract clinical signs (simple regex pattern)
        clinical_signs = re.findall(r'\b(?:fever|pain|swelling|rash|nausea|vomiting|diarrhea|fatigue)\b', text_lower)
        patterns["clinical_signs"] = list(set(clinical_signs))
        
        return patterns

    def perform_differential_analysis(self, question: str, context: str, previous_responses: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform differential diagnosis analysis."""
        analysis = {
            "primary_considerations": [],
            "differential_diagnoses": [],
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "additional_workup": []
        }
        
        # Extract clinical information
        combined_text = f"{question} {context}"
        for responses in previous_responses.values():
            combined_text += " ".join(responses)
        
        clinical_patterns = self.identify_clinical_patterns(combined_text)
        
        # Analyze based on identified patterns
        if clinical_patterns["red_flags"]:
            analysis["primary_considerations"].append("Urgent evaluation needed due to red flag symptoms")
        
        if clinical_patterns["risk_factors"]:
            analysis["supporting_evidence"].extend(clinical_patterns["risk_factors"])
        
        if clinical_patterns["clinical_signs"]:
            analysis["differential_diagnoses"].extend([
                f"Condition associated with {sign}" for sign in clinical_patterns["clinical_signs"]
            ])
        
        # Suggest additional workup based on findings
        if "pain" in clinical_patterns["clinical_signs"]:
            analysis["additional_workup"].append("Pain assessment and localization")
        if "fever" in clinical_patterns["clinical_signs"]:
            analysis["additional_workup"].append("Complete blood count, blood cultures")
        
        return analysis

    def assess_treatment_appropriateness(self, treatment_info: str) -> Dict[str, Any]:
        """Assess the appropriateness of proposed treatments."""
        assessment = {
            "appropriateness_score": 0.5,  # Default neutral score
            "considerations": [],
            "contraindications": [],
            "monitoring_needs": []
        }
        
        treatment_lower = treatment_info.lower()
        
        # Check for evidence-based treatments
        if any(term in treatment_lower for term in ["guideline", "evidence", "standard", "recommended"]):
            assessment["appropriateness_score"] += 0.2
            assessment["considerations"].append("Evidence-based approach")
        
        # Check for safety considerations
        if any(term in treatment_lower for term in ["contraindication", "allergy", "interaction"]):
            assessment["considerations"].append("Safety factors addressed")
        else:
            assessment["considerations"].append("Consider safety factors and contraindications")
        
        # Check for monitoring mentions
        if any(term in treatment_lower for term in ["monitor", "follow-up", "check"]):
            assessment["considerations"].append("Monitoring plan included")
        else:
            assessment["monitoring_needs"].append("Establish monitoring plan")
        
        # Ensure score is within bounds
        assessment["appropriateness_score"] = min(max(assessment["appropriateness_score"], 0.0), 1.0)
        
        return assessment

    def calculate_clinical_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence level based on analysis results."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for comprehensive analysis
        if analysis_results.get("primary_considerations"):
            confidence += 0.1
        if analysis_results.get("supporting_evidence"):
            confidence += 0.1
        if analysis_results.get("additional_workup"):
            confidence += 0.1
        
        # Decrease confidence for red flags (more caution needed)
        if analysis_results.get("red_flags"):
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)

    def process_state(self, state: MedicalState) -> Tuple[str, Dict[str, Any]]:
        """Process the current state and provide clinical analysis."""
        try:
            # Perform differential analysis
            differential_analysis = self.perform_differential_analysis(
                state.question, state.context, state.agent_responses
            )
            
            # Assess treatment information if available
            treatment_assessment = None
            combined_responses = ""
            for responses in state.agent_responses.values():
                combined_responses += " ".join(responses)
            
            if any(term in combined_responses.lower() for term in ["treatment", "therapy", "medication"]):
                treatment_assessment = self.assess_treatment_appropriateness(combined_responses)
            
            # Calculate confidence
            clinical_confidence = self.calculate_clinical_confidence(differential_analysis)
            
            # Create analysis context
            additional_context = f"""
Clinical Patterns Identified: {len(differential_analysis.get('primary_considerations', []))} primary considerations
Red Flags Present: {len(differential_analysis.get('red_flags', []))} 
Risk Factors: {len(differential_analysis.get('supporting_evidence', []))}

Provide systematic clinical analysis and reasoning.
"""
            
            prompt = self.format_prompt(state, additional_context)
            response = self.generate_response(prompt)
            
            # Enhance response with structured analysis
            structured_response = f"""
CLINICAL ANALYSIS:

Differential Analysis:
"""
            
            if differential_analysis["primary_considerations"]:
                structured_response += f"Primary Considerations:\n"
                for consideration in differential_analysis["primary_considerations"]:
                    structured_response += f"- {consideration}\n"
            
            if differential_analysis["supporting_evidence"]:
                structured_response += f"\nSupporting Evidence:\n"
                for evidence in differential_analysis["supporting_evidence"]:
                    structured_response += f"- {evidence}\n"
            
            if differential_analysis["additional_workup"]:
                structured_response += f"\nRecommended Additional Workup:\n"
                for workup in differential_analysis["additional_workup"]:
                    structured_response += f"- {workup}\n"
            
            if treatment_assessment:
                structured_response += f"""
Treatment Appropriateness Assessment:
- Appropriateness Score: {treatment_assessment['appropriateness_score']:.2f}/1.0
- Key Considerations: {'; '.join(treatment_assessment['considerations'])}
"""
                if treatment_assessment['monitoring_needs']:
                    structured_response += f"- Monitoring Needs: {'; '.join(treatment_assessment['monitoring_needs'])}\n"
            
            structured_response += f"""
Clinical Reasoning:
{response}

Confidence Level: {clinical_confidence:.2f}/1.0
Analysis Quality: {'High' if clinical_confidence > 0.7 else 'Moderate' if clinical_confidence > 0.4 else 'Low'}
"""
            
            # Update response history
            self.update_response_history(structured_response)
            
            # Extract medical entities and assess safety
            medical_entities = self.extract_medical_entities(structured_response)
            safety_flags = self.assess_safety(structured_response)
            
            metadata = {
                "agent_name": self.agent_name,
                "differential_analysis": differential_analysis,
                "treatment_assessment": treatment_assessment,
                "clinical_confidence": clinical_confidence,
                "medical_entities": medical_entities,
                "safety_flags": safety_flags,
                "confidence": clinical_confidence
            }
            
            return structured_response, metadata
            
        except Exception as e:
            logger.error(f"Error in analyst processing: {e}")
            fallback_response = f"""
BASIC CLINICAL ANALYSIS:

Question: {state.question}
Context: {state.context}

Clinical Approach:
1. Systematic evaluation of presented information
2. Consider differential diagnoses
3. Assess risk factors and contraindications
4. Recommend appropriate workup and monitoring

Error in detailed analysis: {str(e)}
"""
            metadata = {
                "agent_name": self.agent_name,
                "error": str(e),
                "confidence": 0.2
            }
            return fallback_response, metadata

    def prioritize_diagnoses(self, diagnoses: List[str], risk_factors: List[str]) -> List[Dict[str, Any]]:
        """Prioritize differential diagnoses based on risk factors."""
        prioritized = []
        
        for diagnosis in diagnoses:
            priority_score = 0.5  # Base score
            
            # Increase priority for diagnoses matching risk factors
            for risk_factor in risk_factors:
                if any(rf.lower() in diagnosis.lower() for rf in risk_factor.split()):
                    priority_score += 0.1
            
            prioritized.append({
                "diagnosis": diagnosis,
                "priority_score": min(priority_score, 1.0),
                "rationale": f"Score based on risk factor alignment"
            })
        
        return sorted(prioritized, key=lambda x: x["priority_score"], reverse=True)