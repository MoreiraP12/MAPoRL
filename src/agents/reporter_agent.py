"""
Medical Reporter Agent - Synthesizes findings and provides final answers.
"""

from typing import Dict, List, Any, Tuple
import logging
import json
from .base_agent import BaseMedicalAgent, MedicalState

logger = logging.getLogger(__name__)

class MedicalReporterAgent(BaseMedicalAgent):
    """Agent responsible for synthesizing information and providing final medical reports."""
    
    def __init__(self, model_config: dict):
        super().__init__("reporter", model_config)
        self.report_templates = {
            "diagnosis": {
                "sections": [
                    "Clinical Presentation",
                    "Differential Diagnosis",
                    "Recommended Workup",
                    "Most Likely Diagnosis",
                    "Treatment Recommendations",
                    "Follow-up Plan"
                ]
            },
            "treatment": {
                "sections": [
                    "Clinical Assessment",
                    "Treatment Options",
                    "Recommended Approach",
                    "Monitoring Plan",
                    "Patient Education",
                    "Follow-up Schedule"
                ]
            },
            "general": {
                "sections": [
                    "Question Analysis",
                    "Key Findings",
                    "Evidence Summary",
                    "Clinical Recommendations",
                    "Safety Considerations",
                    "Next Steps"
                ]
            }
        }
        
        self.quality_metrics = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "safety": 0.0,
            "clarity": 0.0
        }
    
    def get_role_description(self) -> str:
        """Return description of reporter role."""
        return """You are a Medical Reporter Agent. Your role is to:
1. Synthesize information from all team members
2. Create comprehensive medical reports
3. Provide clear, actionable recommendations
4. Ensure clinical accuracy and safety
5. Present findings in a structured, professional format

Focus on clarity, completeness, and clinical accuracy in your final reports."""

    def determine_report_type(self, question: str) -> str:
        """Determine the appropriate report template based on the question."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["diagnose", "diagnosis", "what condition", "differential"]):
            return "diagnosis"
        elif any(word in question_lower for word in ["treat", "treatment", "therapy", "manage", "medication"]):
            return "treatment"
        else:
            return "general"

    def extract_key_information(self, state: MedicalState) -> Dict[str, Any]:
        """Extract key information from all agent responses."""
        extracted_info = {
            "medical_entities": set(),
            "safety_flags": [],
            "confidence_scores": {},
            "key_findings": [],
            "recommendations": []
        }
        
        # Process each agent's responses
        for agent_name, responses in state.agent_responses.items():
            if not responses:
                continue
                
            latest_response = responses[-1]
            
            # Extract medical entities
            entities = self.extract_medical_entities(latest_response)
            extracted_info["medical_entities"].update(entities)
            
            # Extract safety flags
            safety_flags = self.assess_safety(latest_response)
            extracted_info["safety_flags"].extend(safety_flags)
            
            # Extract key findings based on agent type
            if agent_name == "planner":
                findings = self.extract_planning_findings(latest_response)
                extracted_info["key_findings"].extend(findings)
            elif agent_name == "researcher":
                findings = self.extract_research_findings(latest_response)
                extracted_info["key_findings"].extend(findings)
            elif agent_name == "analyst":
                findings = self.extract_analysis_findings(latest_response)
                extracted_info["key_findings"].extend(findings)
        
        # Convert set to list for JSON serialization
        extracted_info["medical_entities"] = list(extracted_info["medical_entities"])
        
        return extracted_info

    def extract_planning_findings(self, response: str) -> List[str]:
        """Extract key findings from planner response."""
        findings = []
        
        if "MEDICAL INVESTIGATION PLAN" in response:
            findings.append("Systematic investigation plan developed")
        if "Question Analysis:" in response:
            findings.append("Question type identified and analyzed")
        if "Team Collaboration Strategy:" in response:
            findings.append("Collaborative approach established")
        
        return findings

    def extract_research_findings(self, response: str) -> List[str]:
        """Extract key findings from researcher response."""
        findings = []
        
        if "MEDICAL RESEARCH FINDINGS" in response:
            findings.append("Evidence-based research conducted")
        if "Evidence Level:" in response:
            findings.append("Evidence quality assessed")
        if "Clinical Guidelines:" in response:
            findings.append("Clinical guidelines referenced")
        
        return findings

    def extract_analysis_findings(self, response: str) -> List[str]:
        """Extract key findings from analyst response."""
        findings = []
        
        if "CLINICAL ANALYSIS" in response:
            findings.append("Clinical analysis performed")
        if "Differential Analysis:" in response:
            findings.append("Differential diagnosis considered")
        if "Confidence Level:" in response:
            findings.append("Clinical confidence assessed")
        
        return findings

    def assess_report_quality(self, report: str, extracted_info: Dict[str, Any]) -> Dict[str, float]:
        """Assess the quality of the generated report."""
        quality_scores = {
            "completeness": 0.0,
            "accuracy": 0.0,
            "safety": 0.0,
            "clarity": 0.0
        }
        
        # Assess completeness
        required_sections = ["findings", "recommendations", "considerations"]
        completeness_score = sum(1 for section in required_sections if section.lower() in report.lower())
        quality_scores["completeness"] = completeness_score / len(required_sections)
        
        # Assess accuracy (based on evidence references)
        accuracy_indicators = ["evidence", "guidelines", "study", "clinical"]
        accuracy_score = sum(1 for indicator in accuracy_indicators if indicator in report.lower())
        quality_scores["accuracy"] = min(accuracy_score / len(accuracy_indicators), 1.0)
        
        # Assess safety (fewer safety flags = higher safety score)
        safety_flags_count = len(extracted_info.get("safety_flags", []))
        quality_scores["safety"] = max(0.0, 1.0 - (safety_flags_count * 0.2))
        
        # Assess clarity (based on structure and length)
        clarity_score = 0.5  # Base score
        if len(report.split()) > 100:  # Adequate length
            clarity_score += 0.2
        if report.count('\n') > 5:  # Good structure
            clarity_score += 0.2
        if "SUMMARY" in report or "CONCLUSION" in report:  # Clear conclusion
            clarity_score += 0.1
        
        quality_scores["clarity"] = min(clarity_score, 1.0)
        
        return quality_scores

    def generate_structured_report(self, state: MedicalState, extracted_info: Dict[str, Any]) -> str:
        """Generate a structured medical report."""
        report_type = self.determine_report_type(state.question)
        template = self.report_templates[report_type]
        
        report = f"""
MEDICAL CONSULTATION REPORT
{'=' * 50}

QUESTION: {state.question}

CONTEXT: {state.context}

COLLABORATIVE ANALYSIS SUMMARY:
{'=' * 30}

Key Medical Entities Identified: {', '.join(extracted_info['medical_entities'])}

Team Findings:
{chr(10).join(f"• {finding}" for finding in extracted_info['key_findings'])}

CLINICAL RECOMMENDATIONS:
{'=' * 30}

"""
        
        # Add recommendations based on agent responses
        for agent_name, responses in state.agent_responses.items():
            if responses:
                report += f"\n{agent_name.title()} Input:\n"
                # Extract key points from the latest response
                latest_response = responses[-1]
                if "recommendations" in latest_response.lower():
                    lines = latest_response.split('\n')
                    for line in lines:
                        if any(keyword in line.lower() for keyword in ["recommend", "suggest", "advise", "consider"]):
                            report += f"• {line.strip()}\n"
        
        # Add safety considerations
        if extracted_info['safety_flags']:
            report += f"""
SAFETY CONSIDERATIONS:
{'=' * 30}
"""
            for flag in extracted_info['safety_flags']:
                report += f"⚠️ {flag}\n"
        else:
            report += f"""
SAFETY CONSIDERATIONS:
{'=' * 30}
No specific safety concerns identified by the analysis team.
"""
        
        report += f"""
QUALITY ASSESSMENT:
{'=' * 30}
- Medical entities identified: {len(extracted_info['medical_entities'])}
- Team collaboration: {len([r for r in state.agent_responses.values() if r])} agents contributed
- Safety flags: {len(extracted_info['safety_flags'])}

FINAL ANSWER:
{'=' * 30}
Based on the collaborative analysis of the medical team, the key findings and recommendations are synthesized above. This report represents the collective expertise of the planning, research, and analysis agents working together to address the medical question.
"""
        
        return report

    def process_state(self, state: MedicalState) -> Tuple[str, Dict[str, Any]]:
        """Process the current state and generate the final medical report."""
        try:
            # Extract key information from all agent responses
            extracted_info = self.extract_key_information(state)
            
            # Generate structured report
            structured_report = self.generate_structured_report(state, extracted_info)
            
            # Create additional context for the reporter
            additional_context = f"""
Team Collaboration Summary:
- Agents participated: {len([r for r in state.agent_responses.values() if r])}
- Medical entities identified: {len(extracted_info['medical_entities'])}
- Safety flags raised: {len(extracted_info['safety_flags'])}

Generate a comprehensive final report synthesizing all team inputs.
"""
            
            prompt = self.format_prompt(state, additional_context)
            response = self.generate_response(prompt)
            
            # Combine structured report with generated response
            final_report = f"{structured_report}\n\nADDITIONAL SYNTHESIS:\n{response}"
            
            # Assess report quality
            quality_scores = self.assess_report_quality(final_report, extracted_info)
            
            # Update response history
            self.update_response_history(final_report)
            
            # Extract medical entities and assess safety
            medical_entities = self.extract_medical_entities(final_report)
            safety_flags = self.assess_safety(final_report)
            
            # Calculate overall confidence
            overall_confidence = sum(quality_scores.values()) / len(quality_scores)
            
            metadata = {
                "agent_name": self.agent_name,
                "extracted_info": extracted_info,
                "quality_scores": quality_scores,
                "overall_confidence": overall_confidence,
                "medical_entities": medical_entities,
                "safety_flags": safety_flags,
                "confidence": overall_confidence,
                "report_type": self.determine_report_type(state.question)
            }
            
            return final_report, metadata
            
        except Exception as e:
            logger.error(f"Error in reporter processing: {e}")
            fallback_response = f"""
BASIC MEDICAL REPORT:

Question: {state.question}
Context: {state.context}

Summary: Unable to generate comprehensive report due to technical error.
Recommendation: Consult with medical professionals for clinical guidance.

Error: {str(e)}
"""
            metadata = {
                "agent_name": self.agent_name,
                "error": str(e),
                "confidence": 0.1
            }
            return fallback_response, metadata

    def validate_final_answer(self, answer: str) -> Dict[str, Any]:
        """Validate the final answer for medical appropriateness."""
        validation = {
            "is_valid": True,
            "issues": [],
            "score": 1.0
        }
        
        # Check for dangerous absolute statements
        dangerous_phrases = [
            "definitely", "certainly", "always", "never", "100%", "guaranteed"
        ]
        
        for phrase in dangerous_phrases:
            if phrase in answer.lower():
                validation["issues"].append(f"Overconfident language: {phrase}")
                validation["score"] -= 0.1
        
        # Check for missing disclaimers
        if "consult" not in answer.lower() and "medical professional" not in answer.lower():
            validation["issues"].append("Missing medical professional consultation recommendation")
            validation["score"] -= 0.1
        
        # Ensure score is not negative
        validation["score"] = max(validation["score"], 0.0)
        validation["is_valid"] = validation["score"] > 0.5
        
        return validation 