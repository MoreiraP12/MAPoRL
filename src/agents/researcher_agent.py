"""
Medical Researcher Agent - Gathers evidence and medical knowledge.
"""

from typing import Dict, List, Any, Tuple
import logging
import json
from .base_agent import BaseMedicalAgent, MedicalState

logger = logging.getLogger(__name__)

class MedicalResearcherAgent(BaseMedicalAgent):
    """Agent responsible for researching medical evidence and guidelines."""
    
    def __init__(self, model_config: dict):
        super().__init__("researcher", model_config)
        self.medical_databases = {
            "guidelines": [
                "American College of Physicians (ACP)",
                "American Heart Association (AHA)",
                "Centers for Disease Control (CDC)",
                "World Health Organization (WHO)",
                "National Institutes of Health (NIH)"
            ],
            "evidence_levels": {
                "Level 1": "Systematic reviews and meta-analyses",
                "Level 2": "Randomized controlled trials",
                "Level 3": "Cohort studies",
                "Level 4": "Case-control studies",
                "Level 5": "Case series and expert opinion"
            }
        }
        
        # Simulated medical knowledge base (in real implementation, this would be a vector database)
        self.knowledge_base = {
            "hypertension": {
                "definition": "Blood pressure consistently above 140/90 mmHg",
                "guidelines": "JNC 8, AHA/ACC 2017",
                "first_line_treatment": "ACE inhibitors, ARBs, thiazide diuretics, CCBs",
                "monitoring": "Regular BP checks, kidney function, electrolytes"
            },
            "diabetes": {
                "definition": "Fasting glucose ≥126 mg/dL or HbA1c ≥6.5%",
                "guidelines": "ADA Standards of Care",
                "first_line_treatment": "Metformin, lifestyle modifications",
                "monitoring": "HbA1c every 3-6 months, annual eye/foot exams"
            },
            "pneumonia": {
                "definition": "Infection of lung parenchyma",
                "guidelines": "IDSA/ATS Guidelines",
                "first_line_treatment": "Antibiotics based on severity and risk factors",
                "monitoring": "Clinical response, chest imaging"
            }
        }
    
    def get_role_description(self) -> str:
        """Return description of researcher role."""
        return """You are a Medical Researcher Agent. Your role is to:
1. Search for and synthesize relevant medical evidence
2. Identify current clinical guidelines and best practices
3. Provide evidence-based information to support decision-making
4. Assess the quality and reliability of medical information
5. Stay current with medical literature and recommendations

Focus on evidence-based medicine, cite relevant guidelines, and provide accurate, up-to-date medical information."""

    def search_knowledge_base(self, query_terms: List[str]) -> Dict[str, Any]:
        """Search the medical knowledge base for relevant information."""
        results = {}
        
        for term in query_terms:
            term_lower = term.lower()
            for condition, info in self.knowledge_base.items():
                if term_lower in condition or any(term_lower in str(value).lower() for value in info.values()):
                    results[condition] = info
        
        return results

    def assess_evidence_quality(self, source: str, study_type: str = "unknown") -> Dict[str, Any]:
        """Assess the quality of medical evidence."""
        assessment = {
            "evidence_level": "Level 5",  # Default to lowest level
            "reliability": "Low",
            "recommendation_strength": "Weak"
        }
        
        # Assess based on source
        if any(db in source.lower() for db in ["cochrane", "pubmed", "nejm", "jama"]):
            assessment["reliability"] = "High"
        elif any(db in source.lower() for db in ["uptodate", "guidelines", "consensus"]):
            assessment["reliability"] = "Moderate"
        
        # Assess based on study type
        if "meta-analysis" in study_type.lower() or "systematic review" in study_type.lower():
            assessment["evidence_level"] = "Level 1"
            assessment["recommendation_strength"] = "Strong"
        elif "randomized" in study_type.lower() or "rct" in study_type.lower():
            assessment["evidence_level"] = "Level 2"
            assessment["recommendation_strength"] = "Moderate"
        elif "cohort" in study_type.lower():
            assessment["evidence_level"] = "Level 3"
            assessment["recommendation_strength"] = "Moderate"
        
        return assessment

    def generate_evidence_summary(self, medical_entities: List[str]) -> Dict[str, Any]:
        """Generate a summary of available evidence for medical entities."""
        evidence_summary = {}
        
        # Search knowledge base
        knowledge_results = self.search_knowledge_base(medical_entities)
        
        for condition, info in knowledge_results.items():
            evidence_summary[condition] = {
                "clinical_info": info,
                "evidence_assessment": self.assess_evidence_quality(info.get("guidelines", ""), "guideline"),
                "key_recommendations": []
            }
            
            # Generate key recommendations based on available info
            if "first_line_treatment" in info:
                evidence_summary[condition]["key_recommendations"].append(
                    f"First-line treatment: {info['first_line_treatment']}"
                )
            if "monitoring" in info:
                evidence_summary[condition]["key_recommendations"].append(
                    f"Monitoring: {info['monitoring']}"
                )
        
        return evidence_summary

    def process_state(self, state: MedicalState) -> Tuple[str, Dict[str, Any]]:
        """Process the current state and provide medical research findings."""
        try:
            # Extract medical entities from question and previous responses
            medical_entities = self.extract_medical_entities(state.question + " " + state.context)
            
            # Add entities from previous agent responses
            for agent_responses in state.agent_responses.values():
                for response in agent_responses:
                    medical_entities.extend(self.extract_medical_entities(response))
            
            # Remove duplicates
            medical_entities = list(set(medical_entities))
            
            # Generate evidence summary
            evidence_summary = self.generate_evidence_summary(medical_entities)
            
            # Create research context
            additional_context = f"""
Medical Entities Identified: {', '.join(medical_entities)}
Evidence Search Focus: Current guidelines, treatment recommendations, and monitoring protocols

Provide evidence-based research findings and recommendations.
"""
            
            prompt = self.format_prompt(state, additional_context)
            response = self.generate_response(prompt)
            
            # Enhance response with evidence summary
            structured_response = f"""
MEDICAL RESEARCH FINDINGS:

Key Medical Entities: {', '.join(medical_entities)}

Evidence-Based Information:
"""
            
            for condition, evidence in evidence_summary.items():
                structured_response += f"""
{condition.upper()}:
- Definition: {evidence['clinical_info'].get('definition', 'Not available')}
- Clinical Guidelines: {evidence['clinical_info'].get('guidelines', 'Not specified')}
- Evidence Level: {evidence['evidence_assessment']['evidence_level']}
- Key Recommendations: {'; '.join(evidence['key_recommendations'])}
"""
            
            structured_response += f"""
Additional Research Context:
{response}

Evidence Quality Assessment:
- Sources consulted: Medical guidelines and established protocols
- Recommendation strength: Based on available evidence levels
- Clinical relevance: High for identified medical entities
"""
            
            # Update response history
            self.update_response_history(structured_response)
            
            # Extract medical entities and assess safety
            medical_entities_final = self.extract_medical_entities(structured_response)
            safety_flags = self.assess_safety(structured_response)
            
            metadata = {
                "agent_name": self.agent_name,
                "evidence_summary": evidence_summary,
                "medical_entities": medical_entities_final,
                "safety_flags": safety_flags,
                "confidence": 0.7,  # Moderate confidence based on knowledge base
                "evidence_quality": "moderate"
            }
            
            return structured_response, metadata
            
        except Exception as e:
            logger.error(f"Error in researcher processing: {e}")
            fallback_response = f"""
BASIC RESEARCH FINDINGS:

Medical Question: {state.question}
Context: {state.context}

Recommendation: Consult current medical guidelines and evidence-based resources for:
- Diagnosis and treatment protocols
- Monitoring requirements
- Safety considerations
- Patient-specific factors

Error in detailed research: {str(e)}
"""
            metadata = {
                "agent_name": self.agent_name,
                "error": str(e),
                "confidence": 0.2
            }
            return fallback_response, metadata

    def update_knowledge_base(self, new_knowledge: Dict[str, Any]):
        """Update the medical knowledge base with new information."""
        for condition, info in new_knowledge.items():
            if condition in self.knowledge_base:
                self.knowledge_base[condition].update(info)
            else:
                self.knowledge_base[condition] = info

    def get_evidence_citations(self, topic: str) -> List[str]:
        """Generate mock evidence citations for a given topic."""
        citations = [
            f"Clinical Practice Guidelines for {topic} - Professional Medical Association",
            f"Systematic Review: {topic} Management - Cochrane Database",
            f"Evidence-Based Treatment of {topic} - Medical Journal",
            f"Clinical Decision Support for {topic} - UpToDate"
        ]
        return citations 