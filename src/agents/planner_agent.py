"""
Medical Planner Agent - Creates strategies for approaching medical questions.
"""

from typing import Dict, List, Any, Tuple
import logging
from .base_agent import BaseMedicalAgent, MedicalState

logger = logging.getLogger(__name__)

class MedicalPlannerAgent(BaseMedicalAgent):
    """Agent responsible for planning the approach to medical questions."""
    
    def __init__(self, model_config: dict):
        super().__init__("planner", model_config)
        self.planning_templates = {
            "diagnosis": [
                "1. Analyze symptoms and patient history",
                "2. Consider differential diagnoses",
                "3. Identify required diagnostic tests",
                "4. Assess risk factors and contraindications"
            ],
            "treatment": [
                "1. Confirm diagnosis accuracy",
                "2. Review treatment guidelines",
                "3. Consider patient-specific factors",
                "4. Plan monitoring and follow-up"
            ],
            "medication": [
                "1. Verify indication and dosing",
                "2. Check for contraindications",
                "3. Review drug interactions",
                "4. Consider side effects and monitoring"
            ],
            "general": [
                "1. Understand the medical question",
                "2. Identify key medical concepts",
                "3. Gather relevant information",
                "4. Synthesize findings systematically"
            ]
        }
    
    def get_role_description(self) -> str:
        """Return description of planner role."""
        return """You are a Medical Planner Agent. Your role is to:
1. Analyze the medical question systematically
2. Create a structured approach for answering
3. Identify key areas that need investigation
4. Set priorities for the collaborative discussion
5. Ensure the team follows evidence-based practices

Be methodical, comprehensive, and focus on creating a clear roadmap for the medical team."""

    def identify_question_type(self, question: str) -> str:
        """Identify the type of medical question to apply appropriate template."""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["diagnose", "diagnosis", "what condition"]):
            return "diagnosis"
        elif any(word in question_lower for word in ["treat", "treatment", "therapy", "manage"]):
            return "treatment"
        elif any(word in question_lower for word in ["medication", "drug", "prescribe", "dose"]):
            return "medication"
        else:
            return "general"

    def create_investigation_plan(self, state: MedicalState) -> Dict[str, Any]:
        """Create a structured plan for investigating the medical question."""
        question_type = self.identify_question_type(state.question)
        base_steps = self.planning_templates.get(question_type, self.planning_templates["general"])
        
        plan = {
            "question_type": question_type,
            "investigation_steps": base_steps,
            "key_concepts": self.extract_medical_entities(state.question + " " + state.context),
            "priority_areas": [],
            "collaboration_strategy": {
                "researcher_focus": "Evidence and guidelines",
                "analyst_focus": "Clinical reasoning and analysis",
                "reporter_focus": "Clear synthesis and recommendations"
            }
        }
        
        # Identify priority areas based on question content
        if "urgent" in state.question.lower() or "emergency" in state.question.lower():
            plan["priority_areas"].append("Time-sensitive assessment")
        if "pediatric" in state.question.lower() or "child" in state.question.lower():
            plan["priority_areas"].append("Pediatric considerations")
        if "elderly" in state.question.lower() or "geriatric" in state.question.lower():
            plan["priority_areas"].append("Geriatric considerations")
        
        return plan

    def process_state(self, state: MedicalState) -> Tuple[str, Dict[str, Any]]:
        """Process the current state and create a medical investigation plan."""
        try:
            # Create investigation plan
            investigation_plan = self.create_investigation_plan(state)
            
            # Generate planning prompt
            additional_context = f"""
Question Type Identified: {investigation_plan['question_type']}
Key Medical Concepts: {', '.join(investigation_plan['key_concepts'])}
Priority Areas: {', '.join(investigation_plan['priority_areas']) if investigation_plan['priority_areas'] else 'Standard approach'}

Create a systematic plan for approaching this medical question.
"""
            
            prompt = self.format_prompt(state, additional_context)
            response = self.generate_response(prompt)
            
            # Enhance response with structured plan
            structured_response = f"""
MEDICAL INVESTIGATION PLAN:

Question Analysis: {investigation_plan['question_type']} type question
Key Concepts: {', '.join(investigation_plan['key_concepts'])}

Recommended Approach:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(investigation_plan['investigation_steps']))}

Team Collaboration Strategy:
- Researcher: {investigation_plan['collaboration_strategy']['researcher_focus']}
- Analyst: {investigation_plan['collaboration_strategy']['analyst_focus']}
- Reporter: {investigation_plan['collaboration_strategy']['reporter_focus']}

Additional Considerations:
{response}

Priority Areas: {', '.join(investigation_plan['priority_areas']) if investigation_plan['priority_areas'] else 'Standard medical approach applies'}
"""
            
            # Update response history
            self.update_response_history(structured_response)
            
            # Extract medical entities and assess safety
            medical_entities = self.extract_medical_entities(structured_response)
            safety_flags = self.assess_safety(structured_response)
            
            metadata = {
                "agent_name": self.agent_name,
                "investigation_plan": investigation_plan,
                "medical_entities": medical_entities,
                "safety_flags": safety_flags,
                "confidence": 0.8  # Planner typically has high confidence in planning
            }
            
            return structured_response, metadata
            
        except Exception as e:
            logger.error(f"Error in planner processing: {e}")
            fallback_response = f"""
BASIC MEDICAL APPROACH:
1. Understand the question: {state.question}
2. Gather relevant medical information
3. Analyze findings systematically
4. Provide evidence-based recommendations

Error occurred in detailed planning: {str(e)}
"""
            metadata = {
                "agent_name": self.agent_name,
                "error": str(e),
                "confidence": 0.3
            }
            return fallback_response, metadata

    def evaluate_plan_quality(self, plan: Dict[str, Any]) -> float:
        """Evaluate the quality of the created plan."""
        score = 0.0
        
        # Check if plan has all required components
        required_components = ["question_type", "investigation_steps", "key_concepts", "collaboration_strategy"]
        for component in required_components:
            if component in plan:
                score += 0.2
        
        # Bonus for identifying priority areas
        if plan.get("priority_areas"):
            score += 0.1
        
        # Bonus for medical entity extraction
        if len(plan.get("key_concepts", [])) > 0:
            score += 0.1
        
        return min(score, 1.0) 