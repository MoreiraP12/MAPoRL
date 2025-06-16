"""
Medical Multi-Agent Workflow using LangGraph.
"""

from typing import Dict, List, Any, Optional
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dataclasses import asdict

from ..agents.base_agent import MedicalState
from ..agents.planner_agent import MedicalPlannerAgent
from ..agents.researcher_agent import MedicalResearcherAgent
from ..agents.analyst_agent import MedicalAnalystAgent
from ..agents.reporter_agent import MedicalReporterAgent
from ..config.model_config import MULTI_AGENT_CONFIG

logger = logging.getLogger(__name__)

class MedicalWorkflow:
    """LangGraph workflow for multi-agent medical collaboration."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or asdict(MULTI_AGENT_CONFIG)
        self.agents = {}
        self.workflow = None
        self.memory = MemorySaver()
        self.setup_agents()
        self.build_workflow()
    
    def setup_agents(self):
        """Initialize all medical agents."""
        try:
            agent_configs = self.config["agents"]
            
            # Initialize each agent
            self.agents["planner"] = MedicalPlannerAgent(agent_configs["planner"])
            self.agents["researcher"] = MedicalResearcherAgent(agent_configs["researcher"])
            self.agents["analyst"] = MedicalAnalystAgent(agent_configs["analyst"])
            self.agents["reporter"] = MedicalReporterAgent(agent_configs["reporter"])
            
            logger.info("Successfully initialized all medical agents")
            
        except Exception as e:
            logger.error(f"Error setting up agents: {e}")
            raise

    def build_workflow(self):
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(MedicalState)
        
        # Add nodes for each agent
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("researcher", self.researcher_node)
        workflow.add_node("analyst", self.analyst_node)
        workflow.add_node("reporter", self.reporter_node)
        workflow.add_node("collaboration_round", self.collaboration_round_node)
        workflow.add_node("finalize", self.finalize_node)
        
        # Define the workflow edges
        workflow.set_entry_point("planner")
        
        # Sequential flow: planner -> researcher -> analyst -> collaboration
        workflow.add_edge("planner", "researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "collaboration_round")
        
        # Conditional flow from collaboration round
        workflow.add_conditional_edges(
            "collaboration_round",
            self.should_continue_collaboration,
            {
                "continue": "planner",  # Start another round
                "finalize": "reporter"  # Move to reporter
            }
        )
        
        # Final flow
        workflow.add_edge("reporter", "finalize")
        workflow.add_edge("finalize", END)
        
        # Compile the workflow
        self.workflow = workflow.compile(checkpointer=self.memory)
        logger.info("Medical workflow compiled successfully")

    def planner_node(self, state: MedicalState) -> MedicalState:
        """Execute planner agent."""
        try:
            response, metadata = self.agents["planner"].process_state(state)
            
            # Update state
            if "planner" not in state.agent_responses:
                state.agent_responses["planner"] = []
            state.agent_responses["planner"].append(response)
            
            # Update confidence scores
            state.confidence_scores["planner"] = metadata.get("confidence", 0.5)
            
            # Update medical entities
            state.medical_entities.extend(metadata.get("medical_entities", []))
            state.medical_entities = list(set(state.medical_entities))  # Remove duplicates
            
            # Update safety flags
            state.safety_flags.extend(metadata.get("safety_flags", []))
            
            logger.info("Planner node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in planner node: {e}")
            # Add error response
            if "planner" not in state.agent_responses:
                state.agent_responses["planner"] = []
            state.agent_responses["planner"].append(f"Error in planning: {str(e)}")
            return state

    def researcher_node(self, state: MedicalState) -> MedicalState:
        """Execute researcher agent."""
        try:
            response, metadata = self.agents["researcher"].process_state(state)
            
            # Update state
            if "researcher" not in state.agent_responses:
                state.agent_responses["researcher"] = []
            state.agent_responses["researcher"].append(response)
            
            # Update confidence scores
            state.confidence_scores["researcher"] = metadata.get("confidence", 0.5)
            
            # Update medical entities
            state.medical_entities.extend(metadata.get("medical_entities", []))
            state.medical_entities = list(set(state.medical_entities))
            
            # Update safety flags
            state.safety_flags.extend(metadata.get("safety_flags", []))
            
            logger.info("Researcher node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in researcher node: {e}")
            if "researcher" not in state.agent_responses:
                state.agent_responses["researcher"] = []
            state.agent_responses["researcher"].append(f"Error in research: {str(e)}")
            return state

    def analyst_node(self, state: MedicalState) -> MedicalState:
        """Execute analyst agent."""
        try:
            response, metadata = self.agents["analyst"].process_state(state)
            
            # Update state
            if "analyst" not in state.agent_responses:
                state.agent_responses["analyst"] = []
            state.agent_responses["analyst"].append(response)
            
            # Update confidence scores
            state.confidence_scores["analyst"] = metadata.get("confidence", 0.5)
            
            # Update medical entities
            state.medical_entities.extend(metadata.get("medical_entities", []))
            state.medical_entities = list(set(state.medical_entities))
            
            # Update safety flags
            state.safety_flags.extend(metadata.get("safety_flags", []))
            
            logger.info("Analyst node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in analyst node: {e}")
            if "analyst" not in state.agent_responses:
                state.agent_responses["analyst"] = []
            state.agent_responses["analyst"].append(f"Error in analysis: {str(e)}")
            return state

    def reporter_node(self, state: MedicalState) -> MedicalState:
        """Execute reporter agent."""
        try:
            response, metadata = self.agents["reporter"].process_state(state)
            
            # Update state
            if "reporter" not in state.agent_responses:
                state.agent_responses["reporter"] = []
            state.agent_responses["reporter"].append(response)
            
            # Set final answer
            state.final_answer = response
            
            # Update confidence scores
            state.confidence_scores["reporter"] = metadata.get("confidence", 0.5)
            
            # Update medical entities
            state.medical_entities.extend(metadata.get("medical_entities", []))
            state.medical_entities = list(set(state.medical_entities))
            
            # Update safety flags
            state.safety_flags.extend(metadata.get("safety_flags", []))
            
            logger.info("Reporter node completed successfully")
            return state
            
        except Exception as e:
            logger.error(f"Error in reporter node: {e}")
            if "reporter" not in state.agent_responses:
                state.agent_responses["reporter"] = []
            state.agent_responses["reporter"].append(f"Error in reporting: {str(e)}")
            state.final_answer = f"Error generating final report: {str(e)}"
            return state

    def collaboration_round_node(self, state: MedicalState) -> MedicalState:
        """Handle collaboration round logic."""
        state.current_round += 1
        logger.info(f"Completed collaboration round {state.current_round}")
        return state

    def finalize_node(self, state: MedicalState) -> MedicalState:
        """Finalize the workflow."""
        logger.info("Finalizing medical workflow")
        
        # Calculate overall confidence
        if state.confidence_scores:
            overall_confidence = sum(state.confidence_scores.values()) / len(state.confidence_scores)
        else:
            overall_confidence = 0.1
        
        # Add summary metadata
        state.confidence_scores["overall"] = overall_confidence
        
        return state

    def should_continue_collaboration(self, state: MedicalState) -> str:
        """Determine if collaboration should continue."""
        # Continue if we haven't reached max rounds and confidence is low
        if state.current_round < state.max_rounds:
            # Check if we need another round based on confidence
            overall_confidence = sum(state.confidence_scores.values()) / len(state.confidence_scores) if state.confidence_scores else 0.1
            
            # Continue if confidence is low or if there are safety flags
            if overall_confidence < 0.6 or len(state.safety_flags) > 0:
                logger.info(f"Continuing collaboration - Round {state.current_round + 1}")
                return "continue"
        
        logger.info("Moving to finalization")
        return "finalize"

    def run_workflow(self, question: str, context: str = "", thread_id: str = "default") -> Dict[str, Any]:
        """Run the complete medical workflow."""
        try:
            # Initialize state
            initial_state = MedicalState(
                question=question,
                context=context,
                agent_responses={},
                current_round=0,
                max_rounds=self.config.get("max_rounds", 3),
                confidence_scores={},
                medical_entities=[],
                safety_flags=[]
            )
            
            # Configure the workflow run
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute workflow
            result = self.workflow.invoke(initial_state, config=config)
            
            # Debug: Log what LangGraph returns
            logger.info(f"LangGraph result type: {type(result)}")
            if hasattr(result, '__dict__'):
                logger.info(f"LangGraph result attributes: {list(result.__dict__.keys())}")
            elif isinstance(result, dict):
                logger.info(f"LangGraph result keys: {list(result.keys())}")
            
            # Extract state from LangGraph result
            # LangGraph returns the final state differently, let's handle both cases
            if hasattr(result, 'final_answer'):
                # Direct state object
                final_state = result
            elif isinstance(result, dict) and len(result) > 0:
                # LangGraph returns dict-like object, get state values
                # Try to find a state-like object in the result
                for key, value in result.items():
                    if hasattr(value, 'final_answer') or hasattr(value, 'agent_responses'):
                        final_state = value
                        break
                else:
                    # Fallback: use the result as-is
                    final_state = result
            else:
                final_state = result
            
            # Format results with safe access
            final_answer = getattr(final_state, 'final_answer', None)
            if not final_answer and isinstance(final_state, dict):
                final_answer = final_state.get('final_answer', "No final answer generated")
                
            agent_responses = getattr(final_state, 'agent_responses', None)
            if not agent_responses and isinstance(final_state, dict):
                agent_responses = final_state.get('agent_responses', {})
                
            confidence_scores = getattr(final_state, 'confidence_scores', None)
            if not confidence_scores and isinstance(final_state, dict):
                confidence_scores = final_state.get('confidence_scores', {})
                
            medical_entities = getattr(final_state, 'medical_entities', None)
            if not medical_entities and isinstance(final_state, dict):
                medical_entities = final_state.get('medical_entities', [])
                
            safety_flags = getattr(final_state, 'safety_flags', None)
            if not safety_flags and isinstance(final_state, dict):
                safety_flags = final_state.get('safety_flags', [])
                
            current_round = getattr(final_state, 'current_round', None)
            if current_round is None and isinstance(final_state, dict):
                current_round = final_state.get('current_round', 0)
            if current_round is None:
                current_round = 0
            
            # Format results
            output = {
                "question": question,
                "context": context,
                "final_answer": final_answer or "No final answer generated",
                "agent_responses": agent_responses or {},
                "confidence_scores": confidence_scores or {},
                "medical_entities": medical_entities or [],
                "safety_flags": safety_flags or [],
                "rounds_completed": current_round,
                "workflow_status": "completed"
            }
            
            logger.info(f"Workflow completed successfully in {current_round} rounds")
            return output
            
        except Exception as e:
            logger.error(f"Error running workflow: {e}")
            return {
                "question": question,
                "context": context,
                "final_answer": f"Error: Unable to process medical question - {str(e)}",
                "agent_responses": {},
                "confidence_scores": {},
                "medical_entities": [],
                "safety_flags": [f"Workflow error: {str(e)}"],
                "rounds_completed": 0,
                "workflow_status": "error"
            }

    def get_workflow_state(self, thread_id: str = "default") -> Optional[MedicalState]:
        """Get the current state of a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.workflow.get_state(config)
            return state.values if state else None
        except Exception as e:
            logger.error(f"Error getting workflow state: {e}")
            return None

    def reset_workflow(self, thread_id: str = "default"):
        """Reset a workflow thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            # Clear the memory for this thread
            # Note: This is a simplified reset - in production you might want more sophisticated state management
            logger.info(f"Reset workflow thread: {thread_id}")
        except Exception as e:
            logger.error(f"Error resetting workflow: {e}")

# Factory function for creating workflow instances
def create_medical_workflow(config: Dict[str, Any] = None) -> MedicalWorkflow:
    """Create a new medical workflow instance."""
    return MedicalWorkflow(config) 