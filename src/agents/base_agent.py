"""
Base agent class for the multi-agent medical pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class MedicalState:
    """Shared state between all medical agents using LangGraph."""
    question: str
    context: str
    agent_responses: Dict[str, List[str]]
    current_round: int
    max_rounds: int
    final_answer: Optional[str] = None
    confidence_scores: Dict[str, float] = None
    medical_entities: List[str] = None
    safety_flags: List[str] = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.medical_entities is None:
            self.medical_entities = []
        if self.safety_flags is None:
            self.safety_flags = []

class BaseMedicalAgent(ABC):
    """Base class for all medical agents in the pipeline."""
    
    def __init__(self, agent_name: str, model_config: dict):
        self.agent_name = agent_name
        self.model_config = model_config
        self.tokenizer = None
        self.model = None
        self.device = model_config.get("device_map", "cuda:0")
        self.response_history = []
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model and tokenizer."""
        try:
            model_name = self.model_config["model_name"]
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.model_config.get("trust_remote_code", True)
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            if "bert" in model_name.lower():
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=self.model_config.get("trust_remote_code", True),
                    load_in_8bit=self.model_config.get("load_in_8bit", True),
                    device_map=self.device
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=self.model_config.get("trust_remote_code", True),
                    load_in_8bit=self.model_config.get("load_in_8bit", True),
                    device_map=self.device
                )
            
            self.model.eval()
            logger.info(f"Successfully loaded model for {self.agent_name}")
            
        except Exception as e:
            logger.error(f"Error loading model for {self.agent_name}: {e}")
            raise

    def generate_response(self, prompt: str, max_length: int = None) -> str:
        """Generate response using the agent's model."""
        if max_length is None:
            max_length = self.model_config.get("max_length", 512)
            
        try:
            inputs = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=self.model_config.get("temperature", 0.7),
                    top_p=self.model_config.get("top_p", 0.9),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response for {self.agent_name}: {e}")
            return f"Error: Unable to generate response - {str(e)}"

    def update_response_history(self, response: str):
        """Update the agent's response history."""
        self.response_history.append({
            "response": response,
            "timestamp": torch.cuda.Event(enable_timing=True)
        })

    @abstractmethod
    def process_state(self, state: MedicalState) -> Tuple[str, Dict[str, Any]]:
        """
        Process the current state and return response and metadata.
        
        Args:
            state: Current medical state
            
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        pass

    @abstractmethod
    def get_role_description(self) -> str:
        """Return a description of this agent's role."""
        pass

    def format_prompt(self, state: MedicalState, additional_context: str = "") -> str:
        """Format the prompt for this agent based on current state."""
        role_desc = self.get_role_description()
        
        # Build conversation history
        conversation_history = ""
        for agent_name, responses in state.agent_responses.items():
            if responses:  # Only include agents that have responded
                conversation_history += f"\n{agent_name.title()}: {responses[-1]}\n"
        
        prompt = f"""
{role_desc}

Medical Question: {state.question}

Context: {state.context}

Previous Discussion:
{conversation_history}

{additional_context}

Your Response as {self.agent_name.title()}:
"""
        return prompt.strip()

    def extract_medical_entities(self, text: str) -> List[str]:
        """Extract medical entities from text (simple keyword-based approach)."""
        medical_keywords = [
            "diagnosis", "treatment", "medication", "symptom", "disease",
            "condition", "therapy", "drug", "patient", "clinical", "medical",
            "hospital", "doctor", "nurse", "surgery", "procedure", "test"
        ]
        
        entities = []
        text_lower = text.lower()
        for keyword in medical_keywords:
            if keyword in text_lower:
                entities.append(keyword)
        
        return list(set(entities))

    def assess_safety(self, text: str) -> List[str]:
        """Basic safety assessment for medical content."""
        safety_flags = []
        
        dangerous_phrases = [
            "definitely", "certainly will", "guaranteed to cure",
            "no side effects", "completely safe", "always works",
            "never fails", "100% effective"
        ]
        
        text_lower = text.lower()
        for phrase in dangerous_phrases:
            if phrase in text_lower:
                safety_flags.append(f"Overconfident claim: {phrase}")
        
        return safety_flags 