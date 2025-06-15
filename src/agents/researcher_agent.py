"""
Medical Researcher Agent with BioMCP Integration - Enhanced with real biomedical data access.
"""

import logging
import re
import asyncio
from typing import Dict, List, Any, Tuple, Optional
from .base_agent import BaseMedicalAgent, MedicalState

# BioMCP integration
try:
    import subprocess
    import json
    import tempfile
    import os
    BIOMCP_AVAILABLE = True
except ImportError:
    BIOMCP_AVAILABLE = False

logger = logging.getLogger(__name__)

class MedicalResearcherAgent(BaseMedicalAgent):
    """Enhanced researcher agent with BioMCP integration for real biomedical data access."""
    
    def __init__(self, model_config: dict):
        super().__init__("researcher", model_config)
        self.biomcp_available = BIOMCP_AVAILABLE
        self.research_frameworks = {
            "evidence_based_medicine": [
                "Clinical guidelines",
                "Systematic reviews",
                "Randomized controlled trials",
                "Meta-analyses"
            ],
            "clinical_research": [
                "Phase I-IV trials",
                "Observational studies",
                "Case-control studies",
                "Cohort studies"
            ],
            "genomic_research": [
                "Variant analysis",
                "Pharmacogenomics",
                "Disease associations",
                "Functional studies"
            ]
        }
        
        # Initialize BioMCP if available
        if self.biomcp_available:
            self._setup_biomcp()
        
        self.medical_databases = [
            "PubMed", "ClinicalTrials.gov", "Cochrane Library",
            "MyVariant.info", "ClinVar", "OMIM"
        ]
        
        self.evidence_levels = {
            "systematic_review": 5,
            "rct": 4,
            "cohort_study": 3,
            "case_control": 2,
            "case_series": 1
        }
    
    def _setup_biomcp(self):
        """Setup BioMCP connection and verify availability."""
        try:
            # Test BioMCP availability
            result = subprocess.run(
                ["uv", "run", "--with", "biomcp-python", "biomcp", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("✅ BioMCP successfully configured")
                self.biomcp_ready = True
            else:
                logger.warning("⚠️ BioMCP not properly configured")
                self.biomcp_ready = False
        except Exception as e:
            logger.warning(f"⚠️ BioMCP setup failed: {e}")
            self.biomcp_ready = False
    
    def get_role_description(self) -> str:
        """Return description of researcher role with BioMCP capabilities."""
        return """You are a Medical Researcher Agent with access to real-time biomedical data through BioMCP. Your role is to:
1. Search clinical trials and research literature
2. Gather evidence-based medical information
3. Access genomic variant data and clinical guidelines
4. Evaluate research quality and evidence levels
5. Provide comprehensive background research for medical questions

You have access to:
- ClinicalTrials.gov for trial data
- PubMed/PubTator3 for literature search
- MyVariant.info for genomic data
- Real-time biomedical database access through BioMCP

Focus on evidence-based medicine and current clinical guidelines."""
    
    async def search_clinical_trials(self, condition: str, intervention: str = None, 
                                   phase: str = None, location: str = None) -> Dict[str, Any]:
        """Search clinical trials using BioMCP."""
        if not getattr(self, 'biomcp_ready', False):
            return {"trials": [], "source": "fallback", "message": "BioMCP not available"}
        
        try:
            cmd = ["uv", "run", "--with", "biomcp-python", "biomcp", "trial", "search"]
            cmd.extend(["--condition", condition])
            
            if intervention:
                cmd.extend(["--intervention", intervention])
            if phase:
                cmd.extend(["--phase", phase])
            if location:
                cmd.extend(["--location", location])
            
            cmd.extend(["--format", "json", "--limit", "10"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                trials_data = json.loads(result.stdout)
                logger.info(f"🔬 Found {len(trials_data.get('results', []))} clinical trials")
                return {
                    "trials": trials_data.get('results', []),
                    "source": "ClinicalTrials.gov",
                    "query": condition,
                    "count": len(trials_data.get('results', []))
                }
            else:
                logger.warning(f"Clinical trial search failed: {result.stderr}")
                return {"trials": [], "source": "error", "message": result.stderr}
                
        except Exception as e:
            logger.error(f"Error searching clinical trials: {e}")
            return {"trials": [], "source": "error", "message": str(e)}
    
    async def search_literature(self, query: str, gene: str = None, 
                              disease: str = None, limit: int = 10) -> Dict[str, Any]:
        """Search biomedical literature using BioMCP."""
        if not getattr(self, 'biomcp_ready', False):
            return {"articles": [], "source": "fallback", "message": "BioMCP not available"}
        
        try:
            cmd = ["uv", "run", "--with", "biomcp-python", "biomcp", "article", "search"]
            cmd.extend(["--query", query])
            
            if gene:
                cmd.extend(["--gene", gene])
            if disease:
                cmd.extend(["--disease", disease])
            
            cmd.extend(["--format", "json", "--limit", str(limit)])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                articles_data = json.loads(result.stdout)
                logger.info(f"📚 Found {len(articles_data.get('results', []))} articles")
                return {
                    "articles": articles_data.get('results', []),
                    "source": "PubMed",
                    "query": query,
                    "count": len(articles_data.get('results', []))
                }
            else:
                logger.warning(f"Literature search failed: {result.stderr}")
                return {"articles": [], "source": "error", "message": result.stderr}
                
        except Exception as e:
            logger.error(f"Error searching literature: {e}")
            return {"articles": [], "source": "error", "message": str(e)}
    
    async def search_variants(self, gene: str = None, significance: str = None,
                            variant_type: str = None) -> Dict[str, Any]:
        """Search genetic variants using BioMCP."""
        if not getattr(self, 'biomcp_ready', False):
            return {"variants": [], "source": "fallback", "message": "BioMCP not available"}
        
        try:
            cmd = ["uv", "run", "--with", "biomcp-python", "biomcp", "variant", "search"]
            
            if gene:
                cmd.extend(["--gene", gene])
            if significance:
                cmd.extend(["--significance", significance])
            if variant_type:
                cmd.extend(["--type", variant_type])
            
            cmd.extend(["--format", "json", "--limit", "10"])
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                variants_data = json.loads(result.stdout)
                logger.info(f"🧬 Found {len(variants_data.get('results', []))} variants")
                return {
                    "variants": variants_data.get('results', []),
                    "source": "MyVariant.info",
                    "query": gene or "general",
                    "count": len(variants_data.get('results', []))
                }
            else:
                logger.warning(f"Variant search failed: {result.stderr}")
                return {"variants": [], "source": "error", "message": result.stderr}
                
        except Exception as e:
            logger.error(f"Error searching variants: {e}")
            return {"variants": [], "source": "error", "message": str(e)}
    
    def extract_research_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities relevant to research from text."""
        entities = {
            "conditions": [],
            "interventions": [],
            "genes": [],
            "drugs": [],
            "study_types": []
        }
        
        text_lower = text.lower()
        
        # Extract conditions (simple pattern matching)
        condition_patterns = [
            r'\b(?:cancer|carcinoma|tumor|malignancy)\b',
            r'\b(?:diabetes|hypertension|depression|asthma)\b',
            r'\b(?:covid|pneumonia|influenza|infection)\b',
            r'\b(?:alzheimer|parkinson|dementia)\b'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text_lower)
            entities["conditions"].extend(matches)
        
        # Extract gene names (simplified)
        gene_pattern = r'\b[A-Z]{2,}[0-9]*\b'
        potential_genes = re.findall(gene_pattern, text)
        # Filter for likely gene names
        common_genes = ["TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "PIK3CA", "APC", "PTEN"]
        entities["genes"] = [g for g in potential_genes if g in common_genes or len(g) <= 6]
        
        # Extract interventions
        intervention_patterns = [
            r'\b(?:chemotherapy|radiation|surgery|immunotherapy)\b',
            r'\b(?:treatment|therapy|intervention|medication)\b'
        ]
        
        for pattern in intervention_patterns:
            matches = re.findall(pattern, text_lower)
            entities["interventions"].extend(matches)
        
        # Extract study types
        study_patterns = [
            r'\b(?:randomized|controlled|trial|rct)\b',
            r'\b(?:cohort|case-control|observational)\b',
            r'\b(?:meta-analysis|systematic review)\b'
        ]
        
        for pattern in study_patterns:
            matches = re.findall(pattern, text_lower)
            entities["study_types"].extend(matches)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    async def conduct_comprehensive_research(self, question: str, context: str) -> Dict[str, Any]:
        """Conduct comprehensive research using BioMCP tools."""
        research_results = {
            "clinical_trials": {"data": [], "source": "none"},
            "literature": {"data": [], "source": "none"},
            "variants": {"data": [], "source": "none"},
            "research_quality": "unknown",
            "evidence_level": 0
        }
        
        # Extract research entities
        combined_text = f"{question} {context}"
        entities = self.extract_research_entities(combined_text)
        
        # Search clinical trials
        if entities["conditions"]:
            main_condition = entities["conditions"][0] if entities["conditions"] else None
            main_intervention = entities["interventions"][0] if entities["interventions"] else None
            
            if main_condition:
                trials_result = await self.search_clinical_trials(
                    condition=main_condition,
                    intervention=main_intervention
                )
                research_results["clinical_trials"] = trials_result
        
        # Search literature
        literature_query = question[:100]  # Truncate for API
        if entities["genes"]:
            gene = entities["genes"][0]
            disease = entities["conditions"][0] if entities["conditions"] else None
            literature_result = await self.search_literature(
                query=literature_query,
                gene=gene,
                disease=disease
            )
        else:
            literature_result = await self.search_literature(query=literature_query)
        
        research_results["literature"] = literature_result
        
        # Search variants if genetic context
        if entities["genes"]:
            variants_result = await self.search_variants(
                gene=entities["genes"][0],
                significance="pathogenic"
            )
            research_results["variants"] = variants_result
        
        # Assess research quality
        research_results["research_quality"] = self.assess_research_quality(research_results)
        research_results["evidence_level"] = self.calculate_evidence_level(research_results)
        
        return research_results
    
    def assess_research_quality(self, research_results: Dict[str, Any]) -> str:
        """Assess the quality of research results."""
        quality_score = 0
        
        # Check clinical trials availability
        if research_results["clinical_trials"]["data"]:
            quality_score += 2
        
        # Check literature availability
        if research_results["literature"]["data"]:
            quality_score += 2
        
        # Check variant data availability
        if research_results["variants"]["data"]:
            quality_score += 1
        
        # Check for RCTs or systematic reviews
        literature_data = research_results["literature"]["data"]
        if any("randomized" in str(article).lower() or "systematic" in str(article).lower() 
               for article in literature_data):
            quality_score += 2
        
        if quality_score >= 5:
            return "high"
        elif quality_score >= 3:
            return "moderate"
        else:
            return "low"
    
    def calculate_evidence_level(self, research_results: Dict[str, Any]) -> int:
        """Calculate evidence level based on available research."""
        evidence_level = 1  # Base level
        
        # Boost for clinical trials
        if research_results["clinical_trials"]["data"]:
            evidence_level += 2
        
        # Boost for high-quality literature
        literature_data = research_results["literature"]["data"]
        for article in literature_data:
            article_str = str(article).lower()
            if "systematic review" in article_str or "meta-analysis" in article_str:
                evidence_level += 3
                break
            elif "randomized" in article_str:
                evidence_level += 2
                break
        
        return min(evidence_level, 5)  # Cap at 5
    
    def process_state(self, state: MedicalState) -> Tuple[str, Dict[str, Any]]:
        """Process the current state and provide research-enhanced response."""
        try:
            # Conduct comprehensive research
            research_results = asyncio.run(
                self.conduct_comprehensive_research(state.question, state.context)
            )
            
            # Create research context
            research_context = f"""
Research Database Access: {'✅ BioMCP Active' if getattr(self, 'biomcp_ready', False) else '❌ BioMCP Unavailable'}

Clinical Trials Found: {research_results['clinical_trials'].get('count', 0)}
Literature Articles: {research_results['literature'].get('count', 0)}
Genetic Variants: {research_results['variants'].get('count', 0)}

Research Quality: {research_results['research_quality']}
Evidence Level: {research_results['evidence_level']}/5

Focus on evidence-based research and current clinical guidelines.
"""
            
            prompt = self.format_prompt(state, research_context)
            response = self.generate_response(prompt)
            
            # Enhance response with research findings
            enhanced_response = f"""
RESEARCH FINDINGS:

"""
            
            # Add clinical trials information
            if research_results["clinical_trials"]["data"]:
                enhanced_response += f"🔬 Clinical Trials ({research_results['clinical_trials']['source']}):\n"
                for i, trial in enumerate(research_results["clinical_trials"]["data"][:3]):
                    title = trial.get("title", "Unknown Trial")
                    status = trial.get("status", "Unknown")
                    enhanced_response += f"  {i+1}. {title} (Status: {status})\n"
                enhanced_response += "\n"
            
            # Add literature information
            if research_results["literature"]["data"]:
                enhanced_response += f"📚 Recent Literature ({research_results['literature']['source']}):\n"
                for i, article in enumerate(research_results["literature"]["data"][:3]):
                    title = article.get("title", "Unknown Article")
                    journal = article.get("journal", "Unknown Journal")
                    enhanced_response += f"  {i+1}. {title}\n     Journal: {journal}\n"
                enhanced_response += "\n"
            
            # Add variant information
            if research_results["variants"]["data"]:
                enhanced_response += f"🧬 Genetic Variants ({research_results['variants']['source']}):\n"
                for i, variant in enumerate(research_results["variants"]["data"][:2]):
                    variant_id = variant.get("id", "Unknown Variant")
                    significance = variant.get("clinical_significance", "Unknown")
                    enhanced_response += f"  {i+1}. {variant_id} (Significance: {significance})\n"
                enhanced_response += "\n"
            
            enhanced_response += f"""
EVIDENCE-BASED ANALYSIS:
{response}

Research Quality Assessment: {research_results['research_quality'].upper()}
Evidence Level: {research_results['evidence_level']}/5 ({self.get_evidence_description(research_results['evidence_level'])})

Data Sources: {', '.join([
    research_results['clinical_trials']['source'],
    research_results['literature']['source'],
    research_results['variants']['source']
])}
"""
            
            # Update response history
            self.update_response_history(enhanced_response)
            
            # Extract medical entities and assess safety
            medical_entities = self.extract_medical_entities(enhanced_response)
            safety_flags = self.assess_safety(enhanced_response)
            
            metadata = {
                "agent_name": self.agent_name,
                "research_results": research_results,
                "biomcp_available": getattr(self, 'biomcp_ready', False),
                "evidence_level": research_results['evidence_level'],
                "research_quality": research_results['research_quality'],
                "medical_entities": medical_entities,
                "safety_flags": safety_flags,
                "confidence": min(0.3 + (research_results['evidence_level'] * 0.15), 0.9)
            }
            
            return enhanced_response, metadata
            
        except Exception as e:
            logger.error(f"Error in researcher processing: {e}")
            
            # Fallback response
            fallback_response = f"""
RESEARCH ANALYSIS (Fallback Mode):

Question: {state.question}
Context: {state.context}

Research Approach:
1. Systematic literature review needed
2. Clinical trial database search recommended
3. Evidence-based medicine principles applied
4. Current clinical guidelines consultation required

Note: Advanced research tools temporarily unavailable.
Error: {str(e)}
"""
            
            metadata = {
                "agent_name": self.agent_name,
                "error": str(e),
                "biomcp_available": False,
                "confidence": 0.3
            }
            
            return fallback_response, metadata
    
    def get_evidence_description(self, level: int) -> str:
        """Get description for evidence level."""
        descriptions = {
            1: "Expert opinion/Case reports",
            2: "Case-control studies",
            3: "Cohort studies",
            4: "Randomized controlled trials",
            5: "Systematic reviews/Meta-analyses"
        }
        return descriptions.get(level, "Unknown") 