#!/usr/bin/env python3
"""
BioMCP Integration Test Script
Tests the BioMCP functionality with the medical researcher agent.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.agents.researcher_agent import MedicalResearcherAgent
from src.agents.base_agent import MedicalState
from config.model_config import MEDICAL_AGENT_CONFIGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_biomcp_clinical_trials():
    """Test BioMCP clinical trials search."""
    print("\n🔬 Testing BioMCP Clinical Trials Search")
    print("-" * 50)
    
    # Initialize researcher agent
    agent = MedicalResearcherAgent(MEDICAL_AGENT_CONFIGS["researcher"])
    
    # Test clinical trials search
    try:
        results = await agent.search_clinical_trials(
            condition="breast cancer",
            intervention="immunotherapy",
            phase="PHASE3"
        )
        
        print(f"✅ Clinical Trials Search Results:")
        print(f"   Source: {results.get('source', 'unknown')}")
        print(f"   Count: {results.get('count', 0)}")
        print(f"   Query: {results.get('query', 'N/A')}")
        
        if results.get('trials'):
            print(f"   Sample Trial: {results['trials'][0].get('title', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Clinical Trials Search Failed: {e}")
        return False

async def test_biomcp_literature():
    """Test BioMCP literature search."""
    print("\n📚 Testing BioMCP Literature Search")
    print("-" * 50)
    
    # Initialize researcher agent
    agent = MedicalResearcherAgent(MEDICAL_AGENT_CONFIGS["researcher"])
    
    # Test literature search
    try:
        results = await agent.search_literature(
            query="BRCA1 mutations breast cancer",
            gene="BRCA1",
            disease="breast cancer"
        )
        
        print(f"✅ Literature Search Results:")
        print(f"   Source: {results.get('source', 'unknown')}")
        print(f"   Count: {results.get('count', 0)}")
        print(f"   Query: {results.get('query', 'N/A')}")
        
        if results.get('articles'):
            print(f"   Sample Article: {results['articles'][0].get('title', 'N/A')[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Literature Search Failed: {e}")
        return False

async def test_biomcp_variants():
    """Test BioMCP genetic variants search."""
    print("\n🧬 Testing BioMCP Genetic Variants Search")
    print("-" * 50)
    
    # Initialize researcher agent
    agent = MedicalResearcherAgent(MEDICAL_AGENT_CONFIGS["researcher"])
    
    # Test variants search
    try:
        results = await agent.search_variants(
            gene="BRCA1",
            significance="pathogenic"
        )
        
        print(f"✅ Genetic Variants Search Results:")
        print(f"   Source: {results.get('source', 'unknown')}")
        print(f"   Count: {results.get('count', 0)}")
        print(f"   Query: {results.get('query', 'N/A')}")
        
        if results.get('variants'):
            print(f"   Sample Variant: {results['variants'][0].get('id', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Genetic Variants Search Failed: {e}")
        return False

async def test_full_research_workflow():
    """Test the full research workflow with BioMCP integration."""
    print("\n🏥 Testing Full Research Workflow")
    print("-" * 50)
    
    # Initialize researcher agent
    agent = MedicalResearcherAgent(MEDICAL_AGENT_CONFIGS["researcher"])
    
    # Create a medical state
    state = MedicalState(
        question="What are the latest treatments for BRCA1-positive breast cancer?",
        context="A 45-year-old woman with a confirmed BRCA1 pathogenic mutation has been diagnosed with triple-negative breast cancer. She is seeking information about the most current treatment options.",
        agent_responses={},
        medical_entities=[],
        safety_flags=[],
        confidence_scores={},
        workflow_rounds=0
    )
    
    try:
        # Process the medical question
        response, metadata = agent.process_state(state)
        
        print(f"✅ Full Research Workflow Results:")
        print(f"   Response Length: {len(response)} characters")
        print(f"   BioMCP Available: {metadata.get('biomcp_available', False)}")
        print(f"   Evidence Level: {metadata.get('evidence_level', 'N/A')}")
        print(f"   Research Quality: {metadata.get('research_quality', 'N/A')}")
        print(f"   Confidence: {metadata.get('confidence', 0):.2f}")
        
        print(f"\n📋 Sample Response (first 300 chars):")
        print(f"   {response[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Full Research Workflow Failed: {e}")
        return False

async def main():
    """Main test function."""
    print("🧪 BioMCP Integration Test Suite")
    print("=" * 60)
    
    # Test BioMCP availability
    print("\n🔍 Checking BioMCP Installation")
    print("-" * 50)
    
    import subprocess
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ UV Package Manager: {result.stdout.strip()}")
        else:
            print("❌ UV Package Manager not found")
    except Exception as e:
        print(f"❌ UV Package Manager check failed: {e}")
    
    # Run individual tests
    tests = [
        ("Clinical Trials Search", test_biomcp_clinical_trials()),
        ("Literature Search", test_biomcp_literature()),
        ("Genetic Variants Search", test_biomcp_variants()),
        ("Full Research Workflow", test_full_research_workflow())
    ]
    
    results = []
    for test_name, test_coro in tests:
        try:
            success = await test_coro
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! BioMCP integration is working correctly.")
    elif passed > 0:
        print("⚠️  Some tests passed. BioMCP may be partially working.")
        print("💡 Consider checking BioMCP installation: pip install biomcp-python")
    else:
        print("🚨 All tests failed. BioMCP integration needs attention.")
        print("💡 Installation help: https://biomcp.org/")
    
    print("\n📋 Next Steps:")
    print("  1. If BioMCP tests fail, install with: pip install biomcp-python")
    print("  2. Ensure 'uv' is installed: pip install uv")
    print("  3. Run the MedXpert evaluation: python test_medxpert_evaluation.py")

if __name__ == "__main__":
    asyncio.run(main()) 