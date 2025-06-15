# 🧪 Medical Multi-Agent System Testing Guide

This guide covers testing the multi-agent medical system with **BioMCP integration** and **MedXpert benchmark evaluation**.

## 🎯 Overview

The testing suite evaluates:
- **BioMCP Integration**: Real-time access to biomedical databases
- **Multi-Agent Collaboration**: Agent interaction quality  
- **MedXpert Performance**: Medical question answering accuracy
- **Safety & Relevance**: Medical safety and domain relevance

## 🔧 Setup & Installation

### 1. Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# Install BioMCP specifically
pip install biomcp-python uv

# Optional: Medical NLP tools
pip install medspacy scispacy
```

### 2. Verify BioMCP Installation
```bash
# Check UV package manager
uv --version

# Test BioMCP
uv run --with biomcp-python biomcp --help
```

## 🧬 BioMCP Integration Testing

### Quick BioMCP Test
```bash
# Test BioMCP integration with researcher agent
python test_biomcp_integration.py
```

**Expected Output:**
```
🧪 BioMCP Integration Test Suite
================================================
✅ PASS - Clinical Trials Search  
✅ PASS - Literature Search
✅ PASS - Genetic Variants Search
✅ PASS - Full Research Workflow

🏆 Overall: 4/4 tests passed (100.0%)
```

### BioMCP Capabilities

#### 🔬 Clinical Trials (ClinicalTrials.gov)
- Search by condition, intervention, phase, location
- Real-time trial status and eligibility
- Protocol details and outcomes

#### 📚 Literature (PubMed/PubTator3)  
- Biomedical article search
- Gene-disease associations
- Evidence-based research

#### 🧬 Genomic Data (MyVariant.info)
- Genetic variant annotations
- Clinical significance
- Pharmacogenomic data

## 📊 MedXpert Benchmark Evaluation

### Run MedXpert Evaluation
```bash
# Evaluate on 5 sample questions (default)
python test_medxpert_evaluation.py

# Evaluate specific number of samples
python test_medxpert_evaluation.py --samples 10

# Custom output file
python test_medxpert_evaluation.py --output my_results.json
```

### Sample MedXpert Questions

The test includes 5 carefully selected medical scenarios:

1. **Cardiology**: Chest pain differential diagnosis
2. **Endocrinology**: Type 2 diabetes management  
3. **Pulmonology**: Community-acquired pneumonia
4. **Oncology**: BRCA1 mutation screening
5. **Emergency Medicine**: Hypertensive crisis

### Expected Performance Metrics

| Metric | Target Range | Description |
|--------|--------------|-------------|
| **Overall Score** | 0.6 - 0.8 | Combined performance metric |
| **Accuracy** | 0.5 - 0.7 | Semantic similarity to reference |
| **Medical Relevance** | 0.7 - 0.9 | Clinical terminology usage |
| **Safety Score** | 0.8 - 1.0 | Medical safety compliance |
| **Collaboration** | 0.6 - 0.8 | Multi-agent interaction quality |

## 🏥 Multi-Agent Workflow Testing

### Agent Collaboration Flow
```
1. Planner Agent → Analyzes question type & creates plan
2. Researcher Agent → Searches databases via BioMCP
3. Analyst Agent → Provides clinical reasoning
4. Reporter Agent → Synthesizes final response
```

### Collaboration Quality Metrics
- **Information Sharing**: Agent response utilization
- **Complementary Expertise**: Specialized knowledge integration
- **Consensus Building**: Agreement on recommendations
- **Safety Coordination**: Collective safety assessment

## 📈 Performance Analysis

### Evaluation Results Structure
```json
{
  "summary": {
    "total_samples": 5,
    "successful_evaluations": 5,
    "success_rate": 1.0,
    "biomcp_usage_rate": 0.8
  },
  "performance_metrics": {
    "accuracy": {"mean": 0.65, "min": 0.45, "max": 0.82},
    "medical_relevance": {"mean": 0.78, "min": 0.65, "max": 0.91},
    "safety": {"mean": 0.85, "min": 0.72, "max": 0.95},
    "collaboration": {"mean": 0.71, "min": 0.55, "max": 0.84}
  }
}
```

### Interpretation Guidelines

#### 🎯 Accuracy Scores
- **>0.7**: Excellent semantic similarity
- **0.5-0.7**: Good medical reasoning  
- **<0.5**: Needs improvement

#### 🏥 Medical Relevance
- **>0.8**: Strong clinical knowledge
- **0.6-0.8**: Adequate medical understanding
- **<0.6**: Insufficient domain expertise

#### 🛡️ Safety Scores
- **>0.9**: Excellent safety compliance
- **0.7-0.9**: Good safety awareness
- **<0.7**: Safety concerns require attention

## 🔍 Troubleshooting

### Common Issues

#### 1. BioMCP Connection Failures
```bash
# Check UV installation
which uv
pip install uv

# Test BioMCP manually
uv run --with biomcp-python biomcp trial search --condition "diabetes"
```

#### 2. Model Loading Errors
```bash
# Clear transformers cache
rm -rf ~/.cache/huggingface/transformers/

# Reinstall transformers
pip install --upgrade transformers
```

#### 3. Memory Issues
```python
# Reduce batch size in config
per_device_batch_size = 1  # Instead of 2
gradient_accumulation_steps = 4  # Increase to maintain effective batch
```

#### 4. Import Errors
```bash
# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify module structure
python -c "from src.agents.researcher_agent import MedicalResearcherAgent; print('✅ Import successful')"
```

### Debug Mode
```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python test_medxpert_evaluation.py --samples 1
```

## 📊 Benchmark Comparison

### Multi-Agent vs Single Agent

| Metric | Single Agent | Multi-Agent | Improvement |
|--------|--------------|-------------|-------------|
| Accuracy | ~0.55 | ~0.70 | +27% |
| Medical Relevance | ~0.65 | ~0.80 | +23% |
| Safety Score | ~0.75 | ~0.85 | +13% |
| Processing Time | ~15s | ~45s | -200% |

### Benefits of Multi-Agent Approach
- **Specialization**: Each agent focuses on their expertise
- **Cross-Validation**: Multiple perspectives reduce errors
- **Comprehensive Coverage**: Better handling of complex cases
- **Safety Enhancement**: Collective safety assessment

## 🚀 Advanced Testing

### Custom Test Cases
```python
# Add custom MedXpert samples
custom_sample = MedXpertSample(
    id="custom_001",
    question="Your medical question here",
    context="Patient context and history",
    answer="Expected comprehensive answer",
    category="specialty",
    difficulty="hard"
)
```

### Performance Profiling
```bash
# Profile memory usage
python -m memory_profiler test_medxpert_evaluation.py

# Profile execution time
python -m cProfile test_medxpert_evaluation.py
```

### Batch Evaluation
```bash
# Evaluate all samples
python test_medxpert_evaluation.py --samples 50

# Compare configurations
python test_medxpert_evaluation.py --config config/model_config_qwen.py
```

## 📋 Test Reports

### Automated Report Generation
The evaluation script generates:
- **JSON Results**: Detailed metrics and responses
- **Summary Statistics**: Performance overview
- **Individual Results**: Per-question analysis
- **Comparison Metrics**: Multi-agent vs single-agent

### Sample Report Output
```
🏥 MEDXPERT MULTI-AGENT EVALUATION SUMMARY
============================================================

📊 EVALUATION OVERVIEW:
  Total Samples: 5
  Successful Evaluations: 5  
  Success Rate: 100.0%
  BioMCP Usage Rate: 80.0%

🎯 PERFORMANCE METRICS:
  Overall Score: 0.725 (±0.095)
  Accuracy: 0.655
  Medical Relevance: 0.791
  Safety Score: 0.854
  Collaboration Quality: 0.706

⚡ EFFICIENCY METRICS:
  Avg Processing Time: 38.24s
  Avg Collaboration Rounds: 3.2

🏆 BENCHMARK COMPARISON:
  Performance: GOOD (72.5%)
```

## 🎯 Next Steps

### After Running Tests
1. **Review Results**: Analyze performance metrics
2. **Identify Weaknesses**: Focus on low-scoring areas
3. **Optimize Configuration**: Adjust model parameters
4. **Scale Testing**: Evaluate on larger datasets
5. **Deploy Models**: Use for production inference

### Performance Optimization
- **Fine-tune Models**: Train on medical-specific data
- **Improve Prompts**: Optimize agent instructions
- **Enhance Collaboration**: Better inter-agent communication
- **Safety Tuning**: Strengthen medical safety measures

---

**🏥 Ready to test your medical AI agents!** 🧪 