#!/bin/bash

# ==============================================================================
# SageMaker Multi-Agent Medical Testing Script
# Tests Qwen3 0.6B models on MedXpert benchmark with BioMCP integration
# ==============================================================================

set -e  # Exit on any error

echo "🏥 Medical Multi-Agent Testing Pipeline"
echo "🧪 SageMaker + MedXpert Evaluation"
echo "🤖 Models: Qwen3-0.6B (4 agents)"
echo "💾 Hardware: 4x A10G GPUs"
echo "==========================================="

# ==============================================================================
# Environment Setup
# ==============================================================================

# Set environment variables
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_CACHE="/opt/ml/model/transformers_cache"
export HF_HOME="/opt/ml/model/huggingface_cache"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# SageMaker paths (with fallbacks for local testing)
TRAIN_DATA_PATH="${SM_CHANNEL_TRAINING:-./data/train}"
EVAL_DATA_PATH="${SM_CHANNEL_EVAL:-./data/eval}"
OUTPUT_PATH="${SM_OUTPUT_DATA_DIR:-./output}"
MODEL_PATH="${SM_MODEL_DIR:-./models}"

# Create directories
mkdir -p $OUTPUT_PATH
mkdir -p $MODEL_PATH
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME

echo "📁 Environment configured"

# ==============================================================================
# System Information
# ==============================================================================

echo "🔍 System Information:"
echo "  - CUDA Version: $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"
echo "  - GPU Count: $(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo '0')"
echo "  - Python Version: $(python --version)"
echo "  - PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'N/A')"

# Display GPU info if available
if command -v nvidia-smi &> /dev/null; then
    echo "  - GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -4
fi

# ==============================================================================
# Dependencies Installation
# ==============================================================================

echo "📦 Installing/Updating dependencies..."

# Core ML dependencies
pip install --quiet --upgrade \
    torch>=2.0.0 \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    datasets>=2.12.0 \
    sentence-transformers>=2.2.0

# Multi-agent framework
pip install --quiet \
    langgraph>=0.0.40 \
    langchain>=0.1.0

# BioMCP for real biomedical data
pip install --quiet \
    biomcp-python \
    uv

# Testing and evaluation
pip install --quiet \
    pytest>=7.4.0 \
    jsonlines>=3.1.0 \
    scikit-learn>=1.3.0

echo "✅ Dependencies installed"

# ==============================================================================
# BioMCP Setup and Testing
# ==============================================================================

echo "🧬 Setting up BioMCP..."

# Test BioMCP installation
if command -v uv &> /dev/null; then
    echo "✅ UV package manager available"
    
    # Test BioMCP connectivity
    if timeout 30 uv run --with biomcp-python biomcp --help > /dev/null 2>&1; then
        echo "✅ BioMCP successfully configured"
        export BIOMCP_AVAILABLE=true
    else
        echo "⚠️  BioMCP not responding, will use fallback mode"
        export BIOMCP_AVAILABLE=false
    fi
else
    echo "⚠️  UV not available, installing..."
    pip install uv
    export BIOMCP_AVAILABLE=false
fi

# ==============================================================================
# Qwen Model Preparation
# ==============================================================================

echo "🤖 Preparing Qwen3-0.6B models..."

# Pre-download and cache models
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
logging.basicConfig(level=logging.WARNING)

model_name = 'Qwen/Qwen3-0.6B'
print(f'📥 Downloading {model_name}...')

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir='$TRANSFORMERS_CACHE'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir='$TRANSFORMERS_CACHE'
    )
    
    print(f'✅ Model cached: {model.config.model_type}')
    print(f'   Parameters: ~{model.num_parameters() / 1e6:.1f}M')
    print(f'   Memory usage: ~{model.get_memory_footprint() / 1e9:.2f}GB')
    
except Exception as e:
    print(f'❌ Model download failed: {e}')
    exit(1)
"

echo "✅ Qwen models prepared"

# ==============================================================================
# Test Data Preparation
# ==============================================================================

echo "📊 Preparing MedXpert test data..."

# Create MedXpert sample data
cat > $OUTPUT_PATH/medxpert_samples.jsonl << 'EOF'
{"id": "medx_001", "question": "A 65-year-old patient presents with chest pain and shortness of breath. The chest pain is substernal, radiates to the left arm, and started 2 hours ago. What is the most appropriate initial diagnostic test?", "context": "Patient has a history of hypertension and diabetes mellitus. Vital signs: BP 160/90 mmHg, HR 95 bpm, RR 22/min, O2 sat 92% on room air. Patient appears diaphoretic and anxious.", "answer": "The most appropriate initial diagnostic test is a 12-lead electrocardiogram (ECG) to evaluate for acute coronary syndrome, specifically ST-elevation myocardial infarction (STEMI) or non-ST-elevation myocardial infarction (NSTEMI). This should be obtained within 10 minutes of presentation. Additional immediate tests should include chest X-ray and cardiac biomarkers (troponin).", "category": "cardiology", "difficulty": "medium"}
{"id": "medx_002", "question": "What are the first-line pharmacological treatments for newly diagnosed type 2 diabetes mellitus in adults without contraindications?", "context": "A 55-year-old obese patient (BMI 32) with newly diagnosed type 2 diabetes. HbA1c is 8.5%. No history of cardiovascular disease, kidney disease, or other contraindications to standard medications. Patient is motivated for lifestyle changes.", "answer": "The first-line pharmacological treatment for type 2 diabetes is metformin, typically starting at 500-850 mg twice daily with meals, titrated based on tolerance and glycemic response. This should be combined with intensive lifestyle modifications including dietary changes and regular physical activity. Target HbA1c should be individualized but generally <7% for most adults.", "category": "endocrinology", "difficulty": "easy"}
{"id": "medx_003", "question": "How should community-acquired pneumonia be managed in a previously healthy 35-year-old adult presenting to the emergency department?", "context": "Patient presents with 3-day history of fever, productive cough with purulent sputum, and pleuritic chest pain. Vital signs: temperature 38.8°C, HR 110 bpm, RR 24/min, BP 120/80 mmHg, O2 sat 94% on room air. Chest X-ray shows right lower lobe consolidation.", "answer": "For a previously healthy adult with community-acquired pneumonia, empirical antibiotic therapy should be initiated. Recommended first-line treatment includes amoxicillin 1g three times daily or doxycycline 100mg twice daily for outpatient management. However, given the patient's presentation in the ED with tachycardia and hypoxemia, consider hospitalization criteria using CURB-65 or PSI scores. If hospitalized, use combination therapy with beta-lactam plus macrolide or respiratory fluoroquinolone.", "category": "pulmonology", "difficulty": "medium"}
{"id": "medx_004", "question": "A patient with a BRCA1 mutation asks about cancer screening recommendations. What should be advised?", "context": "A 30-year-old woman with a confirmed pathogenic BRCA1 mutation. Family history includes mother with breast cancer at age 45 and maternal grandmother with ovarian cancer at age 60. Patient is currently healthy with no symptoms.", "answer": "For a BRCA1 mutation carrier, enhanced screening is recommended: 1) Breast surveillance: Annual breast MRI starting at age 25-30, clinical breast exam every 6 months; 2) Consider risk-reducing bilateral mastectomy; 3) Ovarian cancer screening: Risk-reducing bilateral salpingo-oophorectomy between ages 35-40 or after childbearing is complete; 4) Genetic counseling for family planning; 5) Consider chemoprevention options. Screening should be coordinated with a high-risk clinic or genetic counselor.", "category": "oncology", "difficulty": "hard"}
{"id": "medx_005", "question": "What is the appropriate management for a patient presenting with acute severe hypertension (BP 220/120 mmHg) and signs of end-organ damage?", "context": "A 50-year-old patient presents with severe headache, altered mental status, and blood pressure of 220/120 mmHg. Fundoscopic exam shows papilledema and flame-shaped hemorrhages. Patient has a history of poorly controlled hypertension but no recent medication changes.", "answer": "This represents hypertensive emergency requiring immediate but controlled blood pressure reduction. Goals: 1) Reduce BP by 10-20% in the first hour, then gradually to <160/100 mmHg over the next 6 hours; 2) Use IV antihypertensives such as nicardipine (preferred) or clevidipine; 3) Avoid rapid or excessive BP reduction to prevent cerebral, coronary, or renal ischemia; 4) Continuous cardiac monitoring and frequent neurologic assessments; 5) Investigate and treat underlying causes; 6) ICU admission for close monitoring.", "category": "cardiology", "difficulty": "hard"}
EOF

echo "✅ MedXpert test data prepared (5 samples)"

# ==============================================================================
# BioMCP Integration Testing
# ==============================================================================

echo "🧬 Testing BioMCP integration..."

python -c "
import asyncio
import sys
import os
sys.path.append('.')

async def test_biomcp():
    try:
        # Test BioMCP availability
        if os.getenv('BIOMCP_AVAILABLE') == 'true':
            import subprocess
            result = subprocess.run(
                ['uv', 'run', '--with', 'biomcp-python', 'biomcp', 'trial', 'search', '--condition', 'diabetes', '--limit', '1'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print('✅ BioMCP clinical trials search working')
                return True
            else:
                print('⚠️  BioMCP trials search failed, using fallback')
                return False
        else:
            print('⚠️  BioMCP not available, using fallback mode')
            return False
    except Exception as e:
        print(f'⚠️  BioMCP test failed: {e}')
        return False

result = asyncio.run(test_biomcp())
print(f'BioMCP Status: {\"Active\" if result else \"Fallback Mode\"}')
" 2>/dev/null || echo "⚠️  BioMCP test skipped"

# ==============================================================================
# Multi-Agent System Testing
# ==============================================================================

echo "🏥 Testing Multi-Agent Medical System..."

# Create comprehensive test script
cat > $OUTPUT_PATH/run_medxpert_test.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import json
import logging
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append('.')

# Configure logging for SageMaker
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/ml/output/data/test.log' if os.path.exists('/opt/ml/output/data') else './test.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Run comprehensive MedXpert evaluation."""
    logger.info("🚀 Starting MedXpert Multi-Agent Evaluation")
    
    start_time = time.time()
    
    try:
        # Import evaluation modules
        from test_medxpert_evaluation import MedXpertEvaluator, MedXpertSample
        
        # Initialize evaluator
        logger.info("📋 Initializing evaluator...")
        evaluator = MedXpertEvaluator()
        
        # Load test samples
        samples_file = '/opt/ml/output/data/medxpert_samples.jsonl' if os.path.exists('/opt/ml/output/data/medxpert_samples.jsonl') else './output/medxpert_samples.jsonl'
        
        samples = []
        with open(samples_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                sample = MedXpertSample(
                    id=data['id'],
                    question=data['question'],
                    context=data['context'],
                    answer=data['answer'],
                    category=data.get('category', 'general'),
                    difficulty=data.get('difficulty', 'medium')
                )
                samples.append(sample)
        
        logger.info(f"📊 Loaded {len(samples)} test samples")
        
        # Run evaluation
        logger.info("🏥 Running multi-agent evaluation...")
        results = await evaluator.evaluate_all_samples(samples)
        
        # Analyze results
        analysis = evaluator.analyze_results(results)
        
        # Save results
        output_file = '/opt/ml/output/data/medxpert_results.json' if os.path.exists('/opt/ml/output/data') else './medxpert_results.json'
        evaluator.save_results(results, analysis, output_file)
        
        # Print summary
        evaluator.print_summary(analysis)
        
        # Log key metrics
        metrics = analysis['performance_metrics']
        logger.info(f"📊 FINAL METRICS:")
        logger.info(f"   Overall Score: {metrics['overall']['mean']:.3f}")
        logger.info(f"   Accuracy: {metrics['accuracy']['mean']:.3f}")
        logger.info(f"   Medical Relevance: {metrics['medical_relevance']['mean']:.3f}")
        logger.info(f"   Safety Score: {metrics['safety']['mean']:.3f}")
        logger.info(f"   Collaboration Quality: {metrics['collaboration']['mean']:.3f}")
        
        # Create summary for SageMaker
        summary = {
            'test_status': 'completed',
            'total_time': time.time() - start_time,
            'samples_tested': len(samples),
            'overall_score': metrics['overall']['mean'],
            'accuracy': metrics['accuracy']['mean'],
            'medical_relevance': metrics['medical_relevance']['mean'],
            'safety_score': metrics['safety']['mean'],
            'collaboration_quality': metrics['collaboration']['mean'],
            'biomcp_usage_rate': analysis['summary']['biomcp_usage_rate']
        }
        
        summary_file = '/opt/ml/output/data/test_summary.json' if os.path.exists('/opt/ml/output/data') else './test_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Evaluation completed in {time.time() - start_time:.1f}s")
        logger.info(f"📊 Results saved to: {output_file}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error summary
        error_summary = {
            'test_status': 'failed',
            'error': str(e),
            'total_time': time.time() - start_time
        }
        
        error_file = '/opt/ml/output/data/test_summary.json' if os.path.exists('/opt/ml/output/data') else './test_summary.json'
        with open(error_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        raise

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Run the comprehensive test
echo "🚀 Executing MedXpert evaluation..."
python $OUTPUT_PATH/run_medxpert_test.py

TEST_EXIT_CODE=$?

# ==============================================================================
# Results Analysis
# ==============================================================================

echo "📊 Analyzing test results..."

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ MedXpert evaluation completed successfully!"
    
    # Display summary if available
    if [ -f "$OUTPUT_PATH/test_summary.json" ]; then
        python -c "
import json
try:
    with open('$OUTPUT_PATH/test_summary.json', 'r') as f:
        summary = json.load(f)
    
    print('\\n🏆 FINAL PERFORMANCE SUMMARY:')
    print('=' * 50)
    print(f'Overall Score: {summary.get(\"overall_score\", 0):.3f}')
    print(f'Accuracy: {summary.get(\"accuracy\", 0):.3f}')
    print(f'Medical Relevance: {summary.get(\"medical_relevance\", 0):.3f}')
    print(f'Safety Score: {summary.get(\"safety_score\", 0):.3f}')
    print(f'Collaboration Quality: {summary.get(\"collaboration_quality\", 0):.3f}')
    print(f'BioMCP Usage: {summary.get(\"biomcp_usage_rate\", 0):.1%}')
    print(f'Test Duration: {summary.get(\"total_time\", 0):.1f}s')
    print('=' * 50)
    
    overall = summary.get('overall_score', 0)
    if overall >= 0.75:
        print('🎉 EXCELLENT: Multi-agent system performing very well!')
    elif overall >= 0.65:
        print('✅ GOOD: Multi-agent system showing strong performance')
    elif overall >= 0.55:
        print('⚠️  MODERATE: System working but has room for improvement')
    else:
        print('🔧 NEEDS WORK: Consider tuning model parameters')
        
except Exception as e:
    print(f'Could not read summary: {e}')
"
    fi
    
else
    echo "❌ MedXpert evaluation failed with exit code: $TEST_EXIT_CODE"
    echo "📋 Check logs for details:"
    if [ -f "$OUTPUT_PATH/test.log" ]; then
        tail -20 "$OUTPUT_PATH/test.log"
    fi
fi

# ==============================================================================
# Cleanup and Finalization
# ==============================================================================

echo "🧹 Finalizing test results..."

# Compress logs
if [ -f "$OUTPUT_PATH/test.log" ]; then
    gzip "$OUTPUT_PATH/test.log"
    echo "📋 Test log compressed"
fi

# Final GPU memory check
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Final GPU memory usage:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
fi

# Create final status report
cat > $OUTPUT_PATH/sagemaker_test_report.txt << EOF
SageMaker Multi-Agent Medical System Test Report
================================================

Test Date: $(date)
Test Duration: $(echo "scale=1; $(date +%s) - $start_timestamp" | bc 2>/dev/null || echo "N/A")s
Models: Qwen3-0.6B (4 agents)
Benchmark: MedXpert Medical Questions
BioMCP Integration: ${BIOMCP_AVAILABLE:-false}

Test Status: $([ $TEST_EXIT_CODE -eq 0 ] && echo "PASSED" || echo "FAILED")
Exit Code: $TEST_EXIT_CODE

Files Generated:
- medxpert_results.json (detailed evaluation results)
- test_summary.json (performance summary)
- test.log.gz (compressed test logs)
- sagemaker_test_report.txt (this report)

Next Steps:
1. Review detailed results in medxpert_results.json
2. Analyze performance metrics and agent collaboration
3. Compare with baseline single-agent performance
4. Consider parameter tuning for improved performance
5. Scale to larger test sets if performance is satisfactory

$([ $TEST_EXIT_CODE -eq 0 ] && echo "✅ Test completed successfully!" || echo "❌ Test failed - check logs for details")
EOF

echo ""
echo "🎉 SageMaker Multi-Agent Testing Complete!"
echo "📊 Test Report: $OUTPUT_PATH/sagemaker_test_report.txt"
echo "📋 Detailed Results: $OUTPUT_PATH/medxpert_results.json"
echo "📈 Performance Summary: $OUTPUT_PATH/test_summary.json"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Your multi-agent medical system is working!"
    echo "🔗 Next: Review results and consider scaling to full MedXpert dataset"
else
    echo "❌ FAILURE: Check logs and configuration"
    echo "💡 Tip: Ensure all dependencies are installed and models are accessible"
fi

exit $TEST_EXIT_CODE 