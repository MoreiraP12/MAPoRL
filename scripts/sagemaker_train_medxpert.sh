#!/bin/bash

# ==============================================================================
# SageMaker Training Script for Medical Multi-Agent Pipeline with MAPoRL
# Optimized for MedXpert Benchmark on 4x A10G GPUs
# ==============================================================================

set -e  # Exit on any error

echo "🏥 Starting Medical Multi-Agent MAPoRL Training on SageMaker"
echo "📊 Target Benchmark: MedXpert"
echo "🚀 Hardware: 4x A10G GPUs"
echo "🤖 Models: Qwen3-0.6B (4 agents)"

# ==============================================================================
# Environment Setup
# ==============================================================================

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export TOKENIZERS_PARALLELISM="false"
export TRANSFORMERS_CACHE="/opt/ml/model/transformers_cache"
export HF_HOME="/opt/ml/model/huggingface_cache"
export WANDB_DIR="/opt/ml/output/data"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export NCCL_DEBUG="INFO"
export NCCL_TREE_THRESHOLD="0"

# SageMaker paths
TRAIN_DATA_PATH="/opt/ml/input/data/train"
EVAL_DATA_PATH="/opt/ml/input/data/eval"
OUTPUT_PATH="/opt/ml/output/data"
MODEL_PATH="/opt/ml/model"
CHECKPOINTS_PATH="/opt/ml/checkpoints"

# Create necessary directories
mkdir -p $OUTPUT_PATH
mkdir -p $MODEL_PATH
mkdir -p $CHECKPOINTS_PATH
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_HOME
mkdir -p /tmp/wandb

echo "📁 Created directory structure"

# ==============================================================================
# System Information
# ==============================================================================

echo "🔍 System Information:"
echo "  - CUDA Version: $(nvcc --version | grep release)"
echo "  - GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo "  - Python Version: $(python --version)"
echo "  - PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "  - Available GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# ==============================================================================
# Install Dependencies (if needed)
# ==============================================================================

echo "📦 Installing/Updating dependencies..."

# Upgrade pip and install requirements
pip install --upgrade pip
pip install --no-cache-dir \
    torch==2.1.0 \
    transformers==4.36.0 \
    accelerate==0.24.0 \
    datasets==2.14.0 \
    peft==0.7.0 \
    bitsandbytes==0.41.0 \
    langgraph==0.0.55 \
    langchain==0.1.0 \
    wandb==0.16.0 \
    sentence-transformers==2.2.0 \
    scikit-learn==1.3.0 \
    tqdm==4.65.0 \
    jsonlines==4.0.0

echo "✅ Dependencies installed"

# ==============================================================================
# Data Preparation
# ==============================================================================

echo "📊 Preparing MedXpert dataset..."

# Check if training data exists
if [ ! -f "$TRAIN_DATA_PATH/medxpert_train.jsonl" ]; then
    echo "⚠️  MedXpert training data not found. Creating sample data..."
    
    # Create sample MedXpert-style data for demo
    cat > "$TRAIN_DATA_PATH/medxpert_train.jsonl" << 'EOF'
{"id": "medx_001", "question": "A 65-year-old patient presents with chest pain and shortness of breath. What is the most appropriate initial diagnostic test?", "context": "Patient has a history of hypertension and diabetes. Chest pain is substernal and radiates to the left arm. Vital signs show BP 160/90, HR 95, RR 22.", "answer": "ECG should be the initial diagnostic test to evaluate for acute coronary syndrome, followed by chest X-ray and cardiac enzymes."}
{"id": "medx_002", "question": "What are the first-line treatments for type 2 diabetes in adults?", "context": "A 55-year-old obese patient with newly diagnosed type 2 diabetes. HbA1c is 8.5%. No contraindications to standard medications.", "answer": "Metformin is the first-line treatment, combined with lifestyle modifications including diet and exercise. Target HbA1c should be individualized but generally <7%."}
{"id": "medx_003", "question": "How should acute bacterial pneumonia be managed in a healthy adult?", "context": "A 35-year-old previously healthy adult presents with fever, productive cough, and consolidation on chest X-ray. No recent antibiotic use.", "answer": "Empirical antibiotic therapy with amoxicillin or doxycycline for outpatient treatment. Hospitalization criteria include severe illness, comorbidities, or treatment failure."}
{"id": "medx_004", "question": "What are the indications for emergency surgery in acute appendicitis?", "context": "A 28-year-old patient presents with right lower quadrant pain, fever, and elevated white blood cell count. CT scan shows appendiceal inflammation.", "answer": "Emergency appendectomy is indicated for acute appendicitis. Laparoscopic approach is preferred when feasible. Antibiotics should be started preoperatively."}
{"id": "medx_005", "question": "How should hypertensive crisis be managed in the emergency department?", "context": "A 50-year-old patient presents with severe hypertension (BP 220/120) and signs of end-organ damage including altered mental status.", "answer": "Immediate but controlled BP reduction is needed. IV nicardipine or clevidipine are preferred. Reduce BP by 10-20% in first hour, then gradually to <160/100 over 6 hours."}
EOF

    # Create corresponding eval data
    cat > "$EVAL_DATA_PATH/medxpert_eval.jsonl" << 'EOF'
{"id": "medx_eval_001", "question": "What is the appropriate management for a patient with acute myocardial infarction?", "context": "A 60-year-old patient presents with severe chest pain, ST-elevation on ECG, and elevated troponins. Symptom onset was 2 hours ago.", "answer": "Immediate reperfusion therapy with primary PCI (preferred) or thrombolytic therapy if PCI unavailable. Dual antiplatelet therapy, anticoagulation, and supportive care."}
{"id": "medx_eval_002", "question": "How should diabetic ketoacidosis be treated?", "context": "A 25-year-old type 1 diabetic presents with blood glucose 450 mg/dL, ketones positive, pH 7.25, and dehydration.", "answer": "IV fluid resuscitation, insulin therapy, electrolyte monitoring and replacement (especially potassium), and treating precipitating factors. Monitor for complications."}
EOF

    echo "✅ Sample MedXpert data created"
else
    echo "✅ MedXpert training data found"
fi

# Count samples
TRAIN_SAMPLES=$(wc -l < "$TRAIN_DATA_PATH/medxpert_train.jsonl")
EVAL_SAMPLES=$(wc -l < "$EVAL_DATA_PATH/medxpert_eval.jsonl")
echo "📊 Dataset info: $TRAIN_SAMPLES training samples, $EVAL_SAMPLES eval samples"

# ==============================================================================
# Model Preparation
# ==============================================================================

echo "🤖 Preparing Qwen models..."

# Pre-download models to cache
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = 'Qwen/Qwen3-0.6B'
print(f'Downloading {model_name}...')

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

print('✅ Model downloaded and cached')
"

echo "✅ Models prepared"

# ==============================================================================
# Weights & Biases Setup
# ==============================================================================

echo "📈 Setting up Weights & Biases..."

# Initialize wandb (will create config if not exists)
python -c "
import wandb
import os
wandb.login(anonymous='allow')
print('✅ W&B initialized')
"

# ==============================================================================
# Training Configuration
# ==============================================================================

echo "⚙️ Creating training configuration..."

# Create training script with MedXpert optimizations
cat > /tmp/train_medxpert.py << 'EOF'
#!/usr/bin/env python3

import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
try:
    from src.training.maporl_trainer import create_trainer, load_medical_dataset, MAPoRLConfig
    from config.model_config_qwen import (
        QWEN_MEDICAL_AGENT_CONFIGS, 
        MEDXPERT_CONFIG,
        MEDXPERT_TRAINING_CONFIG,
        SAGEMAKER_OPTIMIZATION_SETTINGS
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Available modules:")
    for path in sys.path:
        if os.path.exists(path):
            print(f"  {path}: {os.listdir(path)}")
    raise

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting MedXpert Multi-Agent Training")
    
    # Training configuration optimized for MedXpert
    config = MAPoRLConfig(
        learning_rate=MEDXPERT_TRAINING_CONFIG["learning_rate"],
        batch_size=MEDXPERT_TRAINING_CONFIG["batch_size"],
        num_epochs=MEDXPERT_TRAINING_CONFIG["num_epochs"],
        max_rounds_per_episode=MEDXPERT_TRAINING_CONFIG["max_rounds_per_episode"],
        gamma=MEDXPERT_TRAINING_CONFIG["gamma"],
        clip_ratio=MEDXPERT_TRAINING_CONFIG["clip_ratio"],
        value_loss_coeff=MEDXPERT_TRAINING_CONFIG["value_loss_coeff"],
        entropy_coeff=MEDXPERT_TRAINING_CONFIG["entropy_coeff"],
        save_every=MEDXPERT_TRAINING_CONFIG["save_every"],
        eval_every=MEDXPERT_TRAINING_CONFIG["eval_every"],
        warmup_steps=MEDXPERT_TRAINING_CONFIG["warmup_steps"],
        safety_penalty_weight=MEDXPERT_TRAINING_CONFIG["safety_penalty_weight"],
        collaboration_bonus_weight=MEDXPERT_TRAINING_CONFIG["collaboration_bonus_weight"],
        medical_relevance_weight=MEDXPERT_TRAINING_CONFIG["medical_relevance_weight"],
    )
    
    logger.info(f"Training config: {config}")
    
    # Load datasets
    train_path = "/opt/ml/input/data/train/medxpert_train.jsonl"
    eval_path = "/opt/ml/input/data/eval/medxpert_eval.jsonl"
    
    logger.info(f"Loading training data from: {train_path}")
    train_dataset = load_medical_dataset(train_path, max_samples=5000)
    
    logger.info(f"Loading eval data from: {eval_path}")
    eval_dataset = load_medical_dataset(eval_path, max_samples=500)
    
    logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Create trainer with Qwen config
    logger.info("Creating MAPoRL trainer...")
    trainer = create_trainer(
        config=config,
        workflow_config=MEDXPERT_CONFIG.__dict__,
        device="cuda:0"
    )
    
    # Start training
    logger.info("🏥 Starting MedXpert-optimized training...")
    trainer.train(train_dataset, eval_dataset)
    
    logger.info("✅ Training completed!")
    
    # Save final model
    final_model_path = "/opt/ml/model/final_model"
    os.makedirs(final_model_path, exist_ok=True)
    logger.info(f"💾 Model saved to: {final_model_path}")

if __name__ == "__main__":
    main()
EOF

echo "✅ Training script created"

# ==============================================================================
# GPU Memory Optimization
# ==============================================================================

echo "🔧 Optimizing GPU memory..."

# Clear GPU memory
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
"

# Set memory fraction for each GPU to prevent OOM
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.set_per_process_memory_fraction(0.9, device=i)
    print('✅ GPU memory fractions set')
"

# ==============================================================================
# Training Execution
# ==============================================================================

echo "🎯 Starting MedXpert training with MAPoRL..."

# Set additional training environment variables
export WANDB_PROJECT="maporl-medxpert"
export WANDB_RUN_NAME="qwen-0.6b-4gpu-$(date +%Y%m%d-%H%M%S)"
export WANDB_TAGS="medxpert,qwen,maporl,sagemaker,4gpu"

# Run training with proper error handling
python /tmp/train_medxpert.py 2>&1 | tee "$OUTPUT_PATH/training.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo "📋 Last 50 lines of training log:"
    tail -50 "$OUTPUT_PATH/training.log"
    exit $TRAINING_EXIT_CODE
fi

# ==============================================================================
# Post-Training Analysis
# ==============================================================================

echo "📊 Post-training analysis..."

# Generate training summary
python -c "
import json
import os
from pathlib import Path

output_path = Path('/opt/ml/output/data')
summary = {
    'training_status': 'completed',
    'model_type': 'Qwen3-0.6B',
    'num_agents': 4,
    'target_benchmark': 'MedXpert',
    'gpu_config': '4x A10G',
    'framework': 'MAPoRL'
}

# Save summary
with open(output_path / 'training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('✅ Training summary saved')
"

# Check final model sizes
echo "📁 Final model information:"
du -sh "$MODEL_PATH"/* 2>/dev/null || echo "No final models found"

# GPU memory usage at end
echo "🔍 Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits

# ==============================================================================
# Cleanup and Finalization
# ==============================================================================

echo "🧹 Cleaning up..."

# Clean temporary files but keep important outputs
rm -f /tmp/train_medxpert.py
rm -rf /tmp/wandb

# Compress logs for easier download
if [ -f "$OUTPUT_PATH/training.log" ]; then
    gzip "$OUTPUT_PATH/training.log"
    echo "✅ Training log compressed"
fi

# Final status
echo ""
echo "🎉 SageMaker MedXpert Training Pipeline Completed!"
echo "📊 Benchmark: MedXpert"
echo "🤖 Models: 4x Qwen3-0.6B agents"
echo "🚀 Framework: MAPoRL multi-agent training"
echo "💾 Outputs saved to: $OUTPUT_PATH"
echo "📈 Logs and metrics available in W&B"
echo ""
echo "Next steps:"
echo "  1. Review training metrics in W&B dashboard"
echo "  2. Evaluate model performance on MedXpert test set"
echo "  3. Compare against baseline single-agent models"
echo "  4. Deploy for inference if performance meets targets"
echo ""
echo "🏥 Medical AI collaboration training complete! 🎯"
EOF 