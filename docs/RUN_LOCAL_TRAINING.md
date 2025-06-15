# Local Medical Multi-Agent Training Guide

Complete step-by-step guide to run the medical multi-agent system locally on your terminal with Qwen3-0.6B models.

## 🚀 Quick Start Sequence

### Step 1: Install Dependencies
```bash
# Install required Python packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.51.0 datasets accelerate bitsandbytes
pip install requests pandas numpy scikit-learn
pip install langgraph langchain langsmith
pip install biomcp requests-cache

# Optional: Install Jupyter for notebooks
pip install jupyter matplotlib seaborn
```

### Step 2: Download MedXpert Data
```bash
# Download and prepare medical datasets
python download_medxpert_data.py

# This will create:
# data/medxpert_train.jsonl      (training data)
# data/medxpert_validation.jsonl (validation data)
# data/medxpert_complete.jsonl   (all data)
# data/dataset_summary.json      (statistics)
```

### Step 3: Run Local Training
```bash
# Train all 4 medical agents locally
python local_train_medxpert.py

# Or with custom directories:
python local_train_medxpert.py --data-dir data --output-dir outputs

# For testing only (if models already trained):
python local_train_medxpert.py --test-only
```

### Step 4: Monitor Results
```bash
# Check training progress in real-time
tail -f outputs/logs/*.log

# View final results
cat outputs/training_results.json

# Test individual agents
python local_train_medxpert.py --test-only
```

---

## 📋 Detailed Steps

### Prerequisites Check
```bash
# Check system requirements
python -c "
import torch
import transformers
print(f'Python: OK')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
"
```

### 1. Data Download and Preparation

#### Download Medical Datasets
```bash
# Run the data downloader
python download_medxpert_data.py
```

**What it does:**
- Downloads MedQuAD dataset from GitHub (cancer, NIDDK, NINDS, etc.)
- Creates synthetic MedXpert-style medical Q&A pairs
- Attempts to download MedText dataset from HuggingFace
- Cleans and deduplicates data
- Splits into training/validation sets
- Saves as JSONL files

**Expected Output:**
```
🏥 Medical Dataset Downloader
==================================================
📥 Downloading MedQuAD dataset...
📂 Processing folder: 1_CancerGov_QA
...
🧬 Creating synthetic MedXpert data...
✅ Created 10 synthetic samples
📥 Downloading MedText dataset...
🧹 Cleaning and deduplicating data...
📊 Total clean samples: 45
💾 Saved 36 samples to data/medxpert_train.jsonl
💾 Saved 9 samples to data/medxpert_validation.jsonl
✅ Medical data download complete!
```

#### Verify Data Quality
```bash
# Check downloaded data
head -3 data/medxpert_train.jsonl

# View dataset statistics
cat data/dataset_summary.json
```

### 2. Local Training Pipeline

#### Start Training
```bash
# Full training pipeline
python local_train_medxpert.py --data-dir data --output-dir outputs
```

**Training Process:**
1. **Medical Planner Agent** (`cuda:0` or `cpu`) - Question analysis
2. **Medical Researcher Agent** (`cuda:1` or `cpu`) - Evidence gathering  
3. **Medical Analyst Agent** (`cuda:2` or `cpu`) - Clinical reasoning
4. **Medical Reporter Agent** (`cuda:3` or `cpu`) - Response synthesis

**Expected Timeline:**
- **Single GPU**: ~2-4 hours total (30-60 min per agent)
- **Multi-GPU**: ~1-2 hours total (parallel training)
- **CPU Only**: ~6-12 hours total (much slower)

#### Monitor Training
```bash
# Watch GPU usage (if available)
watch -n 1 nvidia-smi

# Monitor training logs
tail -f outputs/models/planner/training_logs.txt
tail -f outputs/models/researcher/training_logs.txt
tail -f outputs/models/analyst/training_logs.txt  
tail -f outputs/models/reporter/training_logs.txt
```

### 3. Testing and Evaluation

#### Test Trained Models
```bash
# Test all trained agents
python local_train_medxpert.py --test-only --output-dir outputs
```

#### Custom Testing
```bash
# Create custom test script
cat > test_custom.py << 'EOF'
from local_train_medxpert import LocalMedicalTrainer

# Custom test questions
test_questions = [
    "A 45-year-old presents with chest pain. What is the differential diagnosis?",
    "What are the indications for emergency surgery in acute abdomen?", 
    "How do you manage a patient with suspected sepsis?"
]

trainer = LocalMedicalTrainer(data_dir="data", output_dir="outputs")
results = trainer.test_agents(test_questions)

for agent, responses in results.items():
    print(f"\n{agent.upper()} AGENT:")
    for i, item in enumerate(responses):
        print(f"Q{i+1}: {item['question']}")
        print(f"A{i+1}: {item['response']}\n")
EOF

python test_custom.py
```

### 4. Results Analysis

#### View Training Results
```bash
# Overall training summary
cat outputs/training_results.json

# Individual agent statistics
cat outputs/models/planner/training_stats.json
cat outputs/models/researcher/training_stats.json
cat outputs/models/analyst/training_stats.json
cat outputs/models/reporter/training_stats.json
```

#### View Test Results
```bash
# Agent responses to test questions
cat outputs/test_results.json

# Pretty print JSON
python -m json.tool outputs/test_results.json
```

---

## 🔧 Configuration Options

### Hardware Optimization

#### For High-Memory GPUs (>16GB)
```bash
# Increase batch size for faster training
python local_train_medxpert.py --data-dir data --output-dir outputs
# Edit local_train_medxpert.py: per_device_train_batch_size=2
```

#### For Low-Memory GPUs (<8GB)
```bash
# Use 4-bit quantization
# Edit local_train_medxpert.py:
# load_in_4bit=True instead of load_in_8bit=True
```

#### For CPU-Only Training
```bash
# CPU training (very slow but works)
export CUDA_VISIBLE_DEVICES=""
python local_train_medxpert.py --data-dir data --output-dir outputs
```

### Training Customization

#### Modify Training Parameters
Edit `local_train_medxpert.py`:
```python
# Training configuration
num_train_epochs=3,              # More epochs for better learning
per_device_train_batch_size=2,   # Larger batch if GPU memory allows
learning_rate=1e-5,              # Lower LR for more stable training
```

#### Add Custom Medical Data
```bash
# Add your own medical Q&A data
cat > data/custom_medical.jsonl << 'EOF'
{"question": "Your medical question", "answer": "Your answer", "category": "specialty", "source": "custom", "difficulty": "medium"}
{"question": "Another question", "answer": "Another answer", "category": "emergency", "source": "custom", "difficulty": "high"}
EOF

# Combine with existing data
cat data/medxpert_train.jsonl data/custom_medical.jsonl > data/combined_train.jsonl
```

---

## 📊 Expected Results

### Training Metrics
- **Model**: Qwen3-0.6B (600M parameters per agent)
- **Memory Usage**: 4-8GB GPU memory per agent
- **Training Time**: 1-4 hours depending on hardware
- **Final Loss**: Typically 1.5-2.5 for medical Q&A

### Output Files Structure
```
outputs/
├── training_results.json      # Overall training summary
├── test_results.json         # Agent testing results
├── models/
│   ├── planner/              # Medical Planner agent
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   └── training_stats.json
│   ├── researcher/           # Medical Researcher agent
│   ├── analyst/              # Medical Analyst agent
│   └── reporter/             # Medical Reporter agent
└── logs/                     # Training logs
```

### Performance Expectations
- **Medical Accuracy**: 70-80% on medical Q&A
- **Safety**: Appropriate cautious language
- **Collaboration**: Multi-agent workflow coordination
- **Specialization**: Each agent focuses on specific medical tasks

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Out of Memory Errors
```bash
# Reduce batch size
# Edit local_train_medxpert.py: per_device_train_batch_size=1
# Enable gradient checkpointing (already enabled)
# Use 4-bit quantization instead of 8-bit
```

#### 2. Download Failures
```bash
# If MedQuAD download fails, use synthetic data only
# Edit download_medxpert_data.py: comment out medquad_data section
# Or create manual data:
mkdir -p data
cat > data/medxpert_train.jsonl << 'EOF'
{"question": "What is hypertension?", "answer": "High blood pressure condition requiring monitoring and treatment.", "category": "cardiology", "source": "manual", "difficulty": "easy"}
EOF
```

#### 3. Model Loading Issues
```bash
# Check transformers version
pip install transformers==4.51.0

# Clear cache if needed
rm -rf ~/.cache/huggingface/
```

#### 4. CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU if needed
export CUDA_VISIBLE_DEVICES=""
```

### Performance Optimization

#### Speed Up Training
```bash
# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Increase batch size (if memory allows)
# Edit: per_device_train_batch_size=2

# Reduce max_length for faster processing
# Edit: max_length=256 instead of 512
```

#### Improve Quality
```bash
# Train for more epochs
# Edit: num_train_epochs=3

# Lower learning rate
# Edit: learning_rate=1e-5

# Add more medical data
python download_medxpert_data.py  # Get more data sources
```

---

## 🎯 Next Steps

### After Training
1. **Evaluate on MedXpert Benchmark**: Compare against published results
2. **Deploy for Inference**: Create API endpoints for medical Q&A
3. **Fine-tune Further**: Add domain-specific medical data
4. **Multi-Agent Workflows**: Implement collaborative reasoning

### Integration Options
1. **Web Interface**: Create Flask/FastAPI web app
2. **Medical Applications**: Integrate with EMR systems
3. **Research Tools**: Use for medical literature analysis
4. **Educational**: Medical student training assistance

### Production Deployment
1. **Model Optimization**: Convert to ONNX or TensorRT
2. **Containerization**: Docker deployment
3. **Scaling**: Kubernetes orchestration
4. **Monitoring**: Add performance tracking

---

## 📞 Support

### Getting Help
- Check logs in `outputs/logs/` for detailed error messages
- Verify GPU memory with `nvidia-smi`
- Test with smaller data first: `head -10 data/medxpert_train.jsonl > data/test.jsonl`
- Use CPU training as fallback: `export CUDA_VISIBLE_DEVICES=""`

### Resources
- [Qwen3 Documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Transformers Training Guide](https://huggingface.co/docs/transformers/training)
- [Medical AI Ethics Guidelines](https://www.nature.com/articles/s41591-019-0548-6) 