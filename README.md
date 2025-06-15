# MAPoRL - Medical Multi-Agent Pipeline with Reinforcement Learning

A comprehensive framework for training medical multi-agent systems using reinforcement learning, specifically designed for medical question answering and decision support.

## 🚀 Quick Start

### Basic Usage

```bash
# 1. Setup W&B integration and check your environment
python scripts/utilities/setup_wandb.py

# 2. Download medical datasets
python scripts/data/download_medxpert_data.py

# 3. Train medical agents locally
python scripts/training/local_train_medxpert.py --data-dir data --output-dir outputs

# 4. Run the complete training pipeline
bash scripts/utilities/run_local_medical_training.sh
```

### Example W&B Integration

```bash
# Run example to see W&B logging in action
python examples/example_wandb_training.py

# Train with W&B logging
python scripts/training/train.py \
    --train_data data/train.jsonl \
    --wandb_project "my-medical-project" \
    --experiment_name "baseline-experiment"
```

## 📁 Repository Structure

```
MAPoRL/
├── 📚 docs/                          # Documentation
│   ├── WANDB_INTEGRATION_GUIDE.md    # W&B integration guide
│   ├── RUN_LOCAL_TRAINING.md         # Local training guide
│   ├── SAGEMAKER_DEPLOYMENT_GUIDE.md # SageMaker deployment
│   ├── SAGEMAKER_README.md           # SageMaker documentation
│   └── TESTING_README.md             # Testing guide
│
├── 🎯 examples/                       # Example scripts and demos
│   ├── example_wandb_training.py     # W&B integration example
│   └── demo.py                       # Basic demo script
│
├── ☁️  sagemaker/                     # SageMaker training
│   ├── sagemaker_entry.py            # SageMaker entry point
│   ├── launch_sagemaker_training.py  # Launch SageMaker jobs
│   └── requirements_sagemaker.txt    # SageMaker dependencies
│
├── 🛠️ scripts/                        # Training and utility scripts
│   ├── training/                     # Training scripts
│   │   ├── train.py                  # Main training script
│   │   └── local_train_medxpert.py   # Local MedXpert training
│   ├── data/                         # Data processing scripts
│   │   ├── download_medxpert_data.py # Download MedXpert dataset
│   │   └── download_real_medxpert.py # Real MedXpert downloader
│   ├── testing/                      # Testing scripts
│   │   ├── test_medxpert_evaluation.py
│   │   ├── test_biomcp_integration.py
│   │   ├── simple_test.py
│   │   └── simple_test_basic.py
│   └── utilities/                    # Utility scripts
│       ├── setup_wandb.py           # W&B setup utility
│       ├── run_local_medical_training.sh
│       ├── sagemaker_test_medxpert.sh
│       └── sagemaker_train_medxpert.sh
│
├── 🧠 src/                           # Core source code
│   ├── agents/                      # Multi-agent implementations
│   ├── training/                    # Training frameworks
│   ├── workflow/                    # Workflow management
│   ├── reward/                      # Reward systems
│   └── config/                      # Configuration management
│
├── ⚙️  config/                       # Model configurations
│   ├── model_config.py             # Base model config
│   └── model_config_qwen.py        # Qwen-specific config
│
├── 📊 data/                         # Training data (ignored)
├── 📁 outputs/                      # Training outputs (ignored)
└── 📄 requirements.txt              # Dependencies
```

## 🏥 Medical Multi-Agent System

### Architecture

MAPoRL implements a collaborative multi-agent system with specialized roles:

- **🎯 Planner Agent**: Analyzes medical questions and plans response strategy
- **🔍 Researcher Agent**: Gathers medical evidence and clinical guidelines
- **🧮 Analyst Agent**: Provides clinical reasoning and differential diagnosis
- **📝 Reporter Agent**: Synthesizes comprehensive medical responses

### Key Features

- 🤖 **Multi-Agent Collaboration**: Agents work together using LangGraph workflows
- 🧠 **Reinforcement Learning**: MAPoRL (Multi-Agent Post-training with RL) framework
- 🏥 **Medical Safety**: Built-in safety mechanisms and medical disclaimers
- 📊 **Comprehensive Logging**: Full W&B integration for experiment tracking
- ☁️ **Scalable Training**: Local and SageMaker training options
- 🎯 **Medical Benchmarks**: Optimized for MedXpert and medical QA tasks

## 📚 Training Options

### 1. Local Training

**Quick Start:**
```bash
# Complete pipeline
bash scripts/utilities/run_local_medical_training.sh

# Custom training
python scripts/training/local_train_medxpert.py \
    --data-dir data \
    --output-dir outputs \
    --wandb-project "my-experiment"
```

**Features:**
- 🔧 Multi-GPU support (automatically detected)
- 💾 Memory optimization with quantization
- 📈 Real-time W&B logging
- 🎯 Medical-specific metrics

### 2. SageMaker Training

**Setup:**
```bash
# Launch SageMaker training
cd sagemaker/
python launch_sagemaker_training.py
```

**Features:**
- ☁️ 4x A10G GPU training (ml.g5.12xlarge)
- 📊 Automatic hyperparameter optimization
- 🔄 Distributed training support
- 📈 CloudWatch and W&B integration

### 3. Advanced Training

**Custom Training Script:**
```bash
python scripts/training/train.py \
    --train_data data/medxpert_train.jsonl \
    --eval_data data/medxpert_eval.jsonl \
    --learning_rate 1e-5 \
    --batch_size 4 \
    --num_epochs 10 \
    --wandb_project "medical-research" \
    --experiment_name "safety-study" \
    --safety_penalty_weight 3.0
```

## 📊 Monitoring & Evaluation

### Weights & Biases Integration

MAPoRL provides comprehensive W&B logging:

```bash
# Setup W&B integration
python scripts/utilities/setup_wandb.py

# Check integration status
python scripts/utilities/setup_wandb.py --check
```

**Logged Metrics:**
- 🎯 Training accuracy and loss per agent
- 🏥 Medical relevance and safety scores
- 🤝 Agent collaboration quality
- 💻 System resource utilization
- 📈 Real-time training progress

### Testing & Evaluation

```bash
# Run comprehensive tests
python scripts/testing/test_medxpert_evaluation.py

# Test specific components
python scripts/testing/test_biomcp_integration.py

# Simple functionality test
python scripts/testing/simple_test.py
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ VRAM for local training
- W&B account (for logging)

### Installation

1. **Clone and Setup:**
```bash
git clone <repository-url>
cd MAPoRL
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Setup W&B:**
```bash
wandb login
python scripts/utilities/setup_wandb.py
```

4. **Download Data:**
```bash
python scripts/data/download_medxpert_data.py
```

5. **Start Training:**
```bash
bash scripts/utilities/run_local_medical_training.sh
```

## 📖 Documentation

- 📚 **[W&B Integration Guide](docs/WANDB_INTEGRATION_GUIDE.md)** - Complete W&B setup and usage
- 🏃 **[Local Training Guide](docs/RUN_LOCAL_TRAINING.md)** - Local training instructions
- ☁️ **[SageMaker Guide](docs/SAGEMAKER_DEPLOYMENT_GUIDE.md)** - Cloud training setup
- 🧪 **[Testing Guide](docs/TESTING_README.md)** - Testing and evaluation

## 🤝 Contributing

1. Follow the organized directory structure
2. Add comprehensive W&B logging to new training scripts
3. Update relevant documentation
4. Test both local and SageMaker deployments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for medical AI research and education
- Integrates with leading ML frameworks (PyTorch, Transformers, LangGraph)
- Optimized for medical benchmarks (MedXpert, MedQA)

---

For detailed usage instructions, see the documentation in the `docs/` directory. 