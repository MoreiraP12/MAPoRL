# Weights & Biases Integration Guide for MAPoRL

This guide explains how to integrate Weights & Biases (W&B) logging into your MAPoRL medical multi-agent training scripts.

## 🚀 Quick Start

### 1. Install and Setup W&B

W&B is already included in your `requirements.txt`. To set up:

```bash
# Login to W&B (first time only)
wandb login

# Or use API key
export WANDB_API_KEY=your_api_key_here
```

### 2. Environment Variables

Set these environment variables for consistent W&B usage:

```bash
export WANDB_PROJECT="maporl-medical"
export WANDB_ENTITY="your-username"  # Optional: your W&B username/team
export WANDB_DIR="./outputs/wandb"   # Optional: custom directory
```

## 📊 Training Scripts with W&B Integration

### Main Training Script (`train.py`)

**Enhanced with W&B logging:**

```bash
# Basic usage
python train.py --train_data data/train.jsonl --wandb_project "maporl-medical"

# Advanced usage with custom naming and tags
python train.py \
    --train_data data/train.jsonl \
    --eval_data data/eval.jsonl \
    --wandb_project "maporl-medical" \
    --experiment_name "experiment-v1" \
    --wandb_tags "baseline" "qwen" "multi-agent" \
    --wandb_group "baseline-experiments" \
    --wandb_notes "Testing new reward system"

# Disable W&B logging
python train.py --train_data data/train.jsonl --disable_wandb
```

**New arguments added:**
- `--wandb_project`: W&B project name
- `--wandb_tags`: List of tags for the run
- `--wandb_group`: Group name for organizing runs
- `--wandb_notes`: Notes for the run
- `--disable_wandb`: Disable W&B logging

### Local MedXpert Training (`local_train_medxpert.py`)

**Enhanced with comprehensive W&B integration:**

```bash
# Basic usage
python local_train_medxpert.py --data-dir data --output-dir outputs

# With W&B customization
python local_train_medxpert.py \
    --data-dir data \
    --output-dir outputs \
    --wandb-project "maporl-medxpert-local" \
    --wandb-name "qwen-4agents-experiment" \
    --wandb-group "local-training" \
    --wandb-tags "medxpert" "qwen" "local" "4-agents" \
    --wandb-notes "Training 4 agents on MedXpert dataset locally"

# Disable W&B logging
python local_train_medxpert.py --data-dir data --output-dir outputs --disable-wandb
```

**New arguments added:**
- `--wandb-project`: W&B project name
- `--wandb-name`: Custom run name
- `--wandb-group`: Group for organizing runs
- `--wandb-tags`: Tags for the run
- `--wandb-notes`: Run description
- `--disable-wandb`: Disable W&B logging

### SageMaker Training (`sagemaker_entry.py`)

**Automatic W&B integration with environment variables:**

The SageMaker script automatically initializes W&B with:
- Project: `maporl-medxpert-sagemaker`
- Tags: `["sagemaker", "medxpert", "qwen", "multi-agent", "4gpu"]`
- Infrastructure info logged automatically

## 📈 What Gets Logged

### 1. System Information
```python
# Automatically logged
{
    "system/gpu_count": 4,
    "system/device": "cuda:0",
    "system/cuda_available": True,
    "system/pytorch_version": "2.1.0"
}
```

### 2. Training Configuration
```python
# Model and training hyperparameters
{
    "learning_rate": 1e-5,
    "batch_size": 2,
    "num_epochs": 10,
    "max_rounds_per_episode": 3,
    "gamma": 0.99,
    "clip_ratio": 0.2,
    "safety_penalty_weight": 2.0,
    "collaboration_bonus_weight": 1.5,
    "medical_relevance_weight": 1.2
}
```

### 3. Dataset Information
```python
# Dataset statistics
{
    "data/train_samples": 1000,
    "data/eval_samples": 200,
    "data/categories": {"cardiology": 150, "neurology": 100, ...},
    "data/train_data_path": "data/train.jsonl"
}
```

### 4. Training Metrics (Per Agent)
```python
# For each agent (planner, researcher, analyst, reporter)
{
    "agent_planner/final_train_loss": 0.45,
    "agent_planner/training_time": 1200.5,
    "agent_planner/train_samples": 1000,
    "agent_planner/device": "cuda:0"
}
```

### 5. Summary Metrics
```python
# Overall training summary
{
    "summary/total_agents_trained": 4,
    "summary/average_train_loss": 0.42,
    "summary/total_training_time": 4800.0,
    "summary/train_samples": 1000,
    "summary/eval_samples": 200
}
```

### 6. Performance Tables
- Agent performance comparison table
- Training metrics across epochs
- Medical relevance and safety scores

### 7. Artifacts
- Training configuration files
- Model checkpoints (if enabled)
- Training results JSON
- Log files

## 🎯 Advanced Features

### 1. Model Checkpointing
```python
# Environment variable to log model checkpoints
export WANDB_LOG_MODEL="checkpoint"  # Log all checkpoints
export WANDB_LOG_MODEL="end"         # Log only final model
```

### 2. Gradient and Parameter Watching
```python
# Environment variable to watch model parameters
export WANDB_WATCH="all"       # Watch gradients and parameters
export WANDB_WATCH="gradients"  # Watch only gradients
export WANDB_WATCH="parameters" # Watch only parameters
```

### 3. Custom Metrics
The integration includes medical-specific metrics:
- Medical relevance scores
- Safety assessment scores
- Collaboration quality metrics
- Evidence-based reasoning scores

### 4. Real-time Monitoring
- Training loss per agent
- Memory usage and GPU utilization
- Training time per epoch
- Learning rate scheduling

## 🔧 Configuration Examples

### 1. Research Experiment Configuration
```bash
python train.py \
    --train_data data/medxpert_train.jsonl \
    --eval_data data/medxpert_eval.jsonl \
    --wandb_project "maporl-research" \
    --experiment_name "safety-weights-study" \
    --wandb_group "safety-experiments" \
    --wandb_tags "safety" "weights" "ablation" \
    --safety_penalty_weight 3.0 \
    --collaboration_bonus_weight 2.0
```

### 2. Production Training Configuration
```bash
python local_train_medxpert.py \
    --data-dir /data/medxpert \
    --output-dir /models/production \
    --wandb-project "maporl-production" \
    --wandb-name "production-v2.1" \
    --wandb-group "production-models" \
    --wandb-tags "production" "qwen" "4-agents" \
    --wandb-notes "Production model training with enhanced safety measures"
```

### 3. Debugging Configuration
```bash
python train.py \
    --train_data data/small_sample.jsonl \
    --wandb_project "maporl-debug" \
    --experiment_name "debug-session" \
    --wandb_group "debugging" \
    --wandb_tags "debug" "small-data" \
    --num_epochs 2 \
    --batch_size 1
```

## 📱 Monitoring Your Runs

### 1. W&B Dashboard
Visit your W&B dashboard to monitor:
- Real-time training metrics
- System resource usage
- Model performance comparisons
- Hyperparameter optimization

### 2. Command Line Monitoring
```bash
# List recent runs
wandb runs list

# Get run details
wandb runs show <run-id>

# Download artifacts
wandb artifacts download <artifact-name>
```

### 3. Programmatic Access
```python
import wandb

# Get run data
api = wandb.Api()
runs = api.runs("your-project")

# Download artifacts
artifact = api.artifact('your-project/training_config:latest')
artifact.download()
```

## 🚨 Troubleshooting

### Common Issues:

1. **W&B not logging**: Check internet connection and API key
2. **Too many runs**: Use `--wandb_group` to organize
3. **Large artifact sizes**: Use `.wandbignore` file
4. **Memory issues**: Reduce logging frequency

### Debug Commands:
```bash
# Check W&B status
wandb status

# Verify login
wandb whoami

# Test logging
wandb init --test
```

## 🎛️ Environment Variables Summary

```bash
# Required
export WANDB_API_KEY="your-api-key"

# Optional but recommended
export WANDB_PROJECT="maporl-medical"
export WANDB_ENTITY="your-username"
export WANDB_DIR="./outputs/wandb"

# Advanced options
export WANDB_LOG_MODEL="checkpoint"
export WANDB_WATCH="all"
export WANDB_SILENT="true"  # Reduce console output
```

## 📋 Best Practices

1. **Consistent Naming**: Use descriptive experiment names
2. **Grouping**: Organize related experiments with groups
3. **Tagging**: Use meaningful tags for filtering
4. **Notes**: Always add context with notes
5. **Artifacts**: Log important files as artifacts
6. **Cleanup**: Regularly review and clean old runs

## 🔗 Useful Links

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Python API](https://docs.wandb.ai/ref/python)
- [Best Practices Guide](https://docs.wandb.ai/guides/track/limits)

---

This integration provides comprehensive logging for your MAPoRL medical multi-agent training pipeline, enabling better experiment tracking, model comparison, and collaborative research. 