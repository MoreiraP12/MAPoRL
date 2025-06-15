# MAPoRL: Multi-Agent Post-Co-Training for Collaborative Medical LLMs

This repository implements a multi-agent medical pipeline designed for medical benchmarks like MedXpert and OpenAI's Health Bench, with support for MAPoRL (Multi-Agent Post-co-training with Reinforcement Learning) training.

## 🏥 Overview

The system implements a collaborative multi-agent framework where specialized medical agents work together to answer medical questions:

- **🧠 Planner Agent**: Creates systematic investigation plans for medical questions
- **🔬 Researcher Agent**: Gathers evidence-based medical information and guidelines  
- **⚕️ Analyst Agent**: Provides clinical reasoning and differential analysis
- **📋 Reporter Agent**: Synthesizes findings into comprehensive medical reports

## 🎯 Key Features

- **Multi-Agent Collaboration**: Four specialized medical agents using LangGraph for state management
- **Medical-Focused Rewards**: Comprehensive reward system evaluating accuracy, safety, evidence quality, and collaboration
- **MAPoRL Training**: Multi-agent reinforcement learning training framework  
- **GPU Optimized**: Designed for 4 A10G GPUs with small models (117M-66M parameters)
- **Medical Safety**: Built-in safety checks and medical disclaimers
- **Benchmark Ready**: Compatible with MedXpert and OpenAI Health Bench

## 🏗️ Architecture

```
Medical Question
       ↓
   [Planner] → Creates investigation plan
       ↓
  [Researcher] → Gathers evidence & guidelines
       ↓
   [Analyst] → Clinical reasoning & analysis
       ↓
   [Reporter] → Final medical report
       ↓
   Medical Answer + Rewards
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd MAPoRL

# Install dependencies
pip install -r requirements.txt
```

### Demo

Run the demonstration to see the system in action:

```bash
python demo.py
```

This will demonstrate:
1. Multi-agent workflow with sample medical questions
2. Medical reward system evaluation
3. Training setup configuration

### Training

Train the multi-agent system using MAPoRL:

```bash
python train.py \
    --train_data data/medical_train.json \
    --eval_data data/medical_eval.json \
    --batch_size 2 \
    --num_epochs 10 \
    --learning_rate 1e-5 \
    --experiment_name "medical_maporl_v1"
```

## 📊 System Components

### Agents

1. **Medical Planner Agent** (`src/agents/planner_agent.py`)
   - Analyzes question type (diagnosis, treatment, medication)
   - Creates structured investigation plans
   - Sets collaboration strategy for team

2. **Medical Researcher Agent** (`src/agents/researcher_agent.py`)
   - Searches medical knowledge base
   - Evaluates evidence quality
   - References clinical guidelines

3. **Medical Analyst Agent** (`src/agents/analyst_agent.py`)
   - Performs differential diagnosis analysis
   - Identifies clinical patterns and red flags
   - Assesses treatment appropriateness

4. **Medical Reporter Agent** (`src/agents/reporter_agent.py`)
   - Synthesizes team findings
   - Generates structured medical reports
   - Validates final answers for safety

### Reward System

The medical reward system (`src/reward/medical_reward_system.py`) evaluates:

- **Accuracy** (25%): Semantic similarity to ground truth
- **Medical Relevance** (20%): Medical terminology and clinical reasoning
- **Safety** (20%): Avoids overconfident claims, includes disclaimers
- **Collaboration Quality** (15%): Agent interaction and complementary content
- **Evidence Quality** (10%): References to guidelines and studies
- **Clinical Reasoning** (10%): Structured thinking and diagnostic patterns

### Training Framework

MAPoRL training (`src/training/maporl_trainer.py`) implements:

- Multi-agent PPO with shared experiences
- Medical-specific reward optimization
- Collaboration bonus and safety penalties
- Checkpoint saving and evaluation metrics

## 🔧 Configuration

### Model Configuration

Models are configured for 4 A10G GPUs in `src/config/model_config.py`:

```python
MEDICAL_AGENT_CONFIGS = {
    "planner": ModelConfig(
        model_name="microsoft/DialoGPT-small",  # 117M parameters
        device_map="cuda:0",
        load_in_8bit=True
    ),
    "researcher": ModelConfig(
        model_name="distilbert-base-uncased",  # 66M parameters  
        device_map="cuda:1",
        load_in_8bit=True
    ),
    # ... additional agents
}
```

### Training Configuration

```python
config = MAPoRLConfig(
    learning_rate=1e-5,
    batch_size=2,
    num_epochs=10,
    max_rounds_per_episode=3,
    safety_penalty_weight=2.0,
    collaboration_bonus_weight=1.5
)
```

## 📁 Directory Structure

```
MAPoRL/
├── src/
│   ├── agents/              # Medical agents
│   │   ├── base_agent.py    # Base agent class
│   │   ├── planner_agent.py # Planning agent
│   │   ├── researcher_agent.py # Research agent
│   │   ├── analyst_agent.py # Analysis agent
│   │   └── reporter_agent.py # Reporting agent
│   ├── workflow/            # LangGraph workflow
│   │   └── medical_workflow.py
│   ├── reward/              # Reward system
│   │   └── medical_reward_system.py
│   ├── training/            # MAPoRL training
│   │   └── maporl_trainer.py
│   └── config/              # Configuration
│       └── model_config.py
├── demo.py                  # Demonstration script
├── train.py                # Training script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🏥 Medical Benchmarks

### Supported Benchmarks

- **MedXpert**: Medical expert-level question answering
- **OpenAI Health Bench**: Healthcare reasoning and diagnosis

### Data Format

Training data should be in JSON format:

```json
[
  {
    "question": "What are the symptoms of hypertension?",
    "context": "A 45-year-old patient presents with elevated BP readings.",
    "answer": "Hypertension often presents with headaches, dizziness..."
  }
]
```

## ⚕️ Safety Features

- **Overconfidence Detection**: Flags absolute medical claims
- **Disclaimer Requirements**: Ensures consultation recommendations
- **Safety Scoring**: Penalizes dangerous medical advice
- **Evidence Validation**: Promotes evidence-based responses

## 📈 Performance Metrics

The system tracks:

- **Medical Accuracy**: Semantic similarity to correct answers
- **Safety Score**: Absence of harmful medical advice
- **Collaboration Quality**: Agent interaction effectiveness
- **Evidence Quality**: Clinical guideline adherence
- **Clinical Reasoning**: Structured diagnostic thinking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure medical safety compliance
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the MAPoRL framework from the research paper
- Medical knowledge bases and clinical guidelines
- LangGraph for multi-agent orchestration
- Transformers library for model implementations

## 📞 Support

For questions about medical applications or training:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the demo script for examples

---

⚠️ **Medical Disclaimer**: This system is for research purposes only and should not be used for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.