#!/usr/bin/env python3
"""
Local Medical Multi-Agent Training Script
Train Qwen3-0.6B models on MedXpert data locally without SageMaker
"""

import os
import json
import torch
import logging
import wandb
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from datasets import Dataset
import numpy as np
import sys
# Add root directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.model_config_qwen import get_qwen_config, AGENT_CONFIGS, load_qwen_model
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalMedicalTrainer:
    def __init__(self, data_dir: str = "data", output_dir: str = "outputs", wandb_config: Dict[str, Any] = None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = "Qwen/Qwen3-0.6B"
        self.wandb_config = wandb_config or {}
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        
        # Agent configurations - Use single GPU to avoid quantization device conflicts
        # TODO: Re-enable multi-GPU after fixing quantization device mapping
        self.agents = {
            "planner": {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "role": "medical_planner"},
            "researcher": {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "role": "medical_researcher"}, 
            "analyst": {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "role": "medical_analyst"},
            "reporter": {"device": "cuda:0" if torch.cuda.is_available() else "cpu", "role": "medical_reporter"}
        }
        
        # Medical reward weights
        self.reward_weights = {
            "accuracy": 0.25,
            "medical_relevance": 0.20,
            "safety": 0.20,
            "collaboration": 0.15,
            "evidence_quality": 0.10,
            "clinical_reasoning": 0.10
        }
        
        # Initialize W&B
        self.setup_wandb()
        
    def setup_wandb(self):
        """Setup Weights & Biases logging."""
        if self.wandb_config.get("disable_wandb", False):
            logger.info("W&B logging disabled")
            return
            
        try:
            wandb.init(
                project=self.wandb_config.get("project", "maporl-medxpert-local"),
                name=self.wandb_config.get("name", f"local-medxpert-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
                group=self.wandb_config.get("group", "local-training"),
                tags=self.wandb_config.get("tags", ["medxpert", "qwen", "local", "multi-agent"]),
                notes=self.wandb_config.get("notes", "Local MedXpert multi-agent training"),
                config={
                    "model_name": self.model_name,
                    "agents": list(self.agents.keys()),
                    "reward_weights": self.reward_weights,
                    "output_dir": self.output_dir,
                    "data_dir": self.data_dir,
                    **self.wandb_config.get("extra_config", {})
                }
            )
            
            # Log system info
            wandb.log({
                "system/gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "system/cuda_available": torch.cuda.is_available(),
                "system/pytorch_version": torch.__version__,
            })
            
            logger.info(f"W&B initialized: {wandb.run.url}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            logger.info("Continuing without W&B logging")
    
    def load_jsonl_data(self, filepath: str) -> Dataset:
        """Load JSONL data into Hugging Face Dataset"""
        logger.info(f"📥 Loading data from {filepath}")
        
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        
        # Format for training
        formatted_data = []
        for item in data:
            # Create input-output pairs for medical QA
            input_text = f"Medical Question: {item['question']}"
            output_text = item['answer']
            
            formatted_data.append({
                "input_ids": input_text,
                "labels": output_text,
                "category": item.get('category', 'general'),
                "difficulty": item.get('difficulty', 'medium'),
                "source": item.get('source', 'unknown')
            })
        
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"✅ Loaded {len(dataset)} samples")
        
        # Log dataset stats to W&B
        if not self.wandb_config.get("disable_wandb", False):
            categories = [item['category'] for item in formatted_data]
            category_counts = {cat: categories.count(cat) for cat in set(categories)}
            
            wandb.log({
                "data/total_samples": len(dataset),
                "data/categories": category_counts,
                "data/source_file": filepath
            })
        
        return dataset
    
    def setup_model_and_tokenizer(self, agent_type: str = "base"):
        """Setup Qwen3-0.6B model and tokenizer for specific agent with PEFT LoRA"""
        device = self.agents.get(agent_type, {}).get("device", "cpu")
        
        logger.info(f"🤖 Setting up {agent_type} agent on {device}")
        
        # Quantization config for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if torch.cuda.is_available() else None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
        }
        
        if torch.cuda.is_available():
            # Set the current device before loading the model
            if device.startswith("cuda"):
                device_id = int(device.split(":")[-1])
                torch.cuda.set_device(device_id)
            
            model_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": {"": torch.cuda.current_device()},
                "quantization_config": quantization_config,
            })
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Prepare model for k-bit training (required for quantized models)
        if quantization_config is not None:
            model = prepare_model_for_kbit_training(model)
        
        # LoRA Configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # rank
            lora_alpha=32,  # alpha scaling parameter
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"💡 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        logger.info(f"✅ {agent_type} agent loaded on {device}")
        return model, tokenizer
    
    def preprocess_function(self, examples, tokenizer):
        """Preprocess examples for training"""
        # Combine input and output for causal language modeling
        combined_texts = []
        for i in range(len(examples["input_ids"])):
            # Format as instruction-following with clear separation
            text = f"Question: {examples['input_ids'][i]}\nAnswer: {examples['labels'][i]}{tokenizer.eos_token}"
            combined_texts.append(text)
        
        # Tokenize the combined texts
        model_inputs = tokenizer(
            combined_texts,
            truncation=True,
            padding=True,  # Enable padding to ensure consistent lengths
            max_length=384,  # Reduced from 512 to save memory
            return_tensors=None  # Return lists, not tensors
        )
        
        # Ensure labels are proper lists of integers, not nested lists
        labels = []
        for input_ids in model_inputs["input_ids"]:
            # Create a copy of input_ids for labels
            if isinstance(input_ids, list):
                labels.append(input_ids.copy())
            else:
                labels.append(input_ids.tolist())
        
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }
    
    def calculate_medical_reward(self, prediction: str, ground_truth: str, category: str) -> float:
        """Calculate medical-specific reward for MAPoRL training"""
        
        # 1. Accuracy reward (token overlap)
        pred_tokens = set(prediction.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if truth_tokens:
            accuracy_reward = len(pred_tokens.intersection(truth_tokens)) / len(truth_tokens)
        else:
            accuracy_reward = 0.0
        
        # 2. Medical relevance reward
        medical_terms = [
            "diagnosis", "treatment", "patient", "symptom", "condition",
            "medication", "therapy", "clinical", "medical", "disease",
            "syndrome", "pathology", "etiology", "prognosis", "differential"
        ]
        
        pred_lower = prediction.lower()
        medical_relevance = sum(1 for term in medical_terms if term in pred_lower) / len(medical_terms)
        
        # 3. Safety reward (penalize overconfident language)
        unsafe_phrases = ["definitely", "certainly", "100% sure", "impossible", "never", "always"]
        safe_phrases = ["consider", "may", "could", "suggest", "recommend", "consult"]
        
        unsafe_count = sum(1 for phrase in unsafe_phrases if phrase in pred_lower)
        safe_count = sum(1 for phrase in safe_phrases if phrase in pred_lower)
        
        safety_reward = max(0.0, 1.0 - 0.2 * unsafe_count + 0.1 * safe_count)
        
        # 4. Category-specific bonus
        category_bonus = 0.1 if category.lower() in pred_lower else 0.0
        
        # Calculate weighted total
        total_reward = (
            self.reward_weights["accuracy"] * accuracy_reward +
            self.reward_weights["medical_relevance"] * medical_relevance +
            self.reward_weights["safety"] * safety_reward +
            category_bonus
        )
        
        return min(1.0, max(0.0, total_reward))
    
    def create_training_args(self, agent_type: str) -> TrainingArguments:
        """Create training arguments optimized for local training"""
        
        return TrainingArguments(
            output_dir=f"{self.output_dir}/models/{agent_type}",
            
            # Training configuration - Reduced for memory efficiency
            num_train_epochs=2,
            per_device_train_batch_size=1,  # Reduced to minimize memory usage
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            
            # Optimization (higher learning rate for LoRA)
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=50,
            
            # Memory optimization - More aggressive settings
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            
            # Disable DataParallel to avoid multi-GPU issues
            dataloader_drop_last=True,
            
            # Logging and saving - Reduced frequency
            logging_steps=25,
            save_steps=200,
            eval_steps=200,
            save_total_limit=1,  # Keep only best model
            eval_strategy="steps",
            
            # W&B integration
            report_to="wandb" if not self.wandb_config.get("disable_wandb", False) else [],
            run_name=f"{agent_type}-agent" if not self.wandb_config.get("disable_wandb", False) else None,
            
            # Load best model at end
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            
            # Explicitly disable distributed training
            local_rank=-1,
        )
    
    def train_agent(self, agent_type: str, train_dataset: Dataset, eval_dataset: Dataset) -> Dict[str, Any]:
        """Train a specific agent"""
        logger.info(f"🏋️ Training {agent_type} agent...")
        
        # Get device for this agent
        device = self.agents[agent_type]["device"]
        
        # Setup model and tokenizer
        model, tokenizer = self.setup_model_and_tokenizer(agent_type)
        
        # Preprocess datasets
        logger.info(f"📊 Preprocessing {len(train_dataset)} training samples...")
        train_dataset = train_dataset.map(
            lambda x: self.preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Processing train data"
        )
        
        logger.info(f"📊 Preprocessing {len(eval_dataset)} evaluation samples...")
        eval_dataset = eval_dataset.map(
            lambda x: self.preprocess_function(x, tokenizer), 
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Processing eval data"
        )
        
        logger.info(f"✅ Preprocessed datasets: train={len(train_dataset)}, eval={len(eval_dataset)}")
        
        # Debug: Check the first sample
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logger.info(f"🔍 Sample data types: input_ids={type(sample['input_ids'])}, labels={type(sample['labels'])}")
            logger.info(f"🔍 Sample lengths: input_ids={len(sample['input_ids'])}, labels={len(sample['labels'])}")
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Training arguments
        training_args = self.create_training_args(agent_type)
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        
        # Set CUDA device for this agent and disable multi-GPU
        if torch.cuda.is_available() and device.startswith("cuda"):
            device_id = int(device.split(":")[-1])
            torch.cuda.set_device(device_id)
            # Set environment variables to force single GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # Train
        logger.info(f"🚀 Starting training for {agent_type} agent...")
        try:
            train_result = trainer.train()
        except RuntimeError as e:
            if "CUDA" in str(e):
                logger.error(f"CUDA error during training: {e}")
                # Try to recover
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                raise
            else:
                raise
        
        # Save model (PEFT adapter + tokenizer)
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save the PEFT adapter separately as well for easier loading
        model.save_pretrained(training_args.output_dir)
        
        # Aggressive memory cleanup
        del trainer
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Force garbage collection
            import gc
            gc.collect()
        
        # Save training stats
        training_stats = {
            "agent_type": agent_type,
            "model_name": self.model_name,
            "train_loss": train_result.training_loss,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "training_time": train_result.metrics.get("train_runtime", 0),
            "device": self.agents[agent_type]["device"]
        }
        
        # Log training stats to W&B
        if not self.wandb_config.get("disable_wandb", False):
            wandb.log({
                f"agent_{agent_type}/final_train_loss": train_result.training_loss,
                f"agent_{agent_type}/training_time": train_result.metrics.get("train_runtime", 0),
                f"agent_{agent_type}/train_samples": len(train_dataset),
                f"agent_{agent_type}/eval_samples": len(eval_dataset),
                f"agent_{agent_type}/device": self.agents[agent_type]["device"],
                f"agents_completed": agent_type
            })
        
        with open(f"{training_args.output_dir}/training_stats.json", 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        logger.info(f"✅ {agent_type} agent training completed")
        return training_stats
    
    def train_all_agents(self):
        """Train all 4 medical agents"""
        logger.info("🚀 Starting Medical Multi-Agent Training")
        
        # Load datasets
        train_file = os.path.join(self.data_dir, "medxpert_train.jsonl")
        val_file = os.path.join(self.data_dir, "medxpert_validation.jsonl")
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data not found at {train_file}. Run download_medxpert_data.py first.")
        
        train_dataset = self.load_jsonl_data(train_file)
        eval_dataset = self.load_jsonl_data(val_file) if os.path.exists(val_file) else train_dataset.train_test_split(test_size=0.2)["test"]
        
        # Training results
        all_results = []
        
        # Train each agent
        for agent_type in self.agents.keys():
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training {agent_type.upper()} Agent")
                logger.info(f"{'='*50}")
                
                result = self.train_agent(agent_type, train_dataset, eval_dataset)
                all_results.append(result)
                
                # Aggressive cleanup between agents
                if torch.cuda.is_available():
                    # Check memory before cleanup
                    allocated_before = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved_before = torch.cuda.memory_reserved() / 1024**3   # GB
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Force garbage collection
                    import gc
                    gc.collect()
                    
                    # Check memory after cleanup
                    allocated_after = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved_after = torch.cuda.memory_reserved() / 1024**3    # GB
                    
                    logger.info(f"🧹 Memory cleanup: {allocated_before:.2f}GB -> {allocated_after:.2f}GB allocated, "
                              f"{reserved_before:.2f}GB -> {reserved_after:.2f}GB reserved")
                    
                # Longer delay to ensure complete cleanup
                import time
                time.sleep(5)
                
                logger.info(f"✅ {agent_type} agent completed, moving to next agent")
                
            except Exception as e:
                logger.error(f"❌ Failed to train {agent_type} agent: {e}")
                continue
        
        # Save overall results
        overall_results = {
            "training_date": datetime.now().isoformat(),
            "model_name": self.model_name,
            "total_agents": len(all_results),
            "agents": all_results,
            "dataset_info": {
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "data_sources": train_file
            }
        }
        
        # Log final summary to W&B
        if not self.wandb_config.get("disable_wandb", False):
            # Calculate average training loss across agents
            avg_loss = sum(r.get("train_loss", 0) for r in all_results) / len(all_results) if all_results else 0
            total_training_time = sum(r.get("training_time", 0) for r in all_results)
            
            wandb.log({
                "summary/total_agents_trained": len(all_results),
                "summary/average_train_loss": avg_loss,
                "summary/total_training_time": total_training_time,
                "summary/train_samples": len(train_dataset),
                "summary/eval_samples": len(eval_dataset),
                "summary/training_date": datetime.now().isoformat()
            })
            
            # Create summary table
            agent_table = wandb.Table(
                columns=["Agent", "Train Loss", "Training Time", "Device"],
                data=[[r["agent_type"], r.get("train_loss", 0), r.get("training_time", 0), r.get("device", "unknown")] 
                      for r in all_results]
            )
            wandb.log({"summary/agent_performance": agent_table})
            
            # Save results as artifact
            artifact = wandb.Artifact("training_results", type="results")
            results_file = f"{self.output_dir}/training_results.json"
            with open(results_file, 'w') as f:
                json.dump(overall_results, f, indent=2)
            artifact.add_file(results_file)
            wandb.log_artifact(artifact)
        else:
            results_file = f"{self.output_dir}/training_results.json"
            with open(results_file, 'w') as f:
                json.dump(overall_results, f, indent=2)
        
        logger.info(f"\n🎉 Training Complete!")
        logger.info(f"📊 Results saved to: {results_file}")
        logger.info(f"📁 Models saved in: {self.output_dir}/models/")
        
        return overall_results
    
    def test_agents(self, test_questions: List[str] = None):
        """Test trained agents on sample questions"""
        if test_questions is None:
            test_questions = [
                "A 65-year-old patient presents with chest pain and shortness of breath. What are the key differential diagnoses?",
                "What are the contraindications for MRI in a patient with implanted devices?",
                "A 30-year-old presents with sudden severe headache. What is the emergency management?"
            ]
        
        logger.info("🧪 Testing trained agents...")
        
        results = {}
        
        for agent_type in self.agents.keys():
            model_path = f"{self.output_dir}/models/{agent_type}"
            
            if not os.path.exists(model_path):
                logger.warning(f"⚠️ No trained model found for {agent_type}")
                continue
            
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load base model with lighter quantization for inference
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Load and apply PEFT adapter
                model = PeftModel.from_pretrained(base_model, model_path)
                model = model.eval()  # Set to evaluation mode
                
                agent_results = []
                
                for question in test_questions:
                    input_text = f"Medical Question: {question}"
                    inputs = tokenizer(input_text, return_tensors="pt")
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            temperature=0.7,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.replace(input_text, "").strip()
                    
                    agent_results.append({
                        "question": question,
                        "response": response
                    })
                
                results[agent_type] = agent_results
                
                # Clear memory
                del model, base_model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"❌ Failed to test {agent_type}: {e}")
                continue
        
        # Save test results
        test_results_file = f"{self.output_dir}/test_results.json"
        with open(test_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✅ Testing complete. Results saved to: {test_results_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description="Train Medical Multi-Agent System Locally")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing JSONL data files")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for models and results")
    parser.add_argument("--test-only", action="store_true", help="Only run testing on existing models")
    
    # W&B arguments
    parser.add_argument("--wandb-project", type=str, default="maporl-medxpert-local", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, help="W&B run name")
    parser.add_argument("--wandb-group", type=str, default="local-training", help="W&B group name")
    parser.add_argument("--wandb-tags", type=str, nargs="+", default=["medxpert", "qwen", "local", "multi-agent"], help="W&B tags")
    parser.add_argument("--wandb-notes", type=str, help="W&B run notes")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    
    # Setup W&B configuration
    wandb_config = {
        "project": args.wandb_project,
        "name": args.wandb_name,
        "group": args.wandb_group,
        "tags": args.wandb_tags,
        "notes": args.wandb_notes,
        "disable_wandb": args.disable_wandb,
        "extra_config": {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "test_only": args.test_only,
            "gpu_count": torch.cuda.device_count(),
            "cuda_available": torch.cuda.is_available()
        }
    }
    
    print("🏥 Local Medical Multi-Agent Training")
    print("=" * 50)
    print(f"🤖 Model: Qwen3-0.6B")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🖥️ Available GPUs: {torch.cuda.device_count()}")
    print(f"📈 W&B logging: {'Disabled' if args.disable_wandb else 'Enabled'}")
    if not args.disable_wandb:
        print(f"📈 W&B project: {args.wandb_project}")
    print("=" * 50)
    
    trainer = LocalMedicalTrainer(args.data_dir, args.output_dir, wandb_config)
    
    try:
        if args.test_only:
            # Only run testing
            trainer.test_agents()
        else:
            # Full training pipeline
            training_results = trainer.train_all_agents()
            
            # Skip testing for now to focus on training
            # TODO: Re-enable testing after fixing model loading issues
            # trainer.test_agents()
            
            print("\n🎉 Complete Training Pipeline Finished!")
            print(f"📊 Trained {training_results['total_agents']} agents")
            print(f"📁 Check results in: {args.output_dir}/")
            
        # Finish W&B run
        if not args.disable_wandb:
            wandb.finish()
            
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if not args.disable_wandb:
            wandb.finish(exit_code=1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if not args.disable_wandb:
            wandb.finish(exit_code=1)
        raise

if __name__ == "__main__":
    main() 