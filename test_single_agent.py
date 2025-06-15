#!/usr/bin/env python3
"""
Simple test script to debug training issues with a single agent
"""

import os
import json
import torch
import logging
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
    TaskType
)
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_sample_data(filepath: str) -> Dataset:
    """Load a small sample of data for testing"""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 10:  # Only load first 10 samples for testing
                break
            item = json.loads(line.strip())
            data.append({
                "input_ids": f"Medical Question: {item['question']}",
                "labels": item['answer'],
                "category": item.get('category', 'general')
            })
    
    return Dataset.from_list(data)

def preprocess_function(examples, tokenizer):
    """Simple preprocessing function"""
    combined_texts = []
    for i in range(len(examples["input_ids"])):
        text = f"Question: {examples['input_ids'][i]}\nAnswer: {examples['labels'][i]}{tokenizer.eos_token}"
        combined_texts.append(text)
    
    model_inputs = tokenizer(
        combined_texts,
        truncation=True,
        padding=True,  # Enable padding here
        max_length=256,  # Shorter for testing
        return_tensors=None
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

def main():
    print("🧪 Testing Single Agent Training")
    print("=" * 40)
    
    # Load sample data
    train_dataset = load_sample_data("data/medxpert_train.jsonl")
    print(f"✅ Loaded {len(train_dataset)} samples")
    
    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # Setup model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    
    # Quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ) if torch.cuda.is_available() else None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
    }
    
    if torch.cuda.is_available():
        model_kwargs.update({
            "torch_dtype": torch.float16,
            "device_map": {"": torch.cuda.current_device()},
            "quantization_config": quantization_config,
        })
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Prepare for PEFT
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # Smaller rank for testing
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"💡 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Preprocess dataset
    print("📊 Preprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Check sample
    sample = train_dataset[0]
    print(f"🔍 Sample input_ids type: {type(sample['input_ids'])}")
    print(f"🔍 Sample input_ids length: {len(sample['input_ids'])}")
    print(f"🔍 Sample labels type: {type(sample['labels'])}")
    print(f"🔍 Sample labels length: {len(sample['labels'])}")
    
    # Check if all samples have the same length (due to padding)
    lengths = [len(item['input_ids']) for item in train_dataset]
    print(f"🔍 All lengths: {set(lengths)}")
    print(f"🔍 Should all be the same due to padding: {len(set(lengths)) == 1}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=100,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to=[],  # Disable wandb for testing
        run_name="test_single_agent",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    
    print("🚀 Starting training...")
    try:
        train_result = trainer.train()
        print("✅ Training completed successfully!")
        print(f"📊 Final loss: {train_result.training_loss}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 