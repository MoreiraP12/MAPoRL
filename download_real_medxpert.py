#!/usr/bin/env python3
"""
Download Real MedXpertQA Text Dataset from Hugging Face
This script downloads the actual MedXpertQA dataset, not synthetic data
"""

import os
import json
import logging
from datasets import load_dataset
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_medxpert_text_dataset(output_dir: str = "data"):
    """
    Download the real MedXpertQA text-only dataset from Hugging Face
    """
    logger.info("🩺 Downloading MedXpertQA text dataset from Hugging Face...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load the dataset from Hugging Face
        # Using the 'Text' subset specifically as requested
        dataset = load_dataset(
            "TsinghuaC3I/MedXpertQA", 
            "Text",  # Text-only subset 
            trust_remote_code=True
        )
        
        logger.info(f"✅ Successfully loaded MedXpertQA dataset")
        logger.info(f"📊 Dataset info: {dataset}")
        
        # Process and save the data
        all_data = []
        
        # Process test set
        if 'test' in dataset:
            test_data = dataset['test']
            logger.info(f"📋 Test set: {len(test_data)} samples")
            
            for item in test_data:
                processed_item = {
                    'id': item['id'],
                    'question': item['question'],
                    'options': item['options'],
                    'label': item['label'],
                    'medical_task': item['medical_task'],
                    'body_system': item['body_system'],
                    'question_type': item['question_type'],
                    'source': 'MedXpertQA_HF',
                    'subset': 'text'
                }
                all_data.append(processed_item)
        
        # Process dev set
        if 'dev' in dataset:
            dev_data = dataset['dev']
            logger.info(f"📋 Dev set: {len(dev_data)} samples")
            
            for item in dev_data:
                processed_item = {
                    'id': item['id'],
                    'question': item['question'],
                    'options': item['options'],
                    'label': item['label'],
                    'medical_task': item['medical_task'],
                    'body_system': item['body_system'],
                    'question_type': item['question_type'],
                    'source': 'MedXpertQA_HF',
                    'subset': 'text'
                }
                all_data.append(processed_item)
        
        # Save the complete dataset
        output_file = os.path.join(output_dir, "medxpert_text_real.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"💾 Saved {len(all_data)} samples to {output_file}")
        
        # Create separate files for test and dev
        if 'test' in dataset:
            test_file = os.path.join(output_dir, "medxpert_text_test.jsonl")
            with open(test_file, 'w', encoding='utf-8') as f:
                for item in dataset['test']:
                    processed_item = {
                        'id': item['id'],
                        'question': item['question'],
                        'options': item['options'],
                        'label': item['label'],
                        'medical_task': item['medical_task'],
                        'body_system': item['body_system'],
                        'question_type': item['question_type'],
                        'source': 'MedXpertQA_HF',
                        'subset': 'text'
                    }
                    f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
            logger.info(f"💾 Saved test set to {test_file}")
        
        if 'dev' in dataset:
            dev_file = os.path.join(output_dir, "medxpert_text_dev.jsonl")
            with open(dev_file, 'w', encoding='utf-8') as f:
                for item in dataset['dev']:
                    processed_item = {
                        'id': item['id'],
                        'question': item['question'],
                        'options': item['options'],
                        'label': item['label'],
                        'medical_task': item['medical_task'],
                        'body_system': item['body_system'],
                        'question_type': item['question_type'],
                        'source': 'MedXpertQA_HF',
                        'subset': 'text'
                    }
                    f.write(json.dumps(processed_item, ensure_ascii=False) + '\n')
            logger.info(f"💾 Saved dev set to {dev_file}")
        
        # Print dataset statistics
        print_dataset_stats(all_data)
        
        return all_data
        
    except Exception as e:
        logger.error(f"❌ Failed to download MedXpertQA dataset: {e}")
        raise

def print_dataset_stats(data: List[Dict[str, Any]]):
    """Print statistics about the downloaded dataset"""
    logger.info("📊 Dataset Statistics:")
    logger.info(f"   Total samples: {len(data)}")
    
    # Count by medical task
    task_counts = {}
    for item in data:
        task = item.get('medical_task', 'Unknown')
        task_counts[task] = task_counts.get(task, 0) + 1
    
    logger.info("   Medical Tasks:")
    for task, count in task_counts.items():
        logger.info(f"     {task}: {count}")
    
    # Count by body system
    system_counts = {}
    for item in data:
        system = item.get('body_system', 'Unknown')
        system_counts[system] = system_counts.get(system, 0) + 1
    
    logger.info("   Body Systems:")
    for system, count in system_counts.items():
        logger.info(f"     {system}: {count}")
    
    # Count by question type
    type_counts = {}
    for item in data:
        qtype = item.get('question_type', 'Unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    logger.info("   Question Types:")
    for qtype, count in type_counts.items():
        logger.info(f"     {qtype}: {count}")

def remove_synthetic_data(data_dir: str = "data"):
    """Remove synthetic data files"""
    synthetic_files = [
        "medxpert_complete.jsonl",
        "medxpert_train.jsonl", 
        "medxpert_validation.jsonl"
    ]
    
    for filename in synthetic_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            logger.info(f"🗑️  Removing synthetic data file: {filename}")
            os.remove(filepath)

def main():
    """Main function to download real MedXpertQA data"""
    logger.info("🚀 Starting MedXpertQA real data download...")
    
    # Remove synthetic data first
    remove_synthetic_data()
    
    # Download real data
    data = download_medxpert_text_dataset()
    
    logger.info("✅ MedXpertQA real data download completed!")
    logger.info(f"📁 Data saved in: data/")
    logger.info("📝 Files created:")
    logger.info("   - medxpert_text_real.jsonl (complete dataset)")
    logger.info("   - medxpert_text_test.jsonl (test set)")
    logger.info("   - medxpert_text_dev.jsonl (dev set)")

if __name__ == "__main__":
    main() 