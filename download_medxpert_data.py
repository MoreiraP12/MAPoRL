#!/usr/bin/env python3
"""
Download MedXpert and Medical QA Data for Local Training
Creates JSONL files for multi-agent medical training
"""

import json
import requests
import os
import csv
from typing import List, Dict, Any
import logging
from urllib.parse import urljoin
import time
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalDataDownloader:
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def download_medquad_data(self) -> List[Dict[str, Any]]:
        """Download MedQuAD dataset from GitHub"""
        logger.info("📥 Downloading MedQuAD dataset...")
        
        # MedQuAD GitHub API to get file structure
        base_url = "https://api.github.com/repos/abachaa/MedQuAD/contents"
        
        medical_data = []
        
        try:
            # Get list of folders
            response = requests.get(base_url, timeout=30)
            response.raise_for_status()
            folders = response.json()
            
            # Sample folders to download (limit to avoid rate limiting)
            sample_folders = [
                "1_CancerGov_QA", 
                "4_MPlus_Health_Topics_QA",
                "5_NIDDK_QA",
                "6_NINDS_QA"
            ]
            
            for folder in folders:
                if folder['name'] in sample_folders and folder['type'] == 'dir':
                    logger.info(f"📂 Processing folder: {folder['name']}")
                    
                    # Get files in folder
                    folder_url = folder['url']
                    folder_response = requests.get(folder_url, timeout=30)
                    folder_response.raise_for_status()
                    files = folder_response.json()
                    
                    # Process first few XML files
                    xml_files = [f for f in files if f['name'].endswith('.xml')][:5]
                    
                    for xml_file in xml_files:
                        try:
                            # Download XML content
                            xml_response = requests.get(xml_file['download_url'], timeout=30)
                            xml_response.raise_for_status()
                            
                            # Basic XML parsing (simplified)
                            xml_content = xml_response.text
                            
                            # Extract Q&A pairs using simple string parsing
                            qa_pairs = self._parse_medquad_xml(xml_content, folder['name'])
                            medical_data.extend(qa_pairs)
                            
                            time.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to process {xml_file['name']}: {e}")
                            continue
                    
                    time.sleep(2)  # Rate limiting between folders
                    
        except Exception as e:
            logger.error(f"❌ Failed to download MedQuAD data: {e}")
            
        logger.info(f"✅ Downloaded {len(medical_data)} MedQuAD samples")
        return medical_data
    
    def _parse_medquad_xml(self, xml_content: str, category: str) -> List[Dict[str, Any]]:
        """Simple XML parsing for MedQuAD format"""
        qa_pairs = []
        
        try:
            # Simple regex-like parsing for Q&A pairs
            lines = xml_content.split('\n')
            current_question = ""
            current_answer = ""
            in_question = False
            in_answer = False
            
            for line in lines:
                line = line.strip()
                
                if '<Question>' in line or '<question>' in line:
                    in_question = True
                    current_question = line.replace('<Question>', '').replace('<question>', '').strip()
                elif '</Question>' in line or '</question>' in line:
                    in_question = False
                elif '<Answer>' in line or '<answer>' in line:
                    in_answer = True
                    current_answer = line.replace('<Answer>', '').replace('<answer>', '').strip()
                elif '</Answer>' in line or '</answer>' in line:
                    in_answer = False
                    
                    # Save Q&A pair
                    if current_question and current_answer:
                        qa_pairs.append({
                            "question": current_question,
                            "answer": current_answer,
                            "category": category.replace('_QA', '').lower(),
                            "source": "MedQuAD",
                            "difficulty": "medium"
                        })
                    
                    current_question = ""
                    current_answer = ""
                elif in_question:
                    current_question += " " + line
                elif in_answer:
                    current_answer += " " + line
                    
        except Exception as e:
            logger.warning(f"⚠️ XML parsing error: {e}")
            
        return qa_pairs
    
    def create_synthetic_medxpert_data(self) -> List[Dict[str, Any]]:
        """Create synthetic MedXpert-style data for training"""
        logger.info("🧬 Creating synthetic MedXpert data...")
        
        synthetic_data = [
            {
                "question": "A 65-year-old patient presents with sudden onset severe headache, neck stiffness, and photophobia. What is the most likely diagnosis and immediate management?",
                "answer": "Most likely diagnosis is subarachnoid hemorrhage. Immediate management: 1) Stabilize airway, breathing, circulation 2) CT head without contrast 3) If CT negative, lumbar puncture 4) Neurosurgical consultation 5) Blood pressure control 6) Nimodipine for vasospasm prevention",
                "category": "neurology",
                "source": "synthetic",
                "difficulty": "high",
                "keywords": ["subarachnoid hemorrhage", "headache", "neck stiffness", "emergency"]
            },
            {
                "question": "What are the contraindications for thrombolytic therapy in acute stroke?",
                "answer": "Contraindications include: 1) Symptoms >4.5 hours 2) Recent surgery/trauma 3) Active bleeding 4) Severe hypertension >185/110 5) Anticoagulation with elevated INR 6) Platelet count <100k 7) Previous ICH 8) Large stroke on imaging",
                "category": "neurology",
                "source": "synthetic", 
                "difficulty": "medium",
                "keywords": ["thrombolytic", "stroke", "contraindications", "tPA"]
            },
            {
                "question": "A 30-year-old pregnant woman at 32 weeks gestation presents with severe epigastric pain, elevated blood pressure, and proteinuria. Management approach?",
                "answer": "Diagnosis: Preeclampsia with severe features. Management: 1) Antihypertensive therapy (labetalol/hydralazine) 2) Magnesium sulfate for seizure prophylaxis 3) Corticosteroids for fetal lung maturity 4) Continuous fetal monitoring 5) Delivery planning - may need immediate delivery if severe",
                "category": "obstetrics",
                "source": "synthetic",
                "difficulty": "high",
                "keywords": ["preeclampsia", "pregnancy", "hypertension", "emergency"]
            },
            {
                "question": "What is the diagnostic approach for a patient with suspected pulmonary embolism?",
                "answer": "Diagnostic approach: 1) Assess clinical probability (Wells score) 2) D-dimer if low probability 3) CT pulmonary angiogram (CTPA) if high probability or positive D-dimer 4) Consider V/Q scan if contrast contraindicated 5) Echocardiogram if massive PE suspected",
                "category": "pulmonology",
                "source": "synthetic",
                "difficulty": "medium",
                "keywords": ["pulmonary embolism", "Wells score", "D-dimer", "CTPA"]
            },
            {
                "question": "A 65-year-old diabetic presents with foot ulcer and surrounding erythema. Approach to management?",
                "answer": "Management approach: 1) Assess vascular supply (ABI, pulse exam) 2) Wound culture and antibiotic therapy 3) Glycemic control optimization 4) Offloading/pressure relief 5) Wound debridement if indicated 6) Vascular surgery consultation if ischemic 7) Diabetes education",
                "category": "endocrinology",
                "source": "synthetic",
                "difficulty": "medium",
                "keywords": ["diabetic foot", "ulcer", "infection", "wound care"]
            },
            {
                "question": "A 45-year-old patient presents with chest pain, diaphoresis, and ST-elevation on ECG. Initial management?",
                "answer": "STEMI management: 1) Aspirin 325mg immediately 2) Clopidogrel loading dose 3) Atorvastatin 4) Beta-blocker if stable 5) Urgent cardiology consultation for primary PCI 6) If PCI unavailable, consider thrombolysis 7) Monitor for complications",
                "category": "cardiology",
                "source": "synthetic",
                "difficulty": "high",
                "keywords": ["STEMI", "myocardial infarction", "primary PCI", "emergency"]
            },
            {
                "question": "What is the workup for a patient presenting with jaundice?",
                "answer": "Jaundice workup: 1) History and physical exam 2) LFTs, bilirubin fractionation 3) CBC with peripheral smear 4) Hepatitis serologies 5) Ultrasound abdomen 6) Based on findings: MRCP/ERCP, CT, or liver biopsy 7) Consider drug-induced causes",
                "category": "gastroenterology",
                "source": "synthetic",
                "difficulty": "medium",
                "keywords": ["jaundice", "hepatitis", "bilirubin", "liver function"]
            },
            {
                "question": "A 25-year-old presents with fever, neck stiffness, and altered mental status. Emergency management?",
                "answer": "Suspected meningitis: 1) Immediate empirical antibiotics (do not delay for LP) 2) Dexamethasone if bacterial suspected 3) Blood cultures before antibiotics 4) Lumbar puncture if no contraindications 5) CT head if focal signs 6) Isolation precautions 7) Close contact prophylaxis",
                "category": "infectious_disease",
                "source": "synthetic",
                "difficulty": "high",
                "keywords": ["meningitis", "antibiotics", "lumbar puncture", "emergency"]
            },
            {
                "question": "What are the risk factors and prevention strategies for venous thromboembolism?",
                "answer": "VTE risk factors: Age >40, surgery, immobilization, cancer, pregnancy, OCP use, obesity, smoking. Prevention: 1) Early mobilization 2) Mechanical prophylaxis (compression stockings, pneumatic devices) 3) Pharmacologic prophylaxis (heparin, LMWH) 4) Risk stratification tools 5) Duration based on risk",
                "category": "hematology",
                "source": "synthetic",
                "difficulty": "medium",
                "keywords": ["VTE", "deep vein thrombosis", "pulmonary embolism", "prevention"]
            },
            {
                "question": "A 70-year-old patient with dementia presents with behavioral changes and agitation. Management approach?",
                "answer": "Dementia behavioral management: 1) Identify triggers and underlying causes 2) Non-pharmacologic interventions first 3) Optimize environment 4) Treat underlying medical conditions 5) If severe, consider low-dose antipsychotics 6) Regular reassessment 7) Family education and support",
                "category": "psychiatry",
                "source": "synthetic",
                "difficulty": "medium",
                "keywords": ["dementia", "behavioral changes", "agitation", "elderly"]
            }
        ]
        
        logger.info(f"✅ Created {len(synthetic_data)} synthetic samples")
        return synthetic_data
    
    def download_medtext_data(self) -> List[Dict[str, Any]]:
        """Download MedText dataset from HuggingFace"""
        logger.info("📥 Downloading MedText dataset...")
        
        try:
            # Try to use datasets library if available
            try:
                from datasets import load_dataset
                dataset = load_dataset("BI55/MedText", split="train")
                
                medtext_data = []
                for item in dataset:
                    medtext_data.append({
                        "question": item["Prompt"],
                        "answer": item["Completion"],
                        "category": "general_medicine",
                        "source": "MedText",
                        "difficulty": "medium"
                    })
                    
                logger.info(f"✅ Downloaded {len(medtext_data)} MedText samples")
                return medtext_data[:100]  # Limit to 100 samples
                
            except ImportError:
                logger.warning("⚠️ datasets library not available, creating sample data")
                return []
                
        except Exception as e:
            logger.error(f"❌ Failed to download MedText data: {e}")
            return []
    
    def save_as_jsonl(self, data: List[Dict[str, Any]], filename: str):
        """Save data as JSONL file"""
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        logger.info(f"💾 Saved {len(data)} samples to {filepath}")
    
    def create_training_validation_split(self, data: List[Dict[str, Any]], split_ratio: float = 0.8):
        """Split data into training and validation sets"""
        import random
        random.shuffle(data)
        
        split_idx = int(len(data) * split_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        return train_data, val_data
    
    def download_all_data(self):
        """Download and combine all medical datasets"""
        logger.info("🚀 Starting medical data download...")
        
        all_data = []
        
        # 1. Download MedQuAD data
        medquad_data = self.download_medquad_data()
        all_data.extend(medquad_data)
        
        # 2. Create synthetic MedXpert data
        synthetic_data = self.create_synthetic_medxpert_data()
        all_data.extend(synthetic_data)
        
        # 3. Try to download MedText data
        medtext_data = self.download_medtext_data()
        all_data.extend(medtext_data)
        
        # 4. Deduplicate and clean
        logger.info("🧹 Cleaning and deduplicating data...")
        seen_questions = set()
        clean_data = []
        
        for item in all_data:
            question_hash = hash(item['question'].lower().strip())
            if question_hash not in seen_questions:
                seen_questions.add(question_hash)
                clean_data.append(item)
        
        logger.info(f"📊 Total clean samples: {len(clean_data)}")
        
        # 5. Split into train/validation
        train_data, val_data = self.create_training_validation_split(clean_data)
        
        # 6. Save as JSONL files
        self.save_as_jsonl(train_data, "medxpert_train.jsonl")
        self.save_as_jsonl(val_data, "medxpert_validation.jsonl")
        self.save_as_jsonl(clean_data, "medxpert_complete.jsonl")
        
        # 7. Create summary
        summary = {
            "total_samples": len(clean_data),
            "training_samples": len(train_data),
            "validation_samples": len(val_data),
            "categories": list(set(item.get('category', 'unknown') for item in clean_data)),
            "sources": list(set(item.get('source', 'unknown') for item in clean_data)),
            "difficulty_levels": list(set(item.get('difficulty', 'unknown') for item in clean_data))
        }
        
        with open(os.path.join(self.output_dir, "dataset_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✅ Medical data download complete!")
        logger.info(f"📁 Files saved in: {self.output_dir}/")
        logger.info(f"📊 Summary: {summary}")
        
        return clean_data

def main():
    """Main function"""
    print("🏥 Medical Dataset Downloader")
    print("=" * 50)
    
    downloader = MedicalDataDownloader()
    
    try:
        data = downloader.download_all_data()
        
        print("\n🎉 Download Summary:")
        print(f"📊 Total samples: {len(data)}")
        print(f"📁 Output directory: {downloader.output_dir}/")
        print("\n📋 Files created:")
        print("  - medxpert_train.jsonl (training data)")
        print("  - medxpert_validation.jsonl (validation data)")
        print("  - medxpert_complete.jsonl (all data)")
        print("  - dataset_summary.json (dataset statistics)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        raise

if __name__ == "__main__":
    main() 