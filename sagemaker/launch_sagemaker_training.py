#!/usr/bin/env python3
"""
SageMaker Training Job Launcher for Medical Multi-Agent MAPoRL
This script demonstrates how to launch the training job on SageMaker
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
import json
import os
from datetime import datetime
import time

def create_sagemaker_training_job():
    """Create and launch SageMaker training job for MedXpert MAPoRL."""
    
    print("🏥 Setting up SageMaker Training Job for Medical Multi-Agent MAPoRL")
    
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()  # or provide your role ARN
    
    # Configuration
    job_name = f"maporl-medxpert-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # Training configuration
    training_config = {
        "max_train_samples": 1000,
        "max_eval_samples": 100,
        "epochs": 10,
        "learning_rate": 3e-5
    }
    
    print(f"📊 Job Name: {job_name}")
    print(f"⚙️ Training Config: {json.dumps(training_config, indent=2)}")
    
    # Create PyTorch estimator
    estimator = PyTorch(
        # Entry point
        entry_point='sagemaker_entry.py',
        source_dir='.',  # Current directory with all code
        
        # Model and environment
        framework_version='2.1.0',
        py_version='py310',
        
        # Hardware configuration - 4x A10G GPUs
        instance_type='ml.g5.12xlarge',  # 4x A10G GPUs
        instance_count=1,
        
        # SageMaker configuration
        role=role,
        job_name=job_name,
        
        # Hyperparameters
        hyperparameters={
            'max-train-samples': training_config['max_train_samples'],
            'max-eval-samples': training_config['max_eval_samples'],
            'epochs': training_config['epochs'],
            'lr': training_config['learning_rate']
        },
        
        # Environment variables
        environment={
            'CUDA_VISIBLE_DEVICES': '0,1,2,3',
            'TOKENIZERS_PARALLELISM': 'false',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
            'WANDB_PROJECT': 'maporl-medxpert-sagemaker'
        },
        
        # Resource configuration
        volume_size=100,  # GB
        max_run=3*60*60,  # 3 hours max
        
        # Output configuration
        output_path=f's3://{sagemaker_session.default_bucket()}/maporl-medxpert-outputs',
        code_location=f's3://{sagemaker_session.default_bucket()}/maporl-medxpert-code',
        
        # Dependencies
        requirements_file='requirements_sagemaker.txt',
        
        # Debugging and monitoring
        enable_sagemaker_metrics=True,
        metric_definitions=[
            {'Name': 'train:accuracy', 'Regex': 'Train Accuracy: ([0-9\\.]+)'},
            {'Name': 'eval:accuracy', 'Regex': 'Eval Accuracy: ([0-9\\.]+)'},
            {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
            {'Name': 'collaboration:score', 'Regex': 'Collaboration Score: ([0-9\\.]+)'},
            {'Name': 'medical:relevance', 'Regex': 'Medical Relevance: ([0-9\\.]+)'},
        ]
    )
    
    # Prepare training data inputs
    # Note: In practice, you would upload your MedXpert data to S3
    train_input = TrainingInput(
        s3_data=f's3://{sagemaker_session.default_bucket()}/medxpert-data/train/',
        content_type='application/json'
    )
    
    eval_input = TrainingInput(
        s3_data=f's3://{sagemaker_session.default_bucket()}/medxpert-data/eval/',
        content_type='application/json'
    )
    
    print("🚀 Launching SageMaker training job...")
    print(f"📊 Instance Type: ml.g5.12xlarge (4x A10G GPUs)")
    print(f"🤖 Model: Qwen3-0.6B (4 agents)")
    print(f"📈 Expected Training Time: 2-3 hours")
    
    # Launch training job
    estimator.fit({
        'train': train_input,
        'eval': eval_input
    })
    
    print("✅ Training job submitted successfully!")
    print(f"📊 Job Name: {job_name}")
    print(f"🔗 Console URL: https://console.aws.amazon.com/sagemaker/home#/jobs/{job_name}")
    
    return estimator, job_name

def create_sample_data_upload_script():
    """Create a script to upload sample MedXpert data to S3."""
    upload_script = '''
#!/bin/bash

# Upload sample MedXpert data to S3 for SageMaker training
# Run this script before launching the training job

BUCKET_NAME="your-sagemaker-bucket"  # Replace with your bucket
PREFIX="medxpert-data"

echo "📊 Uploading MedXpert sample data to S3..."

# Create sample training data
cat > medxpert_train.jsonl << 'EOF'
{"id": "medx_001", "question": "A 65-year-old patient presents with chest pain and shortness of breath. What is the most appropriate initial diagnostic test?", "context": "Patient has a history of hypertension and diabetes. Chest pain is substernal and radiates to the left arm. Vital signs show BP 160/90, HR 95, RR 22.", "answer": "ECG should be the initial diagnostic test to evaluate for acute coronary syndrome, followed by chest X-ray and cardiac enzymes."}
{"id": "medx_002", "question": "What are the first-line treatments for type 2 diabetes in adults?", "context": "A 55-year-old obese patient with newly diagnosed type 2 diabetes. HbA1c is 8.5%. No contraindications to standard medications.", "answer": "Metformin is the first-line treatment, combined with lifestyle modifications including diet and exercise. Target HbA1c should be individualized but generally <7%."}
{"id": "medx_003", "question": "How should acute bacterial pneumonia be managed in a healthy adult?", "context": "A 35-year-old previously healthy adult presents with fever, productive cough, and consolidation on chest X-ray. No recent antibiotic use.", "answer": "Empirical antibiotic therapy with amoxicillin or doxycycline for outpatient treatment. Hospitalization criteria include severe illness, comorbidities, or treatment failure."}
EOF

# Create sample eval data
cat > medxpert_eval.jsonl << 'EOF'
{"id": "medx_eval_001", "question": "What is the appropriate management for a patient with acute myocardial infarction?", "context": "A 60-year-old patient presents with severe chest pain, ST-elevation on ECG, and elevated troponins. Symptom onset was 2 hours ago.", "answer": "Immediate reperfusion therapy with primary PCI (preferred) or thrombolytic therapy if PCI unavailable. Dual antiplatelet therapy, anticoagulation, and supportive care."}
{"id": "medx_eval_002", "question": "How should diabetic ketoacidosis be treated?", "context": "A 25-year-old type 1 diabetic presents with blood glucose 450 mg/dL, ketones positive, pH 7.25, and dehydration.", "answer": "IV fluid resuscitation, insulin therapy, electrolyte monitoring and replacement (especially potassium), and treating precipitating factors. Monitor for complications."}
EOF

# Upload to S3
aws s3 cp medxpert_train.jsonl s3://$BUCKET_NAME/$PREFIX/train/
aws s3 cp medxpert_eval.jsonl s3://$BUCKET_NAME/$PREFIX/eval/

# Clean up local files
rm medxpert_train.jsonl medxpert_eval.jsonl

echo "✅ Data uploaded successfully!"
echo "📊 Train data: s3://$BUCKET_NAME/$PREFIX/train/medxpert_train.jsonl"
echo "📊 Eval data: s3://$BUCKET_NAME/$PREFIX/eval/medxpert_eval.jsonl"
    '''
    
    with open('scripts/upload_medxpert_data.sh', 'w') as f:
        f.write(upload_script)
    
    os.chmod('scripts/upload_medxpert_data.sh', 0o755)
    print("✅ Created data upload script: scripts/upload_medxpert_data.sh")

def monitor_training_job(job_name: str):
    """Monitor the training job and display logs."""
    sagemaker_session = sagemaker.Session()
    
    print(f"📊 Monitoring training job: {job_name}")
    print("🔍 Training logs:")
    
    # Get training job description
    client = boto3.client('sagemaker')
    response = client.describe_training_job(TrainingJobName=job_name)
    
    print(f"Status: {response['TrainingJobStatus']}")
    print(f"Instance Type: {response['ResourceConfig']['InstanceType']}")
    print(f"Instance Count: {response['ResourceConfig']['InstanceCount']}")
    
    # Monitor logs (simplified version)
    logs_client = boto3.client('logs')
    log_group = f"/aws/sagemaker/TrainingJobs/{job_name}"
    
    try:
        streams = logs_client.describe_log_streams(logGroupName=log_group)
        if streams['logStreams']:
            stream_name = streams['logStreams'][0]['logStreamName']
            events = logs_client.get_log_events(
                logGroupName=log_group,
                logStreamName=stream_name,
                limit=50
            )
            
            print("\n📋 Recent log entries:")
            for event in events['events'][-10:]:  # Last 10 entries
                print(f"  {event['message'].strip()}")
    except Exception as e:
        print(f"⚠️  Could not retrieve logs: {e}")

def main():
    """Main function to demonstrate SageMaker training setup."""
    print("🏥 Medical Multi-Agent MAPoRL Training on SageMaker")
    print("📊 Target: MedXpert Benchmark Improvement")
    print("🚀 Hardware: 4x A10G GPUs (ml.g5.12xlarge)")
    print()
    
    # Create sample data upload script
    create_sample_data_upload_script()
    
    print("📋 Setup Instructions:")
    print("1. Run 'scripts/upload_medxpert_data.sh' to upload sample data")
    print("2. Update the bucket name in the script")
    print("3. Ensure you have SageMaker execution role configured")
    print()
    
    # Option to launch training job
    response = input("🚀 Launch training job now? (y/n): ")
    
    if response.lower() == 'y':
        try:
            estimator, job_name = create_sagemaker_training_job()
            print(f"\n✅ Training job launched: {job_name}")
            
            # Option to monitor
            monitor_response = input("📊 Monitor training job? (y/n): ")
            if monitor_response.lower() == 'y':
                monitor_training_job(job_name)
                
        except Exception as e:
            print(f"❌ Error launching training job: {e}")
            print("💡 Make sure you have:")
            print("  - AWS credentials configured")
            print("  - SageMaker execution role")
            print("  - S3 bucket with training data")
    else:
        print("📋 Training job not launched. Use the setup instructions above.")
        print()
        print("🔗 Useful commands:")
        print("  - aws sagemaker list-training-jobs")
        print("  - aws logs describe-log-groups --log-group-name-prefix '/aws/sagemaker/TrainingJobs'")

if __name__ == "__main__":
    main() 