# AWS SageMaker Deployment Guide
## Medical Multi-Agent System with Qwen3-0.6B

Complete step-by-step guide to deploy and run the medical multi-agent system on AWS SageMaker.

---

## 🚀 Quick Start Sequence

### Step 1: AWS Prerequisites
```bash
# 1. Install AWS CLI and configure credentials
pip install awscli boto3
aws configure

# 2. Set up required environment variables
export AWS_REGION="us-west-2"  # or your preferred region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export S3_BUCKET="medical-multiagent-$(date +%s)"  # Unique bucket name
export SAGEMAKER_ROLE="arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerExecutionRole"
```

### Step 2: Create S3 Bucket and Upload Data
```bash
# Create S3 bucket for data and models
aws s3 mb s3://${S3_BUCKET} --region ${AWS_REGION}

# Create local data directory and prepare MedXpert data
mkdir -p data/medxpert
python3 << 'EOF'
import json
import os

# Sample MedXpert data (replace with your actual dataset)
medxpert_data = [
    {
        "question": "A 45-year-old patient presents with sudden onset severe headache, neck stiffness, and photophobia. What is the most likely diagnosis and immediate management?",
        "answer": "Most likely diagnosis is subarachnoid hemorrhage. Immediate management includes: 1) Stabilize airway, breathing, circulation 2) CT head without contrast 3) If CT negative, lumbar puncture 4) Neurosurgical consultation 5) Blood pressure control 6) Nimodipine for vasospasm prevention",
        "category": "neurology",
        "difficulty": "high"
    },
    {
        "question": "What are the contraindications for thrombolytic therapy in acute stroke?",
        "answer": "Contraindications include: 1) Symptoms >4.5 hours 2) Recent surgery/trauma 3) Active bleeding 4) Severe hypertension >185/110 5) Anticoagulation with elevated INR 6) Platelet count <100k 7) Previous ICH 8) Large stroke on imaging",
        "category": "neurology",
        "difficulty": "medium"
    },
    {
        "question": "A 30-year-old pregnant woman at 32 weeks gestation presents with severe epigastric pain, elevated blood pressure, and proteinuria. Management approach?",
        "answer": "Diagnosis: Preeclampsia with severe features. Management: 1) Antihypertensive therapy (labetalol/hydralazine) 2) Magnesium sulfate for seizure prophylaxis 3) Corticosteroids for fetal lung maturity 4) Continuous fetal monitoring 5) Delivery planning - may need immediate delivery if severe",
        "category": "obstetrics",
        "difficulty": "high"
    }
]

os.makedirs('data/medxpert', exist_ok=True)
with open('data/medxpert/medxpert_train.json', 'w') as f:
    json.dump(medxpert_data, f, indent=2)

print(f"✅ Created MedXpert training data with {len(medxpert_data)} samples")
EOF

# Upload data to S3
aws s3 cp data/medxpert/ s3://${S3_BUCKET}/medxpert-data/ --recursive
echo "✅ Data uploaded to S3"
```

### Step 3: Create SageMaker Execution Role
```bash
# Create trust policy for SageMaker
cat > sagemaker-trust-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create IAM role
aws iam create-role \
    --role-name SageMakerMedicalExecutionRole \
    --assume-role-policy-document file://sagemaker-trust-policy.json

# Attach required policies
aws iam attach-role-policy \
    --role-name SageMakerMedicalExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name SageMakerMedicalExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Update role ARN
export SAGEMAKER_ROLE="arn:aws:iam::${AWS_ACCOUNT_ID}:role/SageMakerMedicalExecutionRole"
echo "✅ SageMaker role created: ${SAGEMAKER_ROLE}"
```

### Step 4: Prepare Training Code
```bash
# Create code archive for SageMaker
mkdir -p sagemaker-code
cp sagemaker_entry.py sagemaker-code/
cp -r config/ sagemaker-code/
cp -r src/ sagemaker-code/ 2>/dev/null || echo "No src directory found"
cp requirements_sagemaker.txt sagemaker-code/requirements.txt

# Create training script wrapper
cat > sagemaker-code/train << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting SageMaker Medical Multi-Agent Training"
echo "🤖 Model: Qwen3-0.6B (4 agents)"
echo "🖥️ Hardware: 4x A10G GPUs"

# Install dependencies
pip install -r requirements.txt

# Run training
python sagemaker_entry.py \
    --model-dir ${SM_MODEL_DIR} \
    --data-dir ${SM_CHANNEL_TRAINING} \
    --output-dir ${SM_OUTPUT_DIR}
EOF

chmod +x sagemaker-code/train

# Upload code to S3
tar -czf medical-multiagent-code.tar.gz -C sagemaker-code .
aws s3 cp medical-multiagent-code.tar.gz s3://${S3_BUCKET}/code/
echo "✅ Training code uploaded to S3"
```

### Step 5: Launch SageMaker Training Job
```bash
# Create training job configuration
python3 << EOF
import boto3
import json
from datetime import datetime

# SageMaker client
sagemaker = boto3.client('sagemaker', region_name='${AWS_REGION}')

# Job configuration
job_name = f"medical-multiagent-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

training_job_config = {
    'TrainingJobName': job_name,
    'RoleArn': '${SAGEMAKER_ROLE}',
    
    # Algorithm specification
    'AlgorithmSpecification': {
        'TrainingInputMode': 'File',
        'TrainingImage': '763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker',
    },
    
    # Input data configuration
    'InputDataConfig': [
        {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://${S3_BUCKET}/medxpert-data/',
                    'S3DataDistributionType': 'FullyReplicated',
                }
            },
            'ContentType': 'application/json',
            'CompressionType': 'None',
        },
        {
            'ChannelName': 'code',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://${S3_BUCKET}/code/',
                    'S3DataDistributionType': 'FullyReplicated',
                }
            },
            'ContentType': 'application/x-tar',
            'CompressionType': 'Gzip',
        }
    ],
    
    # Output configuration
    'OutputDataConfig': {
        'S3OutputPath': 's3://${S3_BUCKET}/models/',
    },
    
    # Resource configuration - 4x A10G GPUs
    'ResourceConfig': {
        'InstanceType': 'ml.g5.12xlarge',  # 4x A10G GPUs
        'InstanceCount': 1,
        'VolumeSizeInGB': 100,
    },
    
    # Hyperparameters
    'HyperParameters': {
        'model-name': 'Qwen/Qwen3-0.6B',
        'num-agents': '4',
        'target-benchmark': 'MedXpert',
        'epochs': '2',
        'batch-size': '1',
        'learning-rate': '2e-5',
    },
    
    # Stopping condition
    'StoppingCondition': {
        'MaxRuntimeInSeconds': 7200,  # 2 hours
    },
    
    # Environment variables
    'Environment': {
        'TRANSFORMERS_CACHE': '/tmp/transformers',
        'HF_HOME': '/tmp/huggingface',
        'CUDA_VISIBLE_DEVICES': '0,1,2,3',
        'PYTHONUNBUFFERED': '1',
    }
}

try:
    print("🚀 Launching SageMaker Training Job")
    print(f"📋 Job Name: {job_name}")
    print(f"🤖 Model: Qwen3-0.6B (4 agents)")
    print(f"🖥️ Instance: ml.g5.12xlarge (4x A10G)")
    
    # Create training job
    response = sagemaker.create_training_job(**training_job_config)
    print(f"✅ Training job created: {response['TrainingJobArn']}")
    
    # Save job name for monitoring
    with open('current_training_job.txt', 'w') as f:
        f.write(job_name)
        
    print(f"📝 Job name saved to: current_training_job.txt")
    print(f"🔍 Monitor with: aws sagemaker describe-training-job --training-job-name {job_name}")
    
except Exception as e:
    print(f"❌ Error launching training job: {e}")
    raise
EOF
```

### Step 6: Monitor Training Job
```bash
# Get current job name
JOB_NAME=$(cat current_training_job.txt)

# Monitor training progress
echo "📊 Monitoring training job: ${JOB_NAME}"
while true; do
    STATUS=$(aws sagemaker describe-training-job --training-job-name ${JOB_NAME} \
        --query 'TrainingJobStatus' --output text)
    
    echo "$(date): Status = ${STATUS}"
    
    if [[ "${STATUS}" == "Completed" || "${STATUS}" == "Failed" || "${STATUS}" == "Stopped" ]]; then
        break
    fi
    
    sleep 30  # Check every 30 seconds
done

# Get final status and results
aws sagemaker describe-training-job --training-job-name ${JOB_NAME} > training_results.json
echo "📋 Final training results saved to: training_results.json"

# Check if successful
if [[ "${STATUS}" == "Completed" ]]; then
    MODEL_ARTIFACTS=$(aws sagemaker describe-training-job --training-job-name ${JOB_NAME} \
        --query 'ModelArtifacts.S3ModelArtifacts' --output text)
    echo "✅ Training completed successfully!"
    echo "📁 Model artifacts: ${MODEL_ARTIFACTS}"
else
    echo "❌ Training failed or was stopped"
    aws sagemaker describe-training-job --training-job-name ${JOB_NAME} \
        --query 'FailureReason' --output text
fi
```

### Step 7: Test Trained Model
```bash
# Download and test the trained model
if [[ "${STATUS}" == "Completed" ]]; then
    # Download model artifacts
    aws s3 cp ${MODEL_ARTIFACTS} model.tar.gz
    mkdir -p trained_model
    tar -xzf model.tar.gz -C trained_model/
    
    # Run test script
    bash scripts/sagemaker_test_medxpert.sh
    
    echo "✅ Model testing completed"
else
    echo "⚠️ Skipping testing - training was not successful"
fi
```

---

## 📊 Expected Results

### Training Metrics
- **Duration**: 1-2 hours on ml.g5.12xlarge
- **Memory Usage**: ~20GB GPU memory (across 4 A10G GPUs)
- **Cost**: ~$15-30 per training run
- **Accuracy**: Expected 75-80% on MedXpert benchmark

### Output Files
- `trained_model/`: Trained Qwen3-0.6B models for all 4 agents
- `training_results.json`: Detailed training metrics
- `model_config.json`: Model configuration and metadata

---

## 🛠️ Troubleshooting

### Common Issues

1. **Insufficient Permissions**
   ```bash
   # Add additional IAM policies if needed
   aws iam attach-role-policy \
       --role-name SageMakerMedicalExecutionRole \
       --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
   ```

2. **GPU Memory Issues**
   ```bash
   # Reduce batch size in hyperparameters
   'batch-size': '1',  # Instead of 2
   'gradient-accumulation-steps': '16'  # Increase to maintain effective batch size
   ```

3. **Training Timeout**
   ```bash
   # Increase max runtime in training job config
   'MaxRuntimeInSeconds': 14400,  # 4 hours instead of 2
   ```

### Monitoring Commands
```bash
# Check training job logs
aws logs describe-log-streams \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name-prefix ${JOB_NAME}

# Monitor GPU utilization (if accessible)
aws sagemaker describe-training-job \
    --training-job-name ${JOB_NAME} \
    --query 'ResourceConfig'
```

---

## 🔧 Customization Options

### Modify Training Parameters
Edit the hyperparameters in Step 5:
```python
'HyperParameters': {
    'model-name': 'Qwen/Qwen3-0.6B',
    'num-agents': '4',
    'epochs': '3',                    # Increase for longer training
    'batch-size': '2',               # Adjust based on memory
    'learning-rate': '1e-5',         # Lower for more stable training
    'warmup-steps': '100',           # Add warmup
}
```

### Use Different Instance Types
```python
# For larger models or more memory
'InstanceType': 'ml.g5.24xlarge',  # 4x A10G with more CPU/RAM

# For cost optimization (single GPU)
'InstanceType': 'ml.g5.xlarge',    # 1x A10G
'HyperParameters': {'num-agents': '1'}  # Reduce to single agent
```

### Add Custom Data
Replace the sample data in Step 2 with your own MedXpert dataset:
```bash
# Upload your own dataset
aws s3 cp your_medxpert_dataset.json s3://${S3_BUCKET}/medxpert-data/
```

---

## 📝 Next Steps

After successful training:
1. **Model Evaluation**: Use the test script to evaluate on MedXpert benchmark
2. **Model Deployment**: Deploy to SageMaker endpoint for inference
3. **Integration**: Integrate with your medical applications
4. **Monitoring**: Set up CloudWatch monitoring for production use

## 🔗 Additional Resources

- [SageMaker PyTorch Documentation](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/index.html)
- [Qwen3 Model Documentation](https://huggingface.co/Qwen/Qwen3-0.6B)
- [MedXpert Benchmark](https://github.com/openmedlab/MedXpert)
- [AWS SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/) 