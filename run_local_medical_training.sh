#!/bin/bash
set -e

# Local Medical Multi-Agent Training Script
# Automates the complete pipeline for training Qwen3-0.6B medical agents

echo "🏥 Medical Multi-Agent Local Training Pipeline"
echo "=============================================="
echo "🤖 Model: Qwen3-0.6B (4 specialized agents)"
echo "📊 Dataset: MedXpert + MedQuAD + Synthetic"
echo "💻 Mode: Local training (no SageMaker/S3)"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "config/model_config_qwen.py" ]]; then
    log_error "Please run this script from the MAPoRL project root directory"
    exit 1
fi

# Command line arguments
SKIP_INSTALL=false
SKIP_DOWNLOAD=false
SKIP_TRAINING=false
TEST_ONLY=false
DATA_DIR="data"
OUTPUT_DIR="outputs"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-install)
            SKIP_INSTALL=true
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-install     Skip dependency installation"
            echo "  --skip-download    Skip data download"
            echo "  --skip-training    Skip training (only test)"
            echo "  --test-only        Only run testing on existing models"
            echo "  --data-dir DIR     Data directory (default: data)"
            echo "  --output-dir DIR   Output directory (default: outputs)"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: Check Prerequisites
log_info "Checking system prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 is required but not installed"
    exit 1
fi

# Check GPU availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB)')
else:
    print('Running on CPU (training will be slow)')
" 2>/dev/null || {
    log_warning "PyTorch not installed or not working properly"
    if [ "$SKIP_INSTALL" = true ]; then
        log_error "PyTorch required but --skip-install specified"
        exit 1
    fi
}

log_success "System check completed"

# Step 2: Install Dependencies
if [ "$SKIP_INSTALL" = false ]; then
    log_info "Installing dependencies..."
    
    # Check if we're in a virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_warning "Not in a virtual environment. Consider using 'python -m venv venv && source venv/bin/activate'"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Install core dependencies
    log_info "Installing PyTorch and transformers..."
    python3 -m pip install --upgrade pip
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    python3 -m pip install transformers==4.51.0 datasets accelerate bitsandbytes
    
    # Install additional dependencies
    log_info "Installing additional packages..."
    python3 -m pip install requests pandas numpy scikit-learn
    python3 -m pip install langgraph langchain langsmith
    python3 -m pip install biomcp requests-cache
    
    log_success "Dependencies installed"
else
    log_info "Skipping dependency installation"
fi

# Step 3: Download Data  
if [ "$SKIP_DOWNLOAD" = false ] && [ "$TEST_ONLY" = false ]; then
    log_info "Downloading medical datasets..."
    
    # Create data directory
    mkdir -p "$DATA_DIR"
    
    # Download data
    if python3 download_medxpert_data.py; then
        log_success "Data download completed"
        
        # Verify data files
        if [[ -f "$DATA_DIR/medxpert_train.jsonl" ]]; then
            TRAIN_COUNT=$(wc -l < "$DATA_DIR/medxpert_train.jsonl")
            log_info "Training samples: $TRAIN_COUNT"
        fi
        
        if [[ -f "$DATA_DIR/medxpert_validation.jsonl" ]]; then
            VAL_COUNT=$(wc -l < "$DATA_DIR/medxpert_validation.jsonl")
            log_info "Validation samples: $VAL_COUNT"
        fi
        
    else
        log_error "Data download failed"
        exit 1
    fi
else
    log_info "Skipping data download"
fi

# Step 4: Training
if [ "$SKIP_TRAINING" = false ] && [ "$TEST_ONLY" = false ]; then
    log_info "Starting medical multi-agent training..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check if training data exists
    if [[ ! -f "$DATA_DIR/medxpert_train.jsonl" ]]; then
        log_error "Training data not found at $DATA_DIR/medxpert_train.jsonl"
        log_info "Run: python3 download_medxpert_data.py"
        exit 1
    fi
    
    # Start training
    log_info "Training 4 medical agents (this may take 1-4 hours)..."
    
    # Create a simple progress monitor
    (
        sleep 30
        while true; do
            if [[ -d "$OUTPUT_DIR/models" ]]; then
                COMPLETED=$(find "$OUTPUT_DIR/models" -name "training_stats.json" | wc -l)
                log_info "Training progress: $COMPLETED/4 agents completed"
            fi
            sleep 60
        done
    ) &
    MONITOR_PID=$!
    
    # Run training
    if python3 local_train_medxpert.py --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"; then
        kill $MONITOR_PID 2>/dev/null || true
        log_success "Training completed successfully"
        
        # Show training summary
        if [[ -f "$OUTPUT_DIR/training_results.json" ]]; then
            log_info "Training Summary:"
            python3 -c "
import json
with open('$OUTPUT_DIR/training_results.json', 'r') as f:
    results = json.load(f)
print(f'  Total agents trained: {results[\"total_agents\"]}')
print(f'  Training date: {results[\"training_date\"]}')
print(f'  Dataset samples: {results[\"dataset_info\"][\"train_samples\"]}')
"
        fi
        
    else
        kill $MONITOR_PID 2>/dev/null || true
        log_error "Training failed"
        exit 1
    fi
else
    log_info "Skipping training"
fi

# Step 5: Testing
log_info "Testing trained agents..."

if python3 local_train_medxpert.py --test-only --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"; then
    log_success "Testing completed"
    
    # Show test results summary
    if [[ -f "$OUTPUT_DIR/test_results.json" ]]; then
        log_info "Test Results Summary:"
        python3 -c "
import json
with open('$OUTPUT_DIR/test_results.json', 'r') as f:
    results = json.load(f)
print(f'  Agents tested: {len(results)}')
for agent, responses in results.items():
    print(f'  {agent}: {len(responses)} test questions')
"
    fi
else
    log_warning "Testing failed or no trained models found"
fi

# Final Summary
echo ""
echo "🎉 Medical Multi-Agent Training Pipeline Complete!"
echo "================================================="
log_info "Output files:"
echo "  📁 Models: $OUTPUT_DIR/models/"
echo "  📊 Training results: $OUTPUT_DIR/training_results.json"
echo "  🧪 Test results: $OUTPUT_DIR/test_results.json"
echo "  📈 Logs: $OUTPUT_DIR/logs/"

echo ""
log_info "Next steps:"
echo "  1. Review results: cat $OUTPUT_DIR/training_results.json"
echo "  2. Test custom questions: python3 local_train_medxpert.py --test-only"
echo "  3. Deploy for inference: Create API endpoints"
echo "  4. Fine-tune further: Add more medical data"

echo ""
log_success "Pipeline completed successfully! 🎯" 