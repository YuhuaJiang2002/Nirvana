#!/bin/bash

# Two-stage MRI reconstruction training script for Nirvana model
# This script runs both stages of training sequentially

set -e  # Exit on any error
# cd /cpfs04/shared/mabasic/jiangyuhua/Nirvana/specialized_ability/MRI_image_reconstruction/train
# Configuration variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Model paths
CONFIG_PATH="$SCRIPT_DIR/../model/nirvana_1_3B.json"
BACKBONE_PATH="$SCRIPT_DIR/../model/model.safetensors"
# BACKBONE_PATH="/cpfs04/shared/mabasic/jiangyuhua/Nirvana-vision/hf-95500/model.safetensors"

# Data paths
DATA_DIR_1="$SCRIPT_DIR/../data/MRI/sft"
DATA_DIR_2="$SCRIPT_DIR/../data/MRI/brain_multicoil_train_batch_1"

# DATA_DIR_1="/cpfs04/shared/mabasic/jiangyuhua/data/MRI/sft"
# DATA_DIR_2="/cpfs04/shared/mabasic/jiangyuhua/data/MRI/brain_multicoil_train_batch_1"

# Training parameters
NUM_COILS=16
IMG_SIZE=320
VIT_EMBED_DIM=768
VIT_NUM_LAYERS=2
VIT_NUM_HEADS=12

# Stage 1 parameters (k-space encoder)
STAGE1_EPOCHS=100
STAGE1_BATCH_SIZE=4
STAGE1_LEARNING_RATE=3e-4
STAGE1_SAVE_DIR="./stage1_checkpoints"

# Stage 2 parameters (image decoder)
STAGE2_EPOCHS=100
STAGE2_BATCH_SIZE=4
STAGE2_LEARNING_RATE=1e-4
STAGE2_SAVE_DIR="./stage2_checkpoints"
DECODER_TYPE="full"  # full: 160M parameters, lightweight: 80M parameters
DECODER_HIDDEN_DIM=512

# Common parameters
NUM_WORKERS=8
SAVE_INTERVAL=10
LOG_INTERVAL=100
WEIGHT_DECAY=1e-4
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.1

# Logging
LOG_FILE="two_stage_training.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required files exist
check_requirements() {
    print_info "Checking requirements..."
    
    # Check if configuration file exists
    if [[ ! -f "$CONFIG_PATH" ]]; then
        print_error "Configuration file not found: $CONFIG_PATH"
        exit 1
    fi
    
    # Check if backbone weights exist
    if [[ ! -f "$BACKBONE_PATH" ]]; then
        print_error "Backbone weights not found: $BACKBONE_PATH"
        exit 1
    fi
    
    # Check if data directory exists
    if [[ ! -d "$DATA_DIR_1" ]]; then
        print_error "Data directory not found: $DATA_DIR_1"
        exit 1
    fi

    if [[ ! -d "$DATA_DIR_2" ]]; then
        print_error "Data directory not found: $DATA_DIR_2"
        exit 1
    fi
    
    print_success "All requirements satisfied"
}

# Function to create directories
create_directories() {
    print_info "Creating output directories..."
    
    mkdir -p "$STAGE1_SAVE_DIR"
    mkdir -p "$STAGE2_SAVE_DIR"
    mkdir -p "./logs"
    
    print_success "Directories created"
}

# Function to run Stage 1 training

cd ..

run_stage1() {
    print_info "Starting Stage 1: K-space encoder training..."
    print_info "Training k-space encoder with cross-entropy loss on MRI analysis tokens"
    
    cd "$(dirname "$SCRIPT_DIR")"  # change to MRI_image_reconstruction directory
    
    CUDA_VISIBLE_DEVICES=0 python -m train.stage1_kspace_encoder \
        --config_path "$CONFIG_PATH" \
        --backbone_path "$BACKBONE_PATH" \
        --data_dir "$DATA_DIR_1" \
        --num_coils "$NUM_COILS" \
        --img_size "$IMG_SIZE" \
        --vit_embed_dim "$VIT_EMBED_DIM" \
        --vit_num_layers "$VIT_NUM_LAYERS" \
        --vit_num_heads "$VIT_NUM_HEADS" \
        --epochs "$STAGE1_EPOCHS" \
        --batch_size "$STAGE1_BATCH_SIZE" \
        --learning_rate "$STAGE1_LEARNING_RATE" \
        --weight_decay "$WEIGHT_DECAY" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --warmup_ratio "$WARMUP_RATIO" \
        --save_dir "$STAGE1_SAVE_DIR" \
        --save_interval "$SAVE_INTERVAL" \
        --log_interval "$LOG_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --label_smoothing 0.1 \
        --l2_reg 0.0 \
        --log_gradients \
        2>&1 | tee "./logs/stage1_${TIMESTAMP}.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_success "Stage 1 training completed successfully!"
    else
        print_error "Stage 1 training failed!"
        exit 1
    fi
}

# Function to run Stage 2 training
run_stage2() {
    print_info "Starting Stage 2: Image decoder training..."
    print_info "Training image decoder with SSIM loss on reconstructed images"
    
    # Find the best Stage 1 checkpoint
    STAGE1_CHECKPOINT=$(find "$STAGE1_SAVE_DIR" -name "stage1_best_model.pt" -type f | head -1)
    
    if [[ -z "$STAGE1_CHECKPOINT" ]]; then
        print_error "Stage 1 best model checkpoint not found!"
        exit 1
    fi
    
    print_info "Using Stage 1 checkpoint: $STAGE1_CHECKPOINT"
    
    cd "$(dirname "$SCRIPT_DIR")"  # 切换到MRI_image_reconstruction目录
    
    CUDA_VISIBLE_DEVICES=0 python -m train.stage2_image_decoder \
        --config_path "$CONFIG_PATH" \
        --backbone_path "$BACKBONE_PATH" \
        --stage1_checkpoint "$STAGE1_CHECKPOINT" \
        --data_dir "$DATA_DIR_2" \
        --num_coils "$NUM_COILS" \
        --img_size "$IMG_SIZE" \
        --vit_embed_dim "$VIT_EMBED_DIM" \
        --vit_num_layers "$VIT_NUM_LAYERS" \
        --vit_num_heads "$VIT_NUM_HEADS" \
        --decoder_type "$DECODER_TYPE" \
        --decoder_hidden_dim "$DECODER_HIDDEN_DIM" \
        --use_bilinear_upsample \
        --epochs "$STAGE2_EPOCHS" \
        --batch_size "$STAGE2_BATCH_SIZE" \
        --learning_rate "$STAGE2_LEARNING_RATE" \
        --weight_decay "$WEIGHT_DECAY" \
        --max_grad_norm "$MAX_GRAD_NORM" \
        --warmup_ratio "$WARMUP_RATIO" \
        --save_dir "$STAGE2_SAVE_DIR" \
        --save_interval "$SAVE_INTERVAL" \
        --log_interval "$LOG_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --l2_reg 0.0 \
        --log_gradients \
        --log_image_stats \
        2>&1 | tee "./logs/stage2_${TIMESTAMP}.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        print_success "Stage 2 training completed successfully!"
    else
        print_error "Stage 2 training failed!"
        exit 1
    fi
}

# Function to display training summary
show_summary() {
    print_info "Training Summary:"
    echo "=================="
    echo "Stage 1 (K-space encoder):"
    echo "  - Epochs: $STAGE1_EPOCHS"
    echo "  - Batch size: $STAGE1_BATCH_SIZE"
    echo "  - Learning rate: $STAGE1_LEARNING_RATE"
    echo "  - Save directory: $STAGE1_SAVE_DIR"
    echo ""
    echo "Stage 2 (Image decoder):"
    echo "  - Epochs: $STAGE2_EPOCHS"
    echo "  - Batch size: $STAGE2_BATCH_SIZE"
    echo "  - Learning rate: $STAGE2_LEARNING_RATE"
    echo "  - Decoder type: $DECODER_TYPE"
    echo "  - Save directory: $STAGE2_SAVE_DIR"
    echo ""
    echo "Model configuration:"
    echo "  - Number of coils: $NUM_COILS"
    echo "  - Image size: $IMG_SIZE"
    echo "  - ViT embedding dim: $VIT_EMBED_DIM"
    echo "  - ViT layers: $VIT_NUM_LAYERS"
    echo "  - ViT heads: $VIT_NUM_HEADS"
    echo ""
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
        echo ""
    else
        print_warning "nvidia-smi not found. GPU training may not be available."
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -s, --stage1-only       Run only Stage 1 training"
    echo "  -d, --stage2-only       Run only Stage 2 training (requires Stage 1 checkpoint)"
    echo "  -c, --checkpoint PATH   Path to Stage 1 checkpoint for Stage 2 only"
    echo "  -v, --verbose           Enable verbose logging"
    echo ""
    echo "Examples:"
    echo "  $0                      # Run complete two-stage training"
    echo "  $0 --stage1-only        # Run only Stage 1"
    echo "  $0 --stage2-only -c ./stage1_checkpoints/stage1_best_model.pt"
    echo ""
}

# Parse command line arguments
STAGE1_ONLY=false
STAGE2_ONLY=true
STAGE1_CHECKPOINT="./stage1_checkpoints/stage1_best_model.pt"
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--stage1-only)
            STAGE1_ONLY=true
            shift
            ;;
        -d|--stage2-only)
            STAGE2_ONLY=true
            shift
            ;;
        -c|--checkpoint)
            STAGE1_CHECKPOINT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Starting two-stage MRI reconstruction training with Nirvana model"
    print_info "Timestamp: $TIMESTAMP"
    echo ""
    
    # Check requirements
    check_requirements
    
    # Check GPU
    check_gpu
    
    # Create directories
    create_directories
    
    # Show training summary
    show_summary
    
    # Run training based on options
    if [[ "$STAGE1_ONLY" == true ]]; then
        print_info "Running Stage 1 only..."
        run_stage1
    elif [[ "$STAGE2_ONLY" == true ]]; then
        if [[ -z "$STAGE1_CHECKPOINT" ]]; then
            print_error "Stage 1 checkpoint path is required for Stage 2 only training!"
            exit 1
        fi
        print_info "Running Stage 2 only with checkpoint: $STAGE1_CHECKPOINT"
        # Modify the checkpoint path for Stage 2
        export STAGE1_CHECKPOINT_PATH="$STAGE1_CHECKPOINT"
        run_stage2
    else
        print_info "Running complete two-stage training..."
        run_stage1
        run_stage2
    fi
    
    print_success "All training completed successfully!"
    print_info "Checkpoints saved to:"
    echo "  Stage 1: $STAGE1_SAVE_DIR"
    echo "  Stage 2: $STAGE2_SAVE_DIR"
    print_info "Logs saved to: ./logs/"
}

# Run main function
main "$@" 