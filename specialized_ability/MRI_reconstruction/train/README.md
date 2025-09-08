# Two-Stage MRI Reconstruction Training with Nirvana

This directory contains the implementation of a two-stage training approach for MRI reconstruction using the Nirvana model. The training process is designed to leverage the pre-trained Nirvana backbone while training specialized components for MRI reconstruction.

## Overview

The two-stage training approach consists of:

1. **Stage 1**: Train the k-space encoder (VarNet + ViT) with cross-entropy loss on MRI analysis tokens
2. **Stage 2**: Train the image decoder with SSIM loss on reconstructed images

This approach ensures that:
- The Nirvana backbone remains frozen, preserving its language understanding capabilities
- The k-space encoder learns to extract meaningful features from k-space signals
- The image decoder learns to reconstruct high-quality MRI images
- Training stability is maintained through proper parameter freezing

## Architecture

### Model Components

- **Nirvana Backbone**: 1.3B parameter frozen Nirvana model
- **K-space Encoder**: Multi-coil variational network (VarNet) + lightweight ViT
- **Image Decoder**: U-Net architecture
- **Layer Normalization**: Applied before k-space encoder and image decoder for training stability

### Training Stages

#### Stage 1: K-space Encoder Training
- **Objective**: Train k-space encoder to generate meaningful features from k-space signals
- **Loss**: Cross-entropy loss on MRI analysis tokens
- **Trainable Parameters**: Only k-space encoder (VarNet + ViT)
- **Frozen Parameters**: Nirvana backbone, image decoder

#### Stage 2: Image Decoder Training
- **Objective**: Train image decoder to reconstruct high-quality MRI images
- **Loss**: SSIM loss between reconstructed and ground truth images
- **Trainable Parameters**: Only image decoder
- **Frozen Parameters**: Nirvana backbone, k-space encoder

## Files Structure

```
train/
├── two_stage_training.py          # Complete two-stage training script
├── stage1_kspace_encoder.py       # Stage 1 training script
├── stage2_image_decoder.py        # Stage 2 training script
├── run_two_stage_training.sh      # Shell script for easy execution
├── config.yaml                    # Configuration file
└── README.md                      # This file
```

## Requirements

### Dependencies
- PyTorch >= 1.12
- Transformers >= 4.20
- Accelerate
- fastMRI
- safetensors
- tqdm
- numpy

### Hardware Requirements
- GPU with sufficient VRAM (recommended: 24GB+)
- CUDA support
- Sufficient disk space for checkpoints and logs

## Quick Start

### 1. Prepare Data
Ensure your MRI data is organized in the expected format:
```
brain_multicoil_train/
├── file1.h5
├── file2.h5
└── ...
```

### 2. Prepare Model Files
Place the required model files in the project root:
- `nirvana_1_3B.json`: Nirvana model configuration
- `model.safetensors`: Pre-trained backbone weights

### 3. Run Training

#### Option A: Complete Two-Stage Training
```bash
# Make script executable
chmod +x run_two_stage_training.sh

# Run complete training
./run_two_stage_training.sh
```

#### Option B: Individual Stage Training
```bash
# Stage 1 only
./run_two_stage_training.sh --stage1-only

# Stage 2 only (requires Stage 1 checkpoint)
./run_two_stage_training.sh --stage2-only -c ./stage1_checkpoints/stage1_best_model.pt
```

#### Option C: Python Scripts Directly
```bash
# Stage 1
python stage1_kspace_encoder.py \
    --config_path ./nirvana_1_3B.json \
    --backbone_path ./model.safetensors \
    --data_dir ./brain_multicoil_train \
    --epochs 100 \
    --batch_size 4

# Stage 2
python stage2_image_decoder.py \
    --config_path ./nirvana_1_3B.json \
    --backbone_path ./model.safetensors \
    --stage1_checkpoint ./stage1_checkpoints/stage1_best_model.pt \
    --data_dir ./brain_multicoil_train \
    --epochs 100 \
    --batch_size 4
```

## Configuration

### Key Parameters

#### Model Configuration
- `num_coils`: Number of MRI coils (default: 16)
- `img_size`: Input image size (default: 320)
- `vit_embed_dim`: ViT embedding dimension (default: 768)
- `vit_num_layers`: Number of ViT layers (default: 6)
- `vit_num_heads`: Number of ViT attention heads (default: 12)

#### Training Configuration
- `epochs`: Number of training epochs per stage (default: 100)
- `batch_size`: Training batch size (default: 4)
- `learning_rate`: Learning rate (Stage 1: 3e-4, Stage 2: 1e-4)
- `weight_decay`: Weight decay (default: 1e-4)
- `max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)

#### Image Decoder Configuration
- `decoder_type`: "full" (160M params) or "lightweight" (80M params)
- `decoder_hidden_dim`: Hidden dimension for decoder (default: 512)
- `use_bilinear_upsample`: Use bilinear upsampling in U-Net

### Configuration File
Modify `config.yaml` to customize training parameters without changing code.

## Training Process

### Stage 1: K-space Encoder Training
1. Load pre-trained Nirvana backbone
2. Freeze backbone parameters
3. Initialize k-space encoder (VarNet + ViT)
4. Train with cross-entropy loss on MRI analysis tokens
5. Save best model checkpoint

### Stage 2: Image Decoder Training
1. Load trained k-space encoder from Stage 1
2. Freeze k-space encoder and backbone
3. Initialize image decoder (U-Net)
4. Train with SSIM loss on reconstructed images
5. Save best model checkpoint

## Output and Checkpoints

### Checkpoints
- **Stage 1**: `./stage1_checkpoints/`
  - `stage1_best_model.pt`: Best model based on validation loss
  - `stage1_final_model.pt`: Final model after all epochs
  - `stage1_epoch_N.pt`: Regular checkpoints every N epochs

- **Stage 2**: `./stage2_checkpoints/`
  - `stage2_best_model.pt`: Best model based on SSIM loss
  - `stage2_final_model.pt`: Final model after all epochs
  - `stage2_epoch_N.pt`: Regular checkpoints every N epochs

### Logs
- Training logs are saved to `./logs/`
- Console output is also displayed in real-time
- Log files include timestamps for easy tracking

## Monitoring and Debugging

### Training Progress
- Real-time progress bars with loss and learning rate
- Detailed logging every N batches
- Gradient norm monitoring
- Image statistics logging (Stage 2)

### Common Issues and Solutions

#### Out of Memory (OOM)
- Reduce batch size
- Reduce image size
- Use gradient accumulation
- Enable mixed precision training

#### Training Instability
- Reduce learning rate
- Increase weight decay
- Adjust gradient clipping
- Check data normalization

#### Poor Convergence
- Verify data quality and preprocessing
- Check loss function implementation
- Monitor gradient flow
- Adjust learning rate schedule

## Advanced Usage

### Custom Loss Functions
Modify the loss functions in the training scripts:
- Stage 1: Custom cross-entropy variants
- Stage 2: Custom image quality metrics

### Model Modifications
- Custom k-space encoder architectures
- Alternative image decoder designs
- Additional loss terms

## Performance Optimization

### Memory Optimization
- Gradient checkpointing
- Mixed precision training
- Dynamic batching
- Memory-efficient data loading

### Speed Optimization
- Data prefetching
- Optimized data transforms
- Efficient model architectures
- Hardware acceleration

## Evaluation and Testing

### Validation
- Implement validation loops
- Compute multiple metrics
- Save validation results
- Early stopping based on validation

### Testing
- Load trained models
- Run inference on test data
- Generate reconstruction results
- Compute final metrics
