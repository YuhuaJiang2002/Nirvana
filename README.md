# Nirvana: A Specialized Genearlist Model With Task-Aware Memory Mechanism

Nirvana is a Specialized Genearlist Model with specialized memory mechanism, linear time complexity, and test-time task information extraction. 

## 🏗️ Project Architecture

### Core Components

```
Nirvana/
├── nirvana_backbone/          # Core model architecture and training
│   ├── train/                 # Training scripts and configurations
│   ├── eval/                  # Evaluation scripts and benchmarks
│   ├── modeling_transformer_rnn.py      # Nirvana model
│   ├── nirvana_1_3B.json                # Nirvana 1.3B configuration
│   ├── configuration_transformer_rnn.py # Nirvana configuration
│   ├── task_aware_delta_net.py          # Specialized Memory Updater
│   └── ttt_cross_layer.py               # Task-Aware Trigger (with cross-layer online gradient descent)
├── specialized_ability/        # Domain-specific capabilities
│   ├── MRI_reconstruction/     # MRI image reconstruction and analysis model
│   │   ├── model/              # Custom MRI reconstruction and analysis model
│   │   ├── dataset/            # MRI dataset handling
│   │   └── train/              # MRI-specific training
└── requirements.txt            # Python dependencies
```

## 🚀 Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Conda or Miniconda
- 8+ GPUs recommended for training

### Environment Setup

1. **Clone the repository**
   ```bash
   cd Nirvana
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n nirvana python=3.10
   conda activate nirvana
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Flash Attention (optional, for enhanced performance in SWA)**
   ```bash
   pip install flash-attn==2.7.0 --no-build-isolation
   ```

### Key Dependencies

- **PyTorch**: 2.5.0
- **Transformers**: 4.52.4
- **Accelerate**: 1.1.1
- **Flash Attention**: 2.7.0.post2
- **FastMRI**: 0.3.0 (for MRI datasets)
- **WandB**: 0.21.1 (for experiment tracking)

## 🎯 Usage

### Training

#### Pre-training the Base Model

```bash
cd nirvana_backbone/train
bash train.sh
```

**Training Configuration:**
- Model: 1.3B parameters
- Data: FineWebEdu dataset
- Precision: BF16
- Distributed training with 8 GPUs
- Checkpointing every 1910 steps
- WandB integration for experiment tracking

#### MRI Reconstruction and Report Generation Training

```bash
cd specialized_ability/MRI_reconstruction
bash ./train/run_two_stage_training.sh
```

### Evaluation

#### Language Model Evaluation

```bash
cd nirvana_backbone/eval

# In-context learning evaluation
bash eval_nirvana_1.3B-icl.sh

# Long sequence evaluation
bash eval_nirvana_1.3B-longbench.sh

# Commonsense reasoning evaluation
bash eval_nirvana_1.3B-commonsense.sh

# NIAH evaluation
bash eval_nirvana_1.3B-niah.sh
```

**Supported Benchmarks:**
- S-NIAH
- LongBench
- Commonsense reasoning tasks
- FastMRI

### Model Configuration

The model configuration is defined in `nirvana_1_3B.json`:

```json
{
    "hidden_size": 2048,
    "num_heads": 16,
    "num_hidden_layers": 22,
    "max_position_embeddings": 32768,
    "vocab_size": 32000,
    "concept_dim": 64,
    "logit_dim": 32,
    "window_size": 2048
}
```

## 🔧 Customization

### Adding New Specialized Abilities

1. Create a new directory under `specialized_ability/`
2. Implement your custom models in the `model/` subdirectory
3. Add dataset handling in the `dataset/` subdirectory
4. Create training scripts in the `train/` subdirectory
5. Update the main `__init__.py` files to register your models

### Model Architecture Modifications

- **Task-Aware Delta Network**: Implement custom delta functions in `task_aware_delta_net.py`
- **Cross-Layer Connections**: Modify `ttt_cross_layer.py` for custom layer interactions
- **Transformer Variants**: Extend `modeling_transformer_rnn.py` for new architectures

## 📊 Performance

### Model Specifications

- **Parameters**: 1.3B
- **Training Context Length**: 4096 tokens
- **Training Precision**: BF16
- **Acceleration**: Flash Linear Attention
- **Parallelism**: Data, tensor, and sequence parallelism support

### Training Efficiency

- **Selective Recompute**: Configurable gradient checkpointing
- **Mixed Precision**: BF16 training with automatic mixed precision
- **Distributed Training**: Multi-GPU and multi-node support
- **Memory Optimization**: Efficient memory management with FSDP

## 🧪 Research Applications

### Medical Imaging

- **MRI Reconstruction**: Fast and accurate MRI image reconstruction from undersampled k-space data
- **Report Generation**: Automated medical report generation from MRI
- **Multi-modal Learning**: Integration of k-space, imaging, and textual data

### Foundation Model Capabilities

- **Language Understanding**: Strong performance on specialized and general language tasks
- **Task Adaptation**: Efficient adaptation for specialized applications

