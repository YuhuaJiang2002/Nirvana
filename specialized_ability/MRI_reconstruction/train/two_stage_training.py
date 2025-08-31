import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import argparse
from typing import Optional, Dict, Any

# fastMRI imports for data loading
from fastmri.data import subsample, mri_data
import fastmri
from fastmri.data.transforms import VarNetDataTransform
from fastmri.data.subsample import EquispacedMaskFractionFunc, EquiSpacedMaskFunc

# accelerate imports
from accelerate import Accelerator
import torch.amp as amp

# transformers and safetensors for model loading
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer

# local imports
from model.modeling_transformer_rnn import TransformerModel_rnn, TransformerConfig_rnn
from model.varnet_nirvana_custom import CustomNirvanaModel
from dataset.mydatasets import SliceDataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwoStageMRITrainer:
    """
    Two-stage trainer for MRI reconstruction with Nirvana model.
    Stage 1: Train k-space encoder with cross-entropy loss on MRI analysis tokens
    Stage 2: Train image decoder with SSIM loss on reconstructed images
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator(mixed_precision="bf16")
        
        # Initialize model components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.current_stage = 1
        self.best_loss = float('inf')
        
    def setup_model(self):
        """Initialize the Nirvana model with k-space encoder and image decoder"""
        logger.info("Setting up Nirvana model...")
        
        # Load configuration
        config = TransformerConfig_rnn.from_pretrained(self.args.config_path)
        
        # Initialize model
        self.model = CustomNirvanaModel(
            config=config,
            varnet_encoder=None,  # Will be initialized in the model
            cum_coils=self.args.num_coils,
            img_size=self.args.img_size,
            vit_embed_dim=self.args.vit_embed_dim,
            vit_num_layers=self.args.vit_num_layers,
            vit_num_heads=self.args.vit_num_heads,
            use_transformer_decoder=False,
            base_language_model=None  # Will be loaded separately
        )
        
        # Load pre-trained backbone weights
        logger.info("Loading pre-trained backbone weights...")
        backbone_weights = load_file(self.args.backbone_path, device=self.device)
        
        # Map weights to base_model
        backbone_dict = {k.replace("model.", "base_model."): v 
                        for k, v in backbone_weights.items() 
                        if k.startswith("model.")}
        
        self.model.load_state_dict(backbone_dict, strict=False)
        
        # Freeze backbone parameters
        logger.info("Freezing backbone parameters...")
        for name, param in self.model.base_model.named_parameters():
            param.requires_grad = False
            
        # Move model to device
        self.model.to(self.device)
        
        # Print model size information
        self._print_model_info()
        
    def _print_model_info(self):
        """Print model size information"""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            total_params += param.numel()
            
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
    def setup_dataset(self):
        """Setup training dataset"""
        logger.info("Setting up training dataset...")
        
        # Create mask for k-space sampling
        mask = EquispacedMaskFractionFunc(
            center_fractions=[0.08], 
            accelerations=[4]
        )
        
        # Create dataset
        self.dataset = SliceDataset(
            body_part="brain",
            partition="train",
            mask_fns=[mask],
            crop_shape=(self.args.img_size, self.args.img_size),
            slug="",
            coils=self.args.num_coils,
            root=pathlib.Path(self.args.data_dir)
        )
        
        # Create data loader
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Number of batches: {len(self.train_loader)}")
        
    def setup_training_components(self, stage: int):
        """Setup optimizer, scheduler, and loss function for specific stage"""
        if stage == 1:
            # Stage 1: Train k-space encoder only
            logger.info("Setting up training components for Stage 1 (k-space encoder)")
            
            # Only k-space encoder parameters are trainable
            trainable_params = []
            for name, param in self.model.named_parameters():
                if "varnet" in name or "vit" in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                else:
                    param.requires_grad = False
                    
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.stage1_lr,
                weight_decay=self.args.weight_decay
            )
            
            # Cross-entropy loss for MRI analysis tokens
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
            
        elif stage == 2:
            # Stage 2: Train image decoder only
            logger.info("Setting up training components for Stage 2 (image decoder)")
            
            # Freeze k-space encoder and backbone
            for name, param in self.model.named_parameters():
                if "varnet" in name or "vit" in name or "base_model" in name:
                    param.requires_grad = False
                elif "image_decoder" in name:
                    param.requires_grad = True
                    
            # Only image decoder parameters are trainable
            trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.stage2_lr,
                weight_decay=self.args.weight_decay
            )
            
            # SSIM loss for image reconstruction
            self.criterion = fastmri.SSIMLoss()
            
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        
    def train_stage_1(self):
        """Train k-space encoder with cross-entropy loss on MRI analysis tokens"""
        logger.info("Starting Stage 1: Training k-space encoder...")
        
        self.setup_training_components(stage=1)
        
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Extract k-space data and ground truth
                kspace = batch['kspace'].to(self.device)  # (B, num_coils, H, W)
                target_analysis = batch['analysis_tokens'].to(self.device)  # (B, seq_len)
                
                # Forward pass through k-space encoder and backbone
                with amp.autocast():
                    outputs = self.model(
                        kspace=kspace,
                        input_ids=target_analysis,
                        labels=target_analysis
                    )
                    
                    # Calculate cross-entropy loss on analysis tokens
                    loss = outputs.loss
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % self.args.log_interval == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate average loss
            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(f"stage1_best_epoch_{epoch}.pt")
                
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f"stage1_epoch_{epoch}.pt")
                
        logger.info("Stage 1 training completed!")
        
    def train_stage_2(self):
        """Train image decoder with SSIM loss on reconstructed images"""
        logger.info("Starting Stage 2: Training image decoder...")
        
        self.setup_training_components(stage=2)
        
        for epoch in range(self.args.epochs):
            self.model.eval()  # Set to eval mode for k-space encoder and backbone
            total_loss = 0.0
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Extract k-space data and ground truth images
                kspace = batch['kspace'].to(self.device)
                target_images = batch['target_images'].to(self.device)  # (B, H, W)
                
                # Forward pass through k-space encoder and backbone to get image tokens
                with torch.no_grad():
                    with amp.autocast():
                        # Get image tokens from backbone
                        image_tokens = self.model.generate_image_tokens(kspace)
                
                # Forward pass through image decoder
                self.model.image_decoder.train()
                with amp.autocast():
                    reconstructed_images = self.model.image_decoder(image_tokens)
                    
                    # Calculate SSIM loss
                    loss = self.criterion(reconstructed_images, target_images)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += loss.item()
                
                if batch_idx % self.args.log_interval == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Calculate average loss
            avg_loss = total_loss / len(self.train_loader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(f"stage2_best_epoch_{epoch}.pt")
                
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f"stage2_epoch_{epoch}.pt")
                
        logger.info("Stage 2 training completed!")
        
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        save_path = os.path.join(self.args.save_dir, filename)
        os.makedirs(self.args.save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_stage': self.current_stage,
            'best_loss': self.best_loss,
            'args': self.args
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_stage = checkpoint['current_stage']
        self.best_loss = checkpoint['best_loss']
        
        logger.info("Checkpoint loaded successfully")
        
    def run_training(self):
        """Run the complete two-stage training process"""
        logger.info("Starting two-stage MRI reconstruction training...")
        
        # Setup model and dataset
        self.setup_model()
        self.setup_dataset()
        
        # Stage 1: Train k-space encoder
        logger.info("=" * 50)
        logger.info("STAGE 1: Training k-space encoder")
        logger.info("=" * 50)
        self.current_stage = 1
        self.train_stage_1()
        
        # Stage 2: Train image decoder
        logger.info("=" * 50)
        logger.info("STAGE 2: Training image decoder")
        logger.info("=" * 50)
        self.current_stage = 2
        self.train_stage_2()
        
        logger.info("Two-stage training completed successfully!")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Two-stage MRI reconstruction training")
    
    # Model configuration
    parser.add_argument("--config_path", type=str, default="./nirvana_1_3B.json",
                       help="Path to Nirvana model configuration")
    parser.add_argument("--backbone_path", type=str, default="./model.safetensors",
                       help="Path to pre-trained backbone weights")
    parser.add_argument("--num_coils", type=int, default=16,
                       help="Number of MRI coils")
    parser.add_argument("--img_size", type=int, default=320,
                       help="Input image size")
    parser.add_argument("--vit_embed_dim", type=int, default=768,
                       help="ViT embedding dimension")
    parser.add_argument("--vit_num_layers", type=int, default=6,
                       help="Number of ViT layers")
    parser.add_argument("--vit_num_heads", type=int, default=12,
                       help="Number of ViT attention heads")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs per stage")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--stage1_lr", type=float, default=3e-4,
                       help="Learning rate for stage 1 (k-space encoder)")
    parser.add_argument("--stage2_lr", type=float, default=1e-4,
                       help="Learning rate for stage 2 (image decoder)")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="./brain_multicoil_train",
                       help="Path to training data directory")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    
    # Logging and saving
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log training progress every N batches")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    return parser.parse_args()

def main():
    """Main training function"""
    args = parse_args()
    
    # Create trainer
    trainer = TwoStageMRITrainer(args)
    
    # Resume training if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.run_training()

if __name__ == "__main__":
    main() 