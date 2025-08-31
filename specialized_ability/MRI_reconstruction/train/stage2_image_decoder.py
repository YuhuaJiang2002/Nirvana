import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import logging
import argparse
from typing import Optional, Dict, Any
import time
from tqdm import tqdm
import numpy as np

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
from transformers import AutoConfig, get_cosine_schedule_with_warmup

# local imports
from model.modeling_transformer_rnn import TransformerModel_rnn, TransformerConfig_rnn
from model.varnet_nirvana_custom import CustomNirvanaModel
from model.image_decoder import create_image_decoder
from dataset.mydatasets import SliceDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage2_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Stage2ImageDecoderTrainer:
    """
    Stage 2 trainer for image decoder with SSIM loss on reconstructed images.
    The k-space encoder and Nirvana backbone are frozen, only the image decoder is trained.
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
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.training_history = []
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
    def setup_model(self):
        """Initialize the Nirvana model with trained k-space encoder and frozen backbone"""
        logger.info("Setting up Nirvana model for Stage 2 training...")
        
        # Load configuration
        config = TransformerConfig_rnn.from_pretrained(self.args.config_path)
        
        # Initialize model
        self.model = CustomNirvanaModel(
            config=config,
            varnet_encoder=None,
            cum_coils=self.args.num_coils,
            img_size=self.args.img_size,
            vit_embed_dim=self.args.vit_embed_dim,
            vit_num_layers=self.args.vit_num_layers,
            vit_num_heads=self.args.vit_num_heads,
            use_transformer_decoder=False,
            base_language_model=None
        )
        
        # Load pre-trained backbone weights
        logger.info("Loading pre-trained backbone weights...")
        backbone_weights = load_file(self.args.backbone_path)
        
        # Map weights to base_model
        backbone_dict = {k.replace("model.", "base_model."): v 
                        for k, v in backbone_weights.items() 
                        if k.startswith("model.")}
        
        try:
            self.model.load_state_dict(backbone_dict, strict=False)
        except:
            pass
        # Load trained k-space encoder from Stage 1
        if self.args.stage1_checkpoint:
            logger.info("Loading trained k-space encoder from Stage 1...")
            stage1_checkpoint = torch.load(self.args.stage1_checkpoint, map_location=self.device)
            self.model.load_state_dict(stage1_checkpoint['model_state_dict'], strict=False)
        
        # Freeze backbone and k-space encoder parameters
        logger.info("Freezing backbone and k-space encoder parameters...")
        for name, param in self.model.named_parameters():
            if "base_model" in name or "varnet" in name or "vit" in name:
                param.requires_grad = False
                
        # Setup image decoder
        logger.info("Setting up image decoder...")
        self.model.image_decoder = create_image_decoder(
            decoder_type=self.args.decoder_type,
            token_dim=self.args.vit_embed_dim,
            hidden_dim=self.args.decoder_hidden_dim,
            spatial_size=self.args.img_size,
            bilinear=self.args.use_bilinear_upsample
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Print model information
        self._print_model_info()
        
    def _print_model_info(self):
        """Print model size and trainable parameters information"""
        total_params = 0
        trainable_params = 0
        image_decoder_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if "image_decoder" in name:
                    image_decoder_params += param.numel()
                    
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Image decoder parameters: {image_decoder_params:,}")
        logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        
        # Verify that only image decoder is trainable
        if trainable_params != image_decoder_params:
            logger.warning("Warning: Non-image decoder parameters are trainable!")
        else:
            logger.info("Only image decoder parameters are trainable")
            
        # Print image decoder parameter count
        if hasattr(self.model, 'image_decoder'):
            decoder_info = self.model.image_decoder.get_parameter_count()
            logger.info(f"Image decoder size: {decoder_info['total_params_millions']:.2f}M parameters")
        
    def setup_dataset(self):
        """Setup training dataset with k-space data and target images"""
        logger.info("Setting up training dataset...")
        
        # Create mask for k-space sampling
        mask = EquispacedMaskFractionFunc(center_fractions=[0.08,0.07,0.06,0.05,0.04],accelerations=[4,5,6,7,8])
        
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
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=False,
            drop_last=False
        )
        
        logger.info(f"Dataset size: {len(self.dataset)}")
        logger.info(f"Number of batches: {len(self.train_loader)}")
        logger.info(f"Batch size: {self.args.batch_size}")
        
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function for Stage 2"""
        logger.info("Setting up training components for Stage 2...")
        
        # Get trainable parameters (only image decoder)
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                
        logger.info(f"Number of trainable parameter groups: {len(trainable_params)}")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup scheduler with warmup
        total_steps = len(self.train_loader) * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # SSIM loss for image reconstruction
        self.criterion = fastmri.SSIMLoss()
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        
        logger.info(f"Training components setup completed")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.eval()  # Set to eval mode for k-space encoder and backbone
        self.model.image_decoder.train()  # Only image decoder in training mode
        
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Extract k-space data and target images
            kspace, mask_, _, target_images,_,_,_ = batch
            
            # Forward pass through k-space encoder and backbone to get image tokens
            with torch.no_grad():
                with amp.autocast():
                    # Get image tokens from backbone (this should be implemented in the model)
                    # For now, we'll use a placeholder approach
                    if hasattr(self.model, 'generate_image_tokens'):
                        image_tokens = self.model.generate_image_tokens(kspace)
                    else:
                        # Fallback: generate dummy tokens for testing
                        batch_size = kspace.shape[0]
                        num_tokens = 100
                        token_dim = self.args.vit_embed_dim
                        image_tokens = torch.randn(batch_size, num_tokens, token_dim, device=self.device)
            
            # Forward pass through image decoder
            with amp.autocast():
                reconstructed_images = self.model.image_decoder(image_tokens)
                
                # Ensure target images have the right shape for SSIM loss
                if target_images.dim() == 3:
                    target_images = target_images.unsqueeze(1)  # Add channel dimension
                
                # Calculate SSIM loss
                loss = self.criterion(reconstructed_images, target_images)
                
                # Add L2 regularization if specified
                if self.args.l2_reg > 0:
                    l2_loss = 0.0
                    for name, param in self.model.image_decoder.named_parameters():
                        if param.requires_grad:
                            l2_loss += torch.norm(param, p=2)
                    loss += self.args.l2_reg * l2_loss
            
            # Backward pass
            self.accelerator.backward(loss)
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update learning rate
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log detailed information periodically
            if batch_idx % self.args.log_interval == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
                # Log gradient norms
                if self.args.log_gradients:
                    grad_norm = self._compute_gradient_norm()
                    logger.info(f"Gradient norm: {grad_norm:.4f}")
                    
                # Log image statistics
                if self.args.log_image_stats:
                    self._log_image_statistics(reconstructed_images, target_images)
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        return avg_loss
        
    def _compute_gradient_norm(self):
        """Compute the L2 norm of gradients"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
        
    def _log_image_statistics(self, reconstructed, target):
        """Log statistics about reconstructed and target images"""
        with torch.no_grad():
            recon_mean = reconstructed.mean().item()
            recon_std = reconstructed.std().item()
            target_mean = target.mean().item()
            target_std = target.std().item()
            
            logger.info(f"Reconstructed - Mean: {recon_mean:.4f}, Std: {recon_std:.4f}")
            logger.info(f"Target - Mean: {target_mean:.4f}, Std: {target_std:.4f}")
        
    def validate(self):
        """Validate the model on validation set"""
        # This would be implemented if validation data is available
        # For now, return None
        return None
        
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        save_path = os.path.join(self.args.save_dir, filename)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history,
            'args': self.args
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.args.save_dir, "stage2_best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.best_epoch = checkpoint['best_epoch']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"Checkpoint loaded successfully from epoch {self.current_epoch}")
        
    def run_training(self):
        """Run the complete Stage 2 training process"""
        logger.info("Starting Stage 2: Image decoder training...")
        logger.info("=" * 60)
        
        # Setup model and dataset
        self.setup_model()
        self.setup_dataset()
        self.setup_training_components()
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Update training state
            self.current_epoch = epoch
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'learning_rate': self.scheduler.get_last_lr()[0]
            })
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s. Average loss: {avg_loss:.4f}")
            
            # Check if this is the best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch
                self.save_checkpoint(f"stage2_epoch_{epoch}.pt", is_best=True)
                logger.info(f"New best model! Loss: {avg_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f"stage2_epoch_{epoch}.pt")
                
            # Save training history
            if (epoch + 1) % self.args.save_interval == 0:
                history_path = os.path.join(self.args.save_dir, "stage2_training_history.pt")
                torch.save(self.training_history, history_path)
        
        # Training completed
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Stage 2 training completed!")
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
        logger.info(f"Final model saved to: {self.args.save_dir}")
        
        # Save final model
        self.save_checkpoint("stage2_final_model.pt")

def parse_args():
    """Parse command line arguments for Stage 2 training"""
    parser = argparse.ArgumentParser(description="Stage 2: Image decoder training")
    
    # Model configuration
    parser.add_argument("--config_path", type=str, default="./nirvana_1_3B.json",
                       help="Path to Nirvana model configuration")
    parser.add_argument("--backbone_path", type=str, default="./model.safetensors",
                       help="Path to pre-trained backbone weights")
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                       help="Path to Stage 1 checkpoint with trained k-space encoder")
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
    
    # Image decoder configuration
    parser.add_argument("--decoder_type", type=str, default="full",
                       choices=["full", "lightweight"],
                       help="Type of image decoder (full: 160M, lightweight: 80M)")
    parser.add_argument("--decoder_hidden_dim", type=int, default=512,
                       help="Hidden dimension for image decoder")
    parser.add_argument("--use_bilinear_upsample", action="store_true",
                       help="Use bilinear upsampling in U-Net")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--l2_reg", type=float, default=0.0,
                       help="L2 regularization weight")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="./brain_multicoil_train",
                       help="Path to training data directory")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    
    # Logging and saving
    parser.add_argument("--save_dir", type=str, default="./stage2_checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log training progress every N batches")
    parser.add_argument("--log_gradients", action="store_true",
                       help="Log gradient norms during training")
    parser.add_argument("--log_image_stats", action="store_true",
                       help="Log image statistics during training")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    return parser.parse_args()

def main():
    """Main training function for Stage 2"""
    args = parse_args()
    
    # Create trainer
    trainer = Stage2ImageDecoderTrainer(args)
    
    # Resume training if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.run_training()

if __name__ == "__main__":
    main() 