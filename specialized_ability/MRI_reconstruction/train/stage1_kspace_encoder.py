"""
Stage 1: K-space encoder training with cross-entropy loss on MRI analysis tokens

This script trains only the k-space encoder (VarNet + ViT) while keeping the Nirvana backbone frozen.
The training uses cross-entropy loss on MRI analysis tokens following the SFT approach.

Author: Updated for MRI reconstruction training
Date: 2024
"""

import os
import sys
import logging
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from accelerate import Accelerator
from safetensors.torch import load_file

# local imports
from model.modeling_transformer_rnn import TransformerModel_rnn, TransformerConfig_rnn
from model.varnet_nirvana_custom import CustomNirvanaModel
from dataset.mydatasets import SliceDataset, CustomDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stage1_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Stage1KSpaceEncoderTrainer:
    """
    Stage 1 trainer for k-space encoder with cross-entropy loss on MRI analysis tokens.
    Only the k-space encoder (VarNet + ViT) is trained, while the Nirvana backbone is frozen.
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
        self.tokenizer = None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.training_history = []
        
        # Create save directory
        os.makedirs(args.save_dir, exist_ok=True)
        
    def setup_model(self):
        """Initialize the Nirvana model and freeze backbone parameters"""
        logger.info("Setting up Nirvana model for Stage 1 training...")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer()
        
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
        
        if hasattr(self.model.base_model, 'config') and hasattr(self.model.base_model.config, 'fuse_cross_entropy'):
            if self.model.base_model.config.fuse_cross_entropy:
                logger.info("fuse_cross_entropy is enabled - disabling for SFT training...")
                self.model.base_model.config.fuse_cross_entropy = False
                logger.info("fuse_cross_entropy disabled")
        
        # Freeze backbone parameters
        logger.info("Freezing backbone parameters...")
        for name, param in self.model.base_model.named_parameters():
            param.requires_grad = False
            
        # Freeze image decoder parameters
        if hasattr(self.model, 'image_decoder'):
            logger.info("Freezing image decoder parameters...")
            for name, param in self.model.image_decoder.named_parameters():
                param.requires_grad = False
        
        # Move model to device
        self.model.to(self.device)
        
        # Print model information
        self._print_model_info()
        
    def _load_tokenizer(self):
        """Load tokenizer following the reference implementation"""
        logger.info("Loading tokenizer...")
        
        try:
            # Try to load the specified tokenizer first
            tokenizer = AutoTokenizer.from_pretrained("/cpfs02/user/jiangyuhua/flash-linear-attention/training/gla_1.3B_100B/gla-1.3B-100B")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            logger.info("Successfully loaded tokenizer from gla-1.3B-100B")
        except Exception as e:
            logger.warning(f"Failed to load specified tokenizer: {e}")
            # Fallback to Qwen2.5-VL tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained("/cpfs04/shared/mabasic/jiangyuhua/models/Qwen2.5-VL-7B-Instruct")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info("Fallback: loaded Qwen2.5-VL tokenizer")
            except:
                # Final fallback to GPT2 tokenizer
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                logger.info("Fallback: loaded GPT2 tokenizer")
        
        return tokenizer
        
    def _print_model_info(self):
        """Print model size and trainable parameters information"""
        total_params = 0
        trainable_params = 0
        kspace_encoder_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if 'varnet_encoder' in name or 'vit_encoder' in name:
                    kspace_encoder_params += param.numel()
        
        logger.info(f"Model Information:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        logger.info(f"  K-space encoder parameters: {kspace_encoder_params:,}")
        logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        # Check if only k-space encoder is trainable
        non_kspace_trainable = trainable_params - kspace_encoder_params
        if non_kspace_trainable > 0:
            logger.warning("Warning: Non-k-space encoder parameters are trainable!")
        else:
            logger.info("Only k-space encoder parameters are trainable")
        
    def setup_dataset(self):
        """Setup dataset using the reference implementation pattern"""
        logger.info("Setting up dataset...")
        
        # Use the same dataset structure as the reference file
        self.dataset = CustomDataset(
            path1=self.args.data_dir+"/img",
            path2=self.args.data_dir+"/kspace",
            path3=self.args.data_dir+"/mask",
            path4=self.args.data_dir+"/text",
            mask_dir=self.args.data_dir+"/mask",
        )
        
        logger.info(f"Dataset loaded with {len(self.dataset)} samples")
        
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Only optimize trainable parameters (k-space encoder)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.95)
        )
        
        # Calculate total training steps
        total_steps = len(self.dataset) // self.args.batch_size * self.args.epochs
        warmup_steps = int(total_steps * self.args.warmup_ratio)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer and scheduler configured:")
        logger.info(f"  Learning rate: {self.args.learning_rate}")
        logger.info(f"  Weight decay: {self.args.weight_decay}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        logger.info(f"  Total steps: {total_steps}")
        
    def process_text_inputs(self, prompt, response, max_length=256):
        """Process text inputs following the reference implementation"""
        # Construct conversation format
        prompt_text = f"User: {prompt}\nAssistant: {response}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # Create labels for next token prediction
        # Shift input_ids to create labels (predict next token)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]  # Shift left by 1
        labels[-1] = -100  # Last token has no next token to predict
        
        return input_ids, attention_mask, labels
        
    def compute_loss(self, logits, labels):
        """Compute cross-entropy loss following the reference implementation"""
        # Remove the last position from logits (no next token to predict)
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., :-1].contiguous()
        
        # Flatten for loss calculation
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = labels.view(-1)
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(flat_logits, flat_labels)
        
        return loss
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # Create data loader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move data to device
                kspace = batch["kspace"].to(self.device, non_blocking=True)
                mask = batch["mask"].to(self.device, non_blocking=True)
                prompt = batch["prompt"]  # List of prompts
                response = batch["response"]  # List of responses
                
                # Process text inputs
                input_ids, attention_mask, labels = self.process_text_inputs(prompt, response)
                
                # Move text data to device
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(
                    kspace=kspace,
                    mask=mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                # Get logits
                if isinstance(outputs, dict):
                    logits = outputs.get("logits")
                else:
                    logits = getattr(outputs, 'logits', None)
                
                if logits is None:
                    logger.warning("No logits found in model outputs, skipping batch")
                    continue
                
                # Compute loss
                loss = self.compute_loss(logits, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Log progress
                if batch_idx % self.args.log_interval == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
        
    def save_checkpoint(self, filename):
        """Save checkpoint with only trainable components"""
        checkpoint_path = os.path.join(self.args.save_dir, filename)
        
        # Save only trainable components (k-space encoder)
        checkpoint = {
            'epoch': self.current_epoch,
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'training_history': self.training_history,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        # Save model state dict (only trainable parameters)
        model_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_state_dict[name] = param.data.cpu()
        
        checkpoint['model_state_dict'] = model_state_dict
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
    def run_training(self):
        """Run the complete training process"""
        logger.info("Starting Stage 1 training...")
        start_time = time.time()
        
        # Setup components
        self.setup_model()
        self.setup_dataset()
        self.setup_optimizer_and_scheduler()
        
        # Training loop
        for epoch in range(self.args.epochs):
            self.current_epoch = epoch
            logger.info(f"Starting epoch {epoch+1}/{self.args.epochs}")
            
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Update best loss
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch
                self.save_checkpoint("stage1_best_model.pt")
                logger.info(f"New best model! Loss: {avg_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.args.save_interval == 0:
                self.save_checkpoint(f"stage1_epoch_{epoch}.pt")
                
            # Save training history
            if (epoch + 1) % self.args.save_interval == 0:
                history_path = os.path.join(self.args.save_dir, "stage1_training_history.pt")
                torch.save(self.training_history, history_path)
        
        # Training completed
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("Stage 1 training completed!")
        logger.info(f"Total training time: {total_time:.2f}s")
        logger.info(f"Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")
        logger.info(f"Final model saved to: {self.args.save_dir}")
        
        # Save final model
        self.save_checkpoint("stage1_final_model.pt")

def parse_args():
    """Parse command line arguments for Stage 1 training"""
    parser = argparse.ArgumentParser(description="Stage 1: K-space encoder training")
    
    # Model configuration
    parser.add_argument("--config_path", type=str, default="./model/nirvana_1_3B.json",
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
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="Label smoothing factor")
    parser.add_argument("--l2_reg", type=float, default=0.0,
                       help="L2 regularization weight")
    
    # Data configuration
    parser.add_argument("--data_dir", type=str, default="./brain_multicoil_train",
                       help="Path to training data directory")
    parser.add_argument("--num_workers", type=int, default=8,
                       help="Number of data loading workers")
    
    # Logging and saving
    parser.add_argument("--save_dir", type=str, default="./stage1_checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N epochs")
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log training progress every N batches")
    parser.add_argument("--log_gradients", action="store_true",
                       help="Log gradient norms during training")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume training from")
    
    return parser.parse_args()

def main():
    """Main training function for Stage 1"""
    args = parse_args()
    
    # Create trainer
    trainer = Stage1KSpaceEncoderTrainer(args)
    
    # Resume training if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Run training
    trainer.run_training()

if __name__ == "__main__":
    main() 