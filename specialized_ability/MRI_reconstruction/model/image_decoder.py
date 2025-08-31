import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class DoubleConv(nn.Module):
    """
    Double convolution block with batch normalization and ReLU activation
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Output convolution layer
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class TokenProjection(nn.Module):
    """
    Project image tokens to spatial features for U-Net processing
    """
    def __init__(self, token_dim: int, hidden_dim: int, spatial_size: int):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.spatial_size = spatial_size
        
        # Project tokens to hidden dimension
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        
        # Project to spatial features
        self.spatial_proj = nn.Linear(hidden_dim, spatial_size * spatial_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, tokens):
        # tokens: (B, num_tokens, token_dim)
        B, N, D = tokens.shape
        
        # Project tokens
        x = self.token_proj(tokens)
        x = self.norm(x)
        
        # Project to spatial features
        spatial_features = self.spatial_proj(x)  # (B, N, H*W)
        
        # Reshape to spatial dimensions
        spatial_features = spatial_features.view(B, N, self.spatial_size, self.spatial_size)
        
        return spatial_features

class UNetImageDecoder(nn.Module):
    """
    U-Net based image decoder for MRI reconstruction
    Target size: approximately 160M parameters
    """
    def __init__(self, 
                 token_dim: int = 768,
                 hidden_dim: int = 512,
                 spatial_size: int = 320,
                 bilinear: bool = True):
        super(UNetImageDecoder, self).__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.spatial_size = spatial_size
        self.bilinear = bilinear
        
        # Token projection layer
        self.token_projection = TokenProjection(token_dim, hidden_dim, spatial_size)
        
        # Initial convolution to process projected tokens
        self.inc = DoubleConv(hidden_dim, 64)
        
        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # Upsampling path
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output convolution
        self.outc = OutConv(64, 1)  # Single channel output for MRI
        
        # Layer normalization before and after the decoder
        self.pre_norm = nn.LayerNorm([spatial_size, spatial_size])
        self.post_norm = nn.LayerNorm([spatial_size, spatial_size])
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, tokens):
        """
        Forward pass through the U-Net decoder
        
        Args:
            tokens: (B, num_tokens, token_dim) - Image tokens from Nirvana backbone
            
        Returns:
            reconstructed_image: (B, 1, H, W) - Reconstructed MRI image
        """
        # Project tokens to spatial features
        x = self.token_projection(tokens)  # (B, N, H, W)
        
        # Apply layer normalization before processing
        x = self.pre_norm(x)
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Upsampling path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output convolution
        x = self.outc(x)
        
        # Apply layer normalization after processing
        x = self.post_norm(x)
        
        return x
    
    def get_parameter_count(self):
        """Calculate and return the total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_millions': total_params / 1e6,
            'trainable_params_millions': trainable_params / 1e6
        }

class LightweightUNetImageDecoder(nn.Module):
    """
    Lightweight U-Net decoder with reduced parameters for faster training
    Target size: approximately 80M parameters
    """
    def __init__(self, 
                 token_dim: int = 768,
                 hidden_dim: int = 256,
                 spatial_size: int = 320,
                 bilinear: bool = True):
        super(LightweightUNetImageDecoder, self).__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.spatial_size = spatial_size
        self.bilinear = bilinear
        
        # Token projection layer
        self.token_projection = TokenProjection(token_dim, hidden_dim, spatial_size)
        
        # Initial convolution
        self.inc = DoubleConv(hidden_dim, 32)
        
        # Downsampling path (reduced channels)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        
        # Upsampling path
        self.up1 = Up(256, 128, bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.up3 = Up(64, 32, bilinear)
        
        # Output convolution
        self.outc = OutConv(32, 1)
        
        # Layer normalization
        self.pre_norm = nn.LayerNorm([spatial_size, spatial_size])
        self.post_norm = nn.LayerNorm([spatial_size, spatial_size])
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, tokens):
        """
        Forward pass through the lightweight U-Net decoder
        
        Args:
            tokens: (B, num_tokens, token_dim) - Image tokens from Nirvana backbone
            
        Returns:
            reconstructed_image: (B, 1, H, W) - Reconstructed MRI image
        """
        # Project tokens to spatial features
        x = self.token_projection(tokens)
        
        # Apply layer normalization
        x = self.pre_norm(x)
        
        # Initial convolution
        x1 = self.inc(x)
        
        # Downsampling path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Upsampling path with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Output convolution
        x = self.outc(x)
        
        # Apply layer normalization
        x = self.post_norm(x)
        
        return x
    
    def get_parameter_count(self):
        """Calculate and return the total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'total_params_millions': total_params / 1e6,
            'trainable_params_millions': trainable_params / 1e6
        }

def create_image_decoder(decoder_type: str = "full", **kwargs):
    """
    Factory function to create image decoder
    
    Args:
        decoder_type: "full" for 160M parameters, "lightweight" for 80M parameters
        **kwargs: Additional arguments for decoder initialization
        
    Returns:
        Image decoder model
    """
    if decoder_type == "full":
        return UNetImageDecoder(**kwargs)
    elif decoder_type == "lightweight":
        return LightweightUNetImageDecoder(**kwargs)
    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")

if __name__ == "__main__":
    # Test the image decoder
    print("Testing Image Decoder...")
    
    # Create full decoder
    full_decoder = UNetImageDecoder()
    param_info = full_decoder.get_parameter_count()
    print(f"Full Decoder Parameters: {param_info['total_params_millions']:.2f}M")
    
    # Create lightweight decoder
    light_decoder = LightweightUNetImageDecoder()
    param_info = light_decoder.get_parameter_count()
    print(f"Lightweight Decoder Parameters: {param_info['total_params_millions']:.2f}M")
    
    # Test forward pass
    batch_size = 2
    num_tokens = 100
    token_dim = 768
    
    test_tokens = torch.randn(batch_size, num_tokens, token_dim)
    
    # Test full decoder
    output_full = full_decoder(test_tokens)
    print(f"Full decoder output shape: {output_full.shape}")
    
    # Test lightweight decoder
    output_light = light_decoder(test_tokens)
    print(f"Lightweight decoder output shape: {output_light.shape}")
    
    print("Image decoder test completed successfully!") 