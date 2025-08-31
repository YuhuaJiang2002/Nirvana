import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_transformer_rnn import TransformerModel_rnn, TransformerConfig_rnn

import math
from typing import List, Optional, Tuple

import fastmri
from fastmri.data import transforms
from fastmri.models.unet import Unet
# from fastmri.datasets import SliceDatasetLMDB

class NormUnet(nn.Module):
    """
    Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, 2, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)
        
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x
class SensitivityModel(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            if len(mask.shape) == 5:
                squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            else:
                squeezed_mask = mask[:, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )

        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(masked_kspace))

        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        )
class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModel(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace, mask, num_low_frequencies) # (B, C, H, W, 2)
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

        return fastmri.rss(fastmri.complex_abs(fastmri.ifft2c(kspace_pred)), dim=1)

class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.dc_weight = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.complex_mul(
            fastmri.ifft2c(x), fastmri.complex_conj(sens_maps)
        ).sum(dim=1, keepdim=True)

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.dc_weight
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1):
        x1 = self.up(x1)
        # # calculate padding to ensure size matching
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
    
from einops import rearrange

class KSpaceEncoder(nn.Module):
    """
    K-space encoder combining VarNet with lightweight ViT.
    This extracts features from k-space signals and generates k-space tokens.
    """
    def __init__(self, in_chans=32, img_size=320, vit_embed_dim=768, 
                 vit_num_layers=2, vit_num_heads=12):
        super().__init__()
        self.in_chans = in_chans
        self.img_size = img_size
        self.vit_embed_dim = vit_embed_dim
        
        # VarNet for initial k-space processing
        self.varnet = VarNet(
            num_cascades=12,
            sens_chans=8,
            sens_pools=4,
            chans=18,
            pools=4,
            mask_center=True
        )
        
        # Lightweight ViT for feature extraction
        self.vit = LightweightViT(
            in_channels=in_chans,
            img_size=img_size,
            embed_dim=vit_embed_dim,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads
        )
        
    def forward(self, kspace):
        """
        Forward pass through k-space encoder.
        
        Args:
            kspace: K-space signals (B, num_coils, H, W, 2)
            
        Returns:
            kspace_tokens: Extracted features (B, N, vit_embed_dim)
        """
        # Process k-space with VarNet
        processed_kspace = self.varnet(kspace, mask=None)
        
        # Extract features with ViT
        kspace_tokens = self.vit(processed_kspace)
        
        return kspace_tokens

class LightweightViT(nn.Module):
    """
    Lightweight Vision Transformer for k-space feature extraction.
    """
    def __init__(self, in_channels=32, img_size=320, embed_dim=768, 
                 num_layers=2, num_heads=12, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim) * 0.02
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Forward pass through lightweight ViT.
        
        Args:
            x: Input features (B, C, H, W)
            
        Returns:
            tokens: Extracted tokens (B, N, embed_dim)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # Flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Final normalization
        x = self.norm(x)
        
        return x

class NormUNetImageDecoder(nn.Module):
    """
    160M parameter NormUNet image decoder for MRI reconstruction.
    """
    def __init__(self, token_dim=768, img_size=320, out_chans=1):
        super().__init__()
        self.token_dim = token_dim
        self.img_size = img_size
        self.out_chans = out_chans
        
        # Token projection to spatial features
        self.token_proj = nn.Linear(token_dim, img_size * img_size)
        
        # NormUNet architecture
        self.norm_unet = NormUnet(
            chans=64,
            num_pools=4,
            in_chans=1,
            out_chans=out_chans,
            drop_prob=0.1
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm([img_size, img_size])
        
    def forward(self, tokens):
        """
        Forward pass through NormUNet decoder.
        
        Args:
            tokens: Image tokens (B, N, token_dim)
            
        Returns:
            reconstructed_image: Reconstructed MRI image (B, out_chans, H, W)
        """
        B, N, D = tokens.shape
        
        # Project tokens to spatial features
        spatial_features = self.token_proj(tokens)  # (B, N, H*W)
        
        # Reshape to spatial dimensions
        spatial_features = spatial_features.view(B, N, self.img_size, self.img_size)
        
        # Apply normalization
        spatial_features = self.norm(spatial_features)
        
        # Process with NormUNet
        reconstructed_image = self.norm_unet(spatial_features)
        
        return reconstructed_image

class Encoder(nn.Module):
    def __init__(self, in_chans=32, patch_size=(16,16), embed_dim=2048):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        # Unfold: kernel_size = patch_size, stride = patch_size
        
        # self.varnet = VarNet()
        self.inc = DoubleConv(self.in_chans, 128)
        # self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)
        # # linear layer: project the unfolded patch to hidden_dim
        self.proj = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        

    def forward(self, x, mask):
        # x: (batch_size, 2, 640, 368)
        x = rearrange(x, 'b c h w q -> b (c q) h w')
        x = self.inc(x)
        # x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        # x shape: (B, 2048, H//16, W//16) 
        # x merge the last two dimensions
        x = x.flatten(-2)  # (B, 2048, H//16*W//16)
        # 2) transpose to (B, N_tokens, 2048)
        x = x.transpose(1, 2)  # (B, 920, 512)
        # 3) linear mapping to hidden_dim
        x = self.proj(x)   # (B, 920, 2048)
        # x = self.norm(x)
        return x

def center_crop_mask(x, target_size=320):
    """
    center crop the input tensor x to the target size target_size
    """
    assert x.shape[2] >= target_size, f"x.shape[2] must be >= target_size, but got {x.shape[2]} and target_size {target_size}"
    center_pos = x.shape[2] // 2
    half_size = target_size // 2
    return x[:, :, center_pos-half_size:center_pos+half_size, :] 

def center_crop_img(x, target_size=320):
    """
    center crop the input tensor x to the target size target_size
    """
    assert x.shape[2] >= target_size, f"x.shape[2] must be >= target_size, but got {x.shape[2]} and target_size {target_size}"
    center_pos = x.shape[2] // 2
    half_size = target_size // 2
    return x[:, :, center_pos-half_size:center_pos+half_size, center_pos-half_size:center_pos+half_size, :]

class Decoder(nn.Module):
    def __init__(self, in_dim=2048, out_channels=32, patch_size=16,
                 img_size=320, eps:float=1e-6, varscale:float=1,num_cascades:int=24):
        """
        reshape tokens (B, N, in_dim) → image (B, 1, H, W)
        """
        super().__init__()
        self.H = img_size
        self.W = img_size
        self.P = patch_size
        # inverse Tokenization + transpose convolution
        self.up1 = Up(in_channels=in_dim   , out_channels=in_dim//2, bilinear=True)
        self.up2 = Up(in_channels=in_dim//2, out_channels=in_dim//4, bilinear=True)
        self.up3 = Up(in_channels=in_dim//4, out_channels=in_dim//8, bilinear=True)
        self.up4 = Up(in_channels=in_dim//8, out_channels=in_dim//16, bilinear=True)
        
        self.outc = nn.Conv2d(in_dim//16, out_channels, kernel_size=1)

        self.varnet = VarNet(
        num_cascades=num_cascades,
        sens_chans=16,
        sens_pools=4,
        chans=18,
        pools=4,
    )

    def forward(self, hidden_states, mask, k_space):
        # hidden_states: (B, N, D)
        B, N, D = hidden_states.shape
        H_p, W_p = self.H // self.P, self.W // self.P
        assert N == H_p * W_p, f"Tokens mismatch: {N}!={H_p*W_p}"
        x = hidden_states.transpose(1, 2).reshape(B, D, H_p, W_p)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)   # -> (B, out_channels, H, W)
        # recover to the target level
        x = rearrange(x, 'b (c q) h w -> b c h w q', q=2)
        # diff = x1 - k_space        
        x = self.varnet(x, mask)
        
        return x  

class CustomNirvanaModel(nn.Module):
    """
    Custom Nirvana model for MRI reconstruction with two-stage training capability.
    
    Architecture:
    1. K-space encoder: VarNet + lightweight ViT to extract features from k-space signals
    2. Nirvana backbone: Frozen transformer for generating image and analysis tokens
    3. Image decoder: 160M NormUNet for reconstructing MRI images
    """
    def __init__(self, config: TransformerConfig_rnn,
                 cum_coils=16, img_size=320, vit_embed_dim=768, 
                 vit_num_layers=2, vit_num_heads=12, use_transformer_decoder=False,
                 base_language_model=None):
        super().__init__()
        
        # Nirvana backbone (frozen during training)
        if base_language_model is not None:
            self.base_model = base_language_model
        else:
            self.base_model = TransformerModel_rnn(config)
        
        # K-space encoder: VarNet + lightweight ViT
        self.kspace_encoder = KSpaceEncoder(
            in_chans=cum_coils * 2,  # 16 coils * 2 (real + imaginary)
            img_size=img_size,
            vit_embed_dim=vit_embed_dim,
            vit_num_layers=vit_num_layers,
            vit_num_heads=vit_num_heads
        )
        
        # Image decoder: 160M NormUNet
        self.image_decoder = NormUNetImageDecoder(
            token_dim=vit_embed_dim,
            img_size=img_size,
            out_chans=1  # Single channel output for MRI
        )
        
        # Layer normalization for training stability
        self.kspace_norm = nn.LayerNorm(vit_embed_dim)
        self.image_norm = nn.LayerNorm([img_size, img_size])
        
        # Token generation parameters
        self.num_image_tokens = (img_size // 16) ** 2
        self.img_token_proj = nn.Linear(vit_embed_dim, config.hidden_size)
        
        # Store configuration
        self.cum_coils = cum_coils
        self.img_size = img_size
        self.vit_embed_dim = vit_embed_dim
        
    def forward(self, kspace=None, input_ids=None, labels=None, **kwargs):
        """
        Forward pass for training and inference.
        
        Args:
            kspace: K-space signals (B, num_coils, H, W, 2)
            input_ids: Instruction prompt tokens (B, seq_len)
            labels: Target tokens for loss calculation (B, seq_len)
            **kwargs: Additional arguments for base model
            
        Returns:
            outputs: Model outputs with loss if labels provided
        """
        if kspace is not None and input_ids is not None:
            return self._forward_with_kspace(kspace, input_ids, labels, **kwargs)
        elif input_ids is not None:
            return self._forward_text_only(input_ids, labels, **kwargs)
        else:
            raise ValueError("Either kspace or input_ids must be provided")
    
    def _forward_with_kspace(self, kspace, input_ids, labels, **kwargs):
        """Forward pass with k-space input for Stage 1 training"""
        B = kspace.size(0)
        
        # Extract features from k-space using VarNet + ViT
        kspace_tokens = self.kspace_encoder(kspace)  # (B, N, vit_embed_dim)
        kspace_tokens = self.kspace_norm(kspace_tokens)
        
        # Project k-space tokens to backbone hidden dimension
        kspace_tokens_proj = self.img_token_proj(kspace_tokens)  # (B, N, hidden_size)
        
        # Get text embeddings from base model
        text_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        # Concatenate k-space tokens with text embeddings
        combined_embeds = torch.cat([kspace_tokens_proj, text_embeds], dim=1)
        
        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def _forward_text_only(self, input_ids, labels, **kwargs):
        """Forward pass with text only for standard language modeling"""
        return self.base_model(
            input_ids=input_ids,
            labels=labels,
            **kwargs
        )
    
    def generate_image_tokens(self, kspace):
        """
        Generate image tokens from k-space signals for Stage 2 training.
        This method is used when the k-space encoder is trained and frozen.
        """
        with torch.no_grad():
            # Extract features from k-space
            kspace_tokens = self.kspace_encoder(kspace)
            kspace_tokens = self.kspace_norm(kspace_tokens)
            
            # Project to backbone dimension
            kspace_tokens_proj = self.img_token_proj(kspace_tokens)
            
            # Generate image tokens using the backbone
            # For now, return the projected tokens as image tokens
            # In a full implementation, this would generate tokens through the backbone
            return kspace_tokens_proj
    
    def reconstruct_image(self, image_tokens):
        """
        Reconstruct MRI image from image tokens using the NormUNet decoder.
        This method is used in Stage 2 training.
        """
        # Apply normalization
        image_tokens = self.image_norm(image_tokens)
        
        # Reconstruct image using NormUNet
        reconstructed_image = self.image_decoder(image_tokens)
        
        return reconstructed_image
    
    def get_trainable_parameters(self, stage=1):
        """
        Get trainable parameters based on training stage.
        
        Args:
            stage: Training stage (1: k-space encoder, 2: image decoder)
            
        Returns:
            List of trainable parameters
        """
        if stage == 1:
            # Stage 1: Only k-space encoder is trainable
            return list(self.kspace_encoder.parameters())
        elif stage == 2:
            # Stage 2: Only image decoder is trainable
            return list(self.image_decoder.parameters())
        else:
            raise ValueError("Stage must be 1 or 2")
    
    def freeze_parameters(self, stage=1):
        """
        Freeze parameters based on training stage.
        
        Args:
            stage: Training stage (1: freeze backbone and image decoder, 2: freeze backbone and k-space encoder)
        """
        if stage == 1:
            # Freeze backbone and image decoder
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.image_decoder.parameters():
                param.requires_grad = False
            # K-space encoder remains trainable
        elif stage == 2:
            # Freeze backbone and k-space encoder
            for param in self.base_model.parameters():
                param.requires_grad = False
            for param in self.kspace_encoder.parameters():
                param.requires_grad = False
            # Image decoder remains trainable
        else:
            raise ValueError("Stage must be 1 or 2")

