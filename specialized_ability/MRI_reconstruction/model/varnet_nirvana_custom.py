import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_transformer_rnn import TransformerModel_rnn, TransformerConfig_rnn
from einops import rearrange
from typing import Optional
import math

class PatchEmbedding(nn.Module):
    """
    patch embedding
    """
    def __init__(self, img_size=320, patch_size=16, in_channels=32, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # use conv2d for patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # position embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches, embed_dim) * 0.02
        )
        
        # layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # project to patch embeddings
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        # flatten to sequence
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        
        # add position embedding
        x = x + self.pos_embedding
        
        # layer norm
        x = self.norm(x)
        
        return x

class MultiHeadAttention(nn.Module):
    """
    multi-head attention
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # calculate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # attention calculation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class TransformerBlock(nn.Module):
    """
    Transformer block
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # attention
        x = x + self.attn(self.norm1(x))
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class CustomVisionTransformer(nn.Module):
    """
    custom lightweight Vision Transformer
    """
    def __init__(self, 
                 img_size=320, 
                 patch_size=16, 
                 in_channels=32, 
                 embed_dim=768,
                 num_layers=2,
                 num_heads=12,
                 mlp_ratio=4.0,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # final normalization
        x = self.norm(x)
        
        return x

class VisionProjection(nn.Module):
    """
    vision feature projection layer
    """
    def __init__(self, vision_hidden_size: int, llm_hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(vision_hidden_size, llm_hidden_size)
        self.norm = nn.LayerNorm(llm_hidden_size)
        
    def forward(self, vision_features):
        projected = self.projection(vision_features)
        return self.norm(projected)

class TransformerDecoder(nn.Module):
    """
    4-layer Transformer decoder, for processing base_model's last_hidden_state
    """
    def __init__(self, config: TransformerConfig_rnn, num_layers=4):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        # 4层transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads if hasattr(config, 'num_attention_heads') else 16,
                mlp_ratio=4.0,
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        print(f"Created TransformerDecoder with {num_layers} layers, hidden_size={config.hidden_size}")
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (B, seq_len, hidden_size) - from base_model
            attention_mask: (B, seq_len) - optional attention mask
            
        Returns:
            hidden_states: (B, seq_len, hidden_size) - processed hidden states
        """

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states

class VarNetDecoder(nn.Module):
    """
    decoder using VarNet
    """
    def __init__(self, varnet_encoder, in_dim=2048, out_channels=32, patch_size=16, img_size=320):
        super().__init__()
        self.H = img_size
        self.W = img_size 
        self.P = patch_size
        self.varnet_encoder = varnet_encoder
        
        # upsampling layers
        self.up1 = self._make_up_layer(in_dim, in_dim//2)
        self.up2 = self._make_up_layer(in_dim//2, in_dim//4)
        self.up3 = self._make_up_layer(in_dim//4, in_dim//8)
        self.up4 = self._make_up_layer(in_dim//8, in_dim//16)
        
        self.outc = nn.Conv2d(in_dim//16, out_channels, kernel_size=1)
        
    def _make_up_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
    def forward(self, hidden_states, mask, k_space):
        # hidden_states: (B, N, D)
        B, N, D = hidden_states.shape
        H_p, W_p = self.H // self.P, self.W // self.P
        assert N == H_p * W_p, f"Tokens mismatch: {N}!={H_p*W_p}"
        
        # reshape to image shape
        x = hidden_states.transpose(1, 2).reshape(B, D, H_p, W_p)
        
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.outc(x)
        
        x = rearrange(x, 'b (c q) h w -> b c h w q', q=2)
        
        # use varnet to reconstruct
        x1 = x + k_space
        x = self.varnet_encoder(x1, mask)
        
        return x

class CustomNirvanaModel(nn.Module):
    """
    CustomNirvana model with custom VarNet and VIT encoder
    """
    def __init__(self, 
                 config: TransformerConfig_rnn,
                 varnet_encoder: nn.Module,
                 cum_coils: int = 16, 
                 img_size: int = 320,
                 vit_embed_dim: int = 768,
                 vit_num_layers: int = 6,
                 vit_num_heads: int = 12,
                 use_transformer_decoder: bool = False,
                 base_language_model: nn.Module = None):
        super().__init__()
        
        # language model - can be provided externally or created new
        if base_language_model is not None:
            print("Using provided base language model")
            self.base_model = base_language_model
            if hasattr(base_language_model, 'bfloat16'):
                self.base_model.bfloat16()
        else:
            print("Creating new TransformerModel_rnn")
            self.base_model = TransformerModel_rnn(config)
            self.base_model.bfloat16()
        
        # custom VIT encoder
        self.vit_encoder = CustomVisionTransformer(
            img_size=img_size,
            patch_size=16,
            in_channels=cum_coils * 2,  # 16 * 2 = 32
            embed_dim=vit_embed_dim,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            mlp_ratio=4.0,
            dropout=0.1
        )
        print(f"Created custom VIT encoder with {vit_embed_dim} embed_dim, {vit_num_layers} layers")
        
        # VarNet encoder
        self.varnet_encoder = varnet_encoder
        print("Using provided VarNet encoder")
        
        # vision feature projection layer
        self.vision_projection = VisionProjection(vit_embed_dim, config.hidden_size)
        
        # number of image tokens
        self.num_image_tokens = (img_size // 16) ** 2
        
        # special tokens
        self.vision_start_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.02
        )
        self.vision_end_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size, dtype=torch.bfloat16) * 0.02
        )
        
        # 4-layer Transformer decoder - create based on parameter
        if use_transformer_decoder:
            print("Creating 4-layer Transformer decoder")
            self.transformer_decoder = TransformerDecoder(config, num_layers=4)
        else:
            print("Skipping Transformer decoder creation")
            self.transformer_decoder = None
        
        # VarNet decoder 
        self.decoder = VarNetDecoder(
            varnet_encoder=varnet_encoder,
            in_dim=config.hidden_size,
            out_channels=cum_coils * 2,
            patch_size=16,
            img_size=img_size
        )
        
        # create learnable lm_head - create based on parameter
        if use_transformer_decoder:
            print("Creating learnable lm_head")
            self.learnable_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self._load_embeddings_to_lm_head()
        else:
            print("Skipping learnable lm_head creation (using original lm_head from base model)")
            self.learnable_lm_head = None
        
        print(f"Model initialized:")
        print(f"   - Custom VIT embed dim: {vit_embed_dim}")
        print(f"   - Custom VIT layers: {vit_num_layers}")
        print(f"   - Custom VIT heads: {vit_num_heads}")
        print(f"   - LLM hidden size: {config.hidden_size}")
        if use_transformer_decoder:
            print(f"   - Transformer Decoder layers: 4")
            print(f"   - Learnable LM Head shape: {self.learnable_lm_head.weight.shape}")
        else:
            print(f"   - Transformer Decoder: Disabled")
            print(f"   - Using original LM Head from base model")
        print(f"   - Image tokens: {self.num_image_tokens}")
        print(f"   - Image size: {img_size}")
        if base_language_model is not None:
            print(f"   - Using external base language model: {type(base_language_model)}")
        else:
            print(f"   - Using new TransformerModel_rnn")
    
    def _load_embeddings_to_lm_head(self):
        """
        load embeddings from base_model to learnable lm_head
        embeddings: [vocab_size, hidden_size] -> lm_head: [hidden_size, vocab_size]
        """
        if self.learnable_lm_head is None:
            print("No learnable_lm_head to load embeddings into")
            return
            
        if hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'embeddings'):
            embeddings = self.base_model.model.embeddings
            if hasattr(embeddings, 'weight'):
                # get embeddings weight and transpose
                embedding_weight = embeddings.weight  # [vocab_size, hidden_size]
                lm_head_weight = embedding_weight.T   # [hidden_size, vocab_size]
                
                # load to learnable lm_head
                with torch.no_grad():
                    self.learnable_lm_head.weight.copy_(lm_head_weight)
                
                print(f"Loaded embeddings to learnable lm_head:")
                print(f"   Embeddings shape: {embedding_weight.shape}")
                print(f"   LM Head shape: {lm_head_weight.shape}")
                print(f"   Learnable LM Head weight shape: {self.learnable_lm_head.weight.shape}")
                
                # set weight sharing (if needed)
                if hasattr(self.base_model.config, 'tie_word_embeddings') and self.base_model.config.tie_word_embeddings:
                    print("Enabling weight sharing between embeddings and learnable lm_head")
                    # can choose to keep sync update
                    
            else:
                print("Embeddings has no weight attribute")
        else:
            print("No embeddings found in base_model")
        
    def encode_kspace_with_custom_vit(self, k_space):
        """encode k-space data with custom VIT"""
        # k_space: (B, H, W, C) -> (B, C, H, W)
        if k_space.dim() == 5:  # (B, C, H, W, 2)
            k_space = rearrange(k_space, 'b c h w q -> b (c q) h w')
        elif k_space.dim() == 4:  # (B, H, W, C)
            k_space = k_space.permute(0, 3, 1, 2)
            
        # directly use custom VIT encoder
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            vision_features = self.vit_encoder(k_space)  # (B, seq_len, embed_dim)
        
        return vision_features
        
    def forward(self, kspace=None, mask=None, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        forward propagation - simplified version, mainly return logits for external loss calculation
        """
        B = kspace.size(0) if kspace is not None else input_ids.size(0)
        device = kspace.device if kspace is not None else input_ids.device
        
        # 1. encode k-space data
        if kspace is not None:
            vision_features = self.encode_kspace_with_custom_vit(kspace)  # (B, seq_len, vit_embed_dim)
            
            # project to LLM space
            vision_tokens = self.vision_projection(vision_features)  # (B, seq_len, llm_hidden_size)
            vision_tokens = vision_tokens.to(torch.bfloat16)
            
            # add vision start and end tokens
            start_tokens = self.vision_start_token.expand(B, -1, -1).to(device)
            end_tokens = self.vision_end_token.expand(B, -1, -1).to(device)
            vision_tokens = torch.cat([start_tokens, vision_tokens, end_tokens], dim=1)
        else:
            vision_tokens = None
            vision_features = None
        
        # 2. process text
        if input_ids is not None:
            text_embeds = self.base_model.get_input_embeddings()(input_ids)
            text_embeds = text_embeds.to(torch.bfloat16)
        else:
            text_embeds = None
        
        # 3. combine inputs
        if vision_tokens is not None and text_embeds is not None:
            # multimodal input: concatenate vision and text tokens
            inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
            
            # update attention mask
            if attention_mask is not None:
                vision_attention = torch.ones(B, vision_tokens.size(1), device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([vision_attention, attention_mask], dim=1)
        elif vision_tokens is not None:
            # only vision input
            inputs_embeds = vision_tokens
            attention_mask = torch.ones(B, vision_tokens.size(1), device=device, dtype=torch.long)
        else:
            # only text input
            inputs_embeds = text_embeds
        
        # 4. through language model
        # ensure not using fuse_cross_entropy, we need to get logits
        original_fuse_cross_entropy = None
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'fuse_cross_entropy'):
            original_fuse_cross_entropy = self.base_model.config.fuse_cross_entropy
            self.base_model.config.fuse_cross_entropy = False
        
        # ensure get complete logits
        kwargs_for_base_model = kwargs.copy()
        kwargs_for_base_model['num_logits_to_keep'] = inputs_embeds.shape[1]  # get all token logits
        
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs_for_base_model
        )
        
        # restore original config
        if original_fuse_cross_entropy is not None:
            self.base_model.config.fuse_cross_entropy = original_fuse_cross_entropy
        
        # 5. decode reconstruction (if k-space input)
        reconstructed_kspace = None
        if kspace is not None and hasattr(outputs, 'last_hidden_state') and vision_tokens is not None:
            # extract hidden states corresponding to vision tokens
            vision_seq_len = vision_tokens.size(1)
            vision_hidden = outputs.last_hidden_state[:, :vision_seq_len]
            
            # only use middle image tokens (remove start and end tokens)
            image_hidden = vision_hidden[:, 1:-1]  # remove start and end tokens
            
            # ensure token number is correct
            if image_hidden.size(1) != self.num_image_tokens:
                # if token number mismatch, interpolate or crop
                if image_hidden.size(1) > self.num_image_tokens:
                    image_hidden = image_hidden[:, :self.num_image_tokens]
                else:
                    # interpolate and extend
                    scale_factor = self.num_image_tokens / image_hidden.size(1)
                    image_hidden = F.interpolate(
                        image_hidden.transpose(1, 2), 
                        scale_factor=scale_factor, 
                        mode='linear'
                    ).transpose(1, 2)
                    image_hidden = image_hidden[:, :self.num_image_tokens]
            
            # decode reconstruction
            image_hidden = image_hidden.to(torch.float32)
            reconstructed_kspace = self.decoder(image_hidden, mask, kspace)
        
        # 6. calculate logits - based on config
        logits = None
        if self.transformer_decoder is not None and self.learnable_lm_head is not None:
            # use 4-layer Transformer decoder + learnable lm_head
            if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                # get base_model's last_hidden_state
                base_hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_size)
                
                # through 4-layer Transformer decoder
                decoded_hidden_states = self.transformer_decoder(
                    hidden_states=base_hidden_states,
                    attention_mask=attention_mask
                )
                
                # use learnable lm_head to calculate logits
                logits = self.learnable_lm_head(decoded_hidden_states)
                print(f"Computed logits using 4-layer decoder + learnable lm_head: {logits.shape}")
            else:
                print(f"No last_hidden_state for transformer decoder")
        else:
            # directly use base_model's original lm_head
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                logits = outputs.logits
                print(f"Using original lm_head from base model: {logits.shape}")
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                # if no direct logits, but has last_hidden_state, use base_model's lm_head
                if hasattr(self.base_model, 'lm_head'):
                    logits = self.base_model.lm_head(outputs.last_hidden_state)
                    print(f"Computed logits using base model lm_head: {logits.shape}")
                else:
                    print(f"No lm_head found in base model")
            else:
                print(f"No valid outputs for logits computation")
        
        # 7. return results
        return {
            "logits": logits,
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
            "reconstructed_kspace": reconstructed_kspace,
            "vision_features": vision_features,
            "vision_tokens": vision_tokens,
        }
    
    def generate(self, kspace=None, mask=None, input_ids=None, attention_mask=None, 
                 max_new_tokens=128, temperature=1.0, top_p=0.9, top_k=0, 
                 do_sample=True, pad_token_id=None, eos_token_id=None, **kwargs):
        """
        custom generation method, for inference
        same as forward logic: first encode kspace, then concatenate vision tokens and text embeddings
        
        Args:
            kspace: k-space data
            mask: mask data
            input_ids: input token IDs  
            attention_mask: attention mask
            max_new_tokens: maximum new tokens
            temperature: temperature parameter
            top_p: nucleus sampling parameter
            top_k: top-k sampling parameter
            do_sample: whether to sample
            pad_token_id: padding token ID
            eos_token_id: end of sequence token ID
            
        Returns:
            generated_ids: generated token IDs
        """
        B = kspace.size(0) if kspace is not None else input_ids.size(0)
        device = kspace.device if kspace is not None else input_ids.device
        
        # 1. encode vision input (same as forward method)
        vision_tokens = None
        if kspace is not None:
            print(f"Encoding k-space data for generation...")
            vision_features = self.encode_kspace_with_custom_vit(kspace)  # (B, seq_len, vit_embed_dim)
            
            # project to LLM space
            vision_tokens = self.vision_projection(vision_features)  # (B, seq_len, llm_hidden_size)
            vision_tokens = vision_tokens.to(torch.bfloat16)
            
            # add vision start and end tokens
            start_tokens = self.vision_start_token.expand(B, -1, -1).to(device)
            end_tokens = self.vision_end_token.expand(B, -1, -1).to(device)
            vision_tokens = torch.cat([start_tokens, vision_tokens, end_tokens], dim=1)
            
            print(f"Vision tokens prepared: {vision_tokens.shape}")
        
        # 2. process initial text input (same as forward method)
        if input_ids is not None:
            text_embeds = self.base_model.get_input_embeddings()(input_ids)
            text_embeds = text_embeds.to(torch.bfloat16)
            print(f"Initial text embeddings: {text_embeds.shape}")
        else:
            text_embeds = None
        
        # 3. combine initial inputs (same as forward method)
        if vision_tokens is not None and text_embeds is not None:
            # multimodal input: concatenate vision and text tokens
            current_inputs_embeds = torch.cat([vision_tokens, text_embeds], dim=1)
            
            # update attention mask
            if attention_mask is not None:
                vision_attention = torch.ones(B, vision_tokens.size(1), device=device, dtype=attention_mask.dtype)
                current_attention_mask = torch.cat([vision_attention, attention_mask], dim=1)
            else:
                current_attention_mask = torch.ones(B, current_inputs_embeds.size(1), device=device, dtype=torch.long)
        elif vision_tokens is not None:
            # only vision input
            current_inputs_embeds = vision_tokens
            current_attention_mask = torch.ones(B, vision_tokens.size(1), device=device, dtype=torch.long)
        else:
            # only text input
            current_inputs_embeds = text_embeds
            current_attention_mask = attention_mask if attention_mask is not None else torch.ones(B, text_embeds.size(1), device=device, dtype=torch.long)
        
        print(f"Combined inputs_embeds: {current_inputs_embeds.shape}")
        print(f"Combined attention_mask: {current_attention_mask.shape}")
        
        # 4. record initial sequence length (for later return results)
        if input_ids is not None:
            initial_text_length = input_ids.size(1)
        else:
            initial_text_length = 0
        
        # 5. disable fuse_cross_entropy to ensure getting logits
        original_fuse_cross_entropy = None
        if hasattr(self.base_model, 'config') and hasattr(self.base_model.config, 'fuse_cross_entropy'):
            original_fuse_cross_entropy = self.base_model.config.fuse_cross_entropy
            self.base_model.config.fuse_cross_entropy = False
        
        # 6. generate loop
        generated_tokens = []
        
        try:
            for step in range(max_new_tokens):
                with torch.no_grad():
                    # directly use base_model to generate, instead of calling the whole forward method
                    outputs = self.base_model(
                        inputs_embeds=current_inputs_embeds,
                        attention_mask=current_attention_mask,
                        num_logits_to_keep=current_inputs_embeds.shape[1]  # get all token logits
                    )
                    
                    # calculate logits - based on config
                    if self.transformer_decoder is not None and self.learnable_lm_head is not None:
                        # use 4-layer Transformer decoder + learnable lm_head to calculate logits
                        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                            # through 4-layer Transformer decoder
                            decoded_hidden_states = self.transformer_decoder(
                                hidden_states=outputs.last_hidden_state,
                                attention_mask=current_attention_mask
                            )
                            # use learnable lm_head to calculate logits
                            logits = self.learnable_lm_head(decoded_hidden_states)
                        else:
                            print(f"No last_hidden_state at step {step}")
                            break
                    else:
                        # directly use base_model's original lm_head
                        if hasattr(outputs, 'logits') and outputs.logits is not None:
                            logits = outputs.logits
                        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                            # use base_model's lm_head
                            if hasattr(self.base_model, 'lm_head'):
                                logits = self.base_model.lm_head(outputs.last_hidden_state)
                            else:
                                print(f"No lm_head found in base model at step {step}")
                                break
                        else:
                            print(f"No valid outputs at step {step}")
                            break
                    
                    # get last token's logits
                    next_token_logits = logits[:, -1, :]  # (B, vocab_size)
                    
                    # apply temperature
                    if temperature != 1.0:
                        next_token_logits = next_token_logits / temperature
                    
                    # filter logits
                    if top_k > 0:
                        # Top-k filtering
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    if top_p < 1.0:
                        # Nucleus (top-p) filtering
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        for i in range(B):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i][indices_to_remove] = float('-inf')
                    
                    # sample or greedy selection
                    if do_sample:
                        # sample
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # greedy selection
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # convert new token to embedding and add to inputs_embeds
                    next_token_embed = self.base_model.get_input_embeddings()(next_token)  # (B, 1, hidden_size)
                    next_token_embed = next_token_embed.to(torch.bfloat16)
                    
                    # update inputs_embeds and attention_mask
                    current_inputs_embeds = torch.cat([current_inputs_embeds, next_token_embed], dim=1)
                    new_attention = torch.ones(B, 1, device=device, dtype=current_attention_mask.dtype)
                    current_attention_mask = torch.cat([current_attention_mask, new_attention], dim=1)
                    
                    # record generated token
                    generated_tokens.append(next_token)
                    
                    # check if EOS token is found
                    if eos_token_id is not None:
                        if (next_token == eos_token_id).any():
                            print(f"Found EOS token at step {step}")
                            break
                    
                    if step % 10 == 0:
                        print(f"Generation step {step}, token: {next_token.item()}")
            
        finally:
            # restore original config
            if original_fuse_cross_entropy is not None:
                self.base_model.config.fuse_cross_entropy = original_fuse_cross_entropy
        
        # 7. build complete generation results
        if input_ids is not None:
            # if has initial input_ids, return complete sequence
            if generated_tokens:
                new_tokens = torch.cat(generated_tokens, dim=1)  # (B, new_length)
                generated_ids = torch.cat([input_ids, new_tokens], dim=1)
            else:
                generated_ids = input_ids
        else:
            # if no initial input_ids, only return generated tokens
            if generated_tokens:
                generated_ids = torch.cat(generated_tokens, dim=1)
            else:
                generated_ids = torch.empty(B, 0, device=device, dtype=torch.long)
        
        print(f"Generation completed: {generated_ids.shape}")
        return generated_ids 
    
if __name__ == "__main__":
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM
    config = AutoConfig.from_pretrained("./model_path", trust_remote_code=True)
    varnet_encoder = None
    cum_coils = 16
    img_size = 320
    vit_embed_dim = 768
    vit_num_layers = 6
    vit_num_heads = 12
    base_language_model = AutoModelForCausalLM.from_pretrained("./model_path", trust_remote_code=True)

    model = CustomNirvanaModel(
        config=config,
        varnet_encoder=varnet_encoder,
        cum_coils=cum_coils,
        img_size=img_size,
        vit_embed_dim=vit_embed_dim,
        vit_num_layers=vit_num_layers,
        vit_num_heads=vit_num_heads,
        use_transformer_decoder=False,
        base_language_model=base_language_model
    )

    for name, param in model.named_parameters():
        print(name, param.shape)
