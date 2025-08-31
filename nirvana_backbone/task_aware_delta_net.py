# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.modules import FusedRMSNormSwishGate, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                      fused_recurrent_gated_delta_rule)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache

def elu_p1(x):
    return (F.elu(x, 1., False) + 1.).to(x)

def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)

from fla.modules import RMSNorm, RotaryEmbedding

if TYPE_CHECKING:
    from fla.models.utils import Cache
import warnings
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         unpad_input)
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

from fla.ops.linear_attn.utils import normalize_output
# def scattering_mixer(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     v: torch.Tensor,
#     gamma: torch.Tensor,
#     # chi: torch.Tensor,
#     scale: Optional[float] = None,
#     normalize: bool = False
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     if scale is None:
#         scale = q.shape[-1] ** -0.5
#     chunk_size = 64
#     # split_size = 2
#     q = rearrange(q, 'b (n c) h d -> b h n c d', c=chunk_size) * scale
#     # k = rearrange(k, 'b h (n c) d -> b h n c d', c=chunk_size)

#     # gamma (b , n*c, h) -> (b, h, n*c, 1)
#     gamma = rearrange(gamma, 'b l h -> b h l').unsqueeze(-1)
#     gamma_cumprod = torch.cumprod(gamma, dim=2)
#     gamma_cumprod_chunk = rearrange(gamma_cumprod, 'b h (n c) d -> b h n c d', c=chunk_size)
#     gamma_cumprod_chunk = gamma_cumprod_chunk[:, :, :, -1, :].unsqueeze(-2) # [b, h, n, 1, 1]
    
#     gamma_cumprod = rearrange(gamma_cumprod, 'b h l d -> b l h d')
#     k_cumprod = k / gamma_cumprod
#     k = rearrange(k, 'b (n c) h d -> b h n c d', c=chunk_size)
#     k_cumprod_chunk = rearrange(k_cumprod, 'b (n c) h d -> b h n c d', c=chunk_size)
#     # gamma_cumprod_chunk = rearrange(gamma_cumprod, 'b h n c d -> b h (n c) d')

#     v = rearrange(v, 'b (n c) h d -> b h n c d', c=chunk_size)

#     gamma = rearrange(gamma, 'b h (n c) d -> b h n c d', c=chunk_size) # d = 1
#     # gamma_cumprod_chunk_inter = torch.cumprod(gamma, dim=3)
#     gamma_inter = torch.cumprod(gamma, dim=3) # [b, h, n, c, 1]  
    
#     kv = k_cumprod_chunk.transpose(-1, -2) @ v # [b, h, n, d, d]
#     kv = kv.cumsum(2) # [b, h, n, d, d]   n << seq_len
#     kv = kv * gamma_cumprod_chunk # [b, h, n, d, d]
    
#     kv = torch.cat([torch.zeros_like(kv[:, :, :1]), kv[:, :, :-1]], dim=2) # [b, h, n, d, d]
#     inter = (q @ kv) * gamma_inter # [b, h, n, c, d]
#     intra = (
#         ((q @ (k / gamma_inter).transpose(-1, -2)) ).masked_fill_(
#         torch.triu(torch.ones(chunk_size, chunk_size, dtype=bool, device=q.device), diagonal=1),
#         0
#     )) @ v * gamma_inter  # [b, h, n, c, d]
#     o = inter + intra # [b, h, n, c, d]
#     if normalize:
#         o = normalize_output(q * scale, k, o)
#     return rearrange(o, 'b h n c d -> b (n c) h d') , None
def scattering_mixer_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    G0: torch.Tensor,
    split_size: int,
    past_kv: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    # chi: torch.Tensor,
    scale: Optional[float] = None,
    normalize: bool = False,
    order: int = 2,
    perturb: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # chunk_size = 64
    q = rearrange(q, 'b l h (f s) -> b h l s f', s=split_size) * scale
    k = rearrange(k, 'b l h (f s) -> b h l s f', s=split_size)
    v = rearrange(v, 'b l h (d s) -> b h l s d', s=split_size)
    if order == 2:
        G0 = rearrange(G0, 'b l h d f -> b h l d f')
        # kv = k.transpose(-1, -2) @ v # [b, h, l, f, d]
        second_term = torch.einsum('b h l s d, b h l d f -> b h l s f', v, G0) # [b, h, l, s, f]
        G1 = second_term @ k.transpose(-1, -2) # [b, h, l, s, s]
        kv2 = k.transpose(-1, -2) @ G1 + k.transpose(-1, -2) # [b, h, l, f ,s]
    else:
        kv2 = k.transpose(-1, -2) # [b, h, l, f ,s]
    kv = kv2 @ v # [b, h, l, f, d]
    # kv = kv + kv2

    perturb = rearrange(perturb, 'b l h f k -> b h l f k') # [b, h, l, f, f]
    M = q.transpose(-1, -2) @ q # [b, h, l, f, f]
    M = perturb @ M # [b, h, l, f, f]
    M = q @ M # [b, h, l, s, f]
    q = q + M # [b, h, l, s, f]

    if past_kv is None:
        if beta is not None:
            beta = rearrange(beta, 'b l h -> b h l')
            beta_cumprod = torch.cumprod(beta, dim=2)
            # print('the shape of beta_cumprod', beta_cumprod.shape)
            beta_cumprod = torch.cat([torch.ones_like(beta_cumprod[:, :, :1]), beta_cumprod[:, :, :-1]], dim=2)
            # kv = kv + kv2
            beta_cumprod = rearrange(beta_cumprod, 'b h l -> b h l 1 1')
            kv = kv / beta_cumprod # [b, h, l, f, d]
            kv = kv.cumsum(2) # [b, h, l, f, d]
            kv = kv * beta_cumprod # [b, h, l, f, d]
        else:
            kv = kv.cumsum(2) # [b, h, l, f, d]
        o = q @ kv # [b, h, l, s, d]
    else:
        if beta is not None:
            beta = rearrange(beta, 'b l h -> b h l')
            kv = kv[:, :, -1, :, :] + past_kv * (beta[:, :, -2]).unsqueeze(-1).unsqueeze(-1)
        else:
            kv = kv[:, :, -1, :, :] + past_kv # [b, h, l, f, d]
        o = q @ kv # [b, h, l, s, d]
        # print('the shape of o', o.shape)
    if normalize:
        o = normalize_output(q * scale, k, o) # [b, h, l, s, d]
    return rearrange(o, 'b h l s d -> b l h (s d)') , kv

def safe_exp(x):
    return torch.exp(x - torch.max(x,dim=-1,keepdim=True)[0])

def random_proj(q, down_proj_matrix, up_proj_matrix, control_vec):
    temp = q @ down_proj_matrix
    temp = temp * control_vec
    temp = temp @ up_proj_matrix
    return torch.concat([torch.cos(temp), torch.sin(temp)], dim=-1)

def lora_proj(x, down_proj_matrix, up_proj_matrix, control_vec):
    temp = x @ down_proj_matrix
    temp = temp * control_vec
    temp = temp @ up_proj_matrix
    return temp

def gaussian_basis(x, basis_a, basis_c, basis_h):
    # x.shape = [b, q_len, channel]
    x = x.unsqueeze(-1) # [b, q_len, channel, 1]
    # basis_a.shape = [b, q_len, 1, num_basis]
    # basis_c.shape = [b, q_len, 1, num_basis]
    # basis_h.shape = [b, q_len, 1, num_basis]
    eps = 1e-6
    temp = F.sigmoid(basis_a) * torch.exp(-(x - basis_c) ** 2 / (2 * basis_h ** 2 + eps)) # [b, q_len, channel, num_basis]
    # temp = F.sigmoid(basis_a) * torch.exp(-(x - basis_c) ** 2 * (basis_h ** 2) ) # [b, q_len, channel, num_basis]
    return temp.sum(dim=-1, keepdim=False) # [b, q_len, channel]

def pad_time_cond(t, len):
    t_sin = torch.cat([torch.sin(w * t) for w in range(1, len + 1)], dim=-1)
    t_cos = torch.cat([torch.cos(w * t) for w in range(1, len + 1)], dim=-1)
    t = torch.cat([t_sin, t_cos, t], dim=-1)
    return t


class condition_interpolation(nn.Module):
    def __init__(
        self,
        hidden_size: int = 2048,
        concept_dim: int = 64,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.concept_dim = concept_dim
        self.r = 8
        # self.len = 15
        
        self.lora = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.concept_dim * 2, self.hidden_size // self.r, bias=False),
            nn.SiLU(),
            nn.Linear(self.hidden_size // self.r, self.hidden_size, bias=False)
        )
        nn.init.xavier_uniform_(self.lora[0].weight)
        nn.init.zeros_(self.lora[2].weight)

    def forward(self, start, end, h_new):
        # t = pad_time_cond(t, self.len)

        x = torch.cat([start, end, h_new, h_new], dim=-1)
        x = self.lora(x)

        return x

class Task_Aware_Delta_Net(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.
    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size
    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 256,
        num_heads: int = 6,
        num_heads_delta: int = 6,
        mode: str = 'chunk',
        use_gate: bool = True,
        use_short_conv: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.,
        max_position_embeddings: int = None,
        window_size: int = None,
        concept_dim: int = 128,
        **kwargs: Unpack[Dict]
    ) -> Task_Aware_Delta_Net:
        super().__init__()
        self.split_size = 64 # 64

        self.mode = mode
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        # self.use_short_conv = False
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.head_dim = head_dim
        self.strict_head = False
        if self.strict_head:
            head_dim_delta = int (0.75 * hidden_size / num_heads_delta)
            head_dim = head_dim_delta
            self.head_dim_delta = head_dim_delta
            self.head_dim = head_dim_delta
        self.num_heads = num_heads
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim * self.expand_v
        self.head_qk_dim = head_dim
        self.head_v_dim = head_dim * self.expand_v
        self.layer_idx = layer_idx
        self.silu = nn.SiLU()
        assert mode in ['chunk', 'fused_recurrent'], f"Not suppoerted mode `{mode}`."
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        # self.D = nn.Parameter(torch.ones(self.num_heads))
        # self.D._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                activation='silu'
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                activation='silu'
            )
        else:
            raise UserWarning(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing."
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        self.num_prelude = 2
        self.ttt = True 
        if self.ttt and self.layer_idx >= self.num_prelude: # use TTT as cross-layer concept learner
            self.concept_dim = concept_dim  # hidden_size // 8
            self.concept_proj = nn.Linear(hidden_size, self.concept_dim * 3, bias=False)
            # self.lr1_proj = nn.Linear(hidden_size, 1, bias=False)
            # self.lr2_proj = nn.Linear(hidden_size, 1, bias=False)
            # self.router  = nn.Linear(hidden_size, self.num_heads * 2, bias=False) # , bias=False
            # self.router2 = nn.Linear(self.concept_dim, self.num_heads * 2, bias=False)
            # self.router3 = nn.Linear(self.concept_dim, 2, bias=False)

            self.condition_interpolation = condition_interpolation(hidden_size, concept_dim)
            self.t_proj = nn.Linear(concept_dim, 1, bias=False)
            self.t2_proj = nn.Linear(concept_dim, 1, bias=False)

            # # self.num_basis = 2
            # # self.basis_proj = nn.Linear(self.concept_dim, self.num_basis * 3, bias=False)
            # self.special_mask = nn.Parameter(torch.zeros(self.hidden_size))
            # # self.special_mask_gated_delta = nn.Parameter(torch.zeros(self.hidden_size))
            # self.use_bias = True
            # if self.use_bias:    
            #     self.learnable_bias0 = nn.Parameter(torch.zeros(1))

        self.apply(self._initialize_weights)
        # Initialize LoRA matrices for q, k, v, and o projections using nn.Sequential
        if self.layer_idx >= self.num_prelude:
            self.r = 16
            self.q_lora = nn.Sequential(
                nn.Linear(self.hidden_size, self.key_dim // self.r, bias=False),
                # nn.SiLU(),
                nn.Linear(self.key_dim // self.r, self.key_dim, bias=False)
            )
            nn.init.xavier_uniform_(self.q_lora[0].weight)
            nn.init.zeros_(self.q_lora[1].weight)
            self.k_lora = nn.Sequential(
                nn.Linear(self.hidden_size, self.key_dim // self.r, bias=False),
                # nn.SiLU(),
                nn.Linear(self.key_dim // self.r, self.key_dim, bias=False)
            )
            nn.init.xavier_uniform_(self.k_lora[0].weight)
            nn.init.zeros_(self.k_lora[1].weight)

            self.v_lora = nn.Sequential(
                nn.Linear(self.hidden_size, self.value_dim // self.r, bias=False),
                # nn.SiLU(),
                nn.Linear(self.value_dim // self.r, self.value_dim, bias=False)
            )
            nn.init.xavier_uniform_(self.v_lora[0].weight)
            nn.init.zeros_(self.v_lora[1].weight)

            self.r2 = 8
            self.o_proj_attn = nn.Sequential(
                nn.Linear(self.value_dim, self.value_dim // self.r2, bias=False),
                # nn.SiLU(),
                nn.Linear(self.value_dim // self.r2, self.hidden_size, bias=False)
            )
            nn.init.xavier_uniform_(self.o_proj_attn[0].weight)
            nn.init.zeros_(self.o_proj_attn[1].weight)
            # self.o_proj_attn = nn.Linear(self.value_dim, self.hidden_size, bias=False)
            # nn.init.xavier_uniform_(self.o_proj_attn.weight, gain=2 ** -2.5)

            # self.rope_theta = rope_theta
            # self.max_position_embeddings = max_position_embeddings
            # self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
            self.window_size = window_size


    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values1: Optional[Cache] = None,
        all_past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        rnn_router: Optional[nn.Module] = None,
        h_old: Optional[torch.Tensor] = None,
        params: Optional[Dict] = None,
        **kwargs: Unpack[Dict]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # output: return o, None, past_key_values1, past_key_values2, h_new, params
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode
        # # mode = self.mode
        # mode = 'chunk'
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."
        last_state2 = None
        if all_past_key_values is not None:
            if all_past_key_values._seen_tokens > 0:
                past_key_values1, past_key_values2 = all_past_key_values
            else:   
                from fla.models.utils import Cache
                past_key_values1, past_key_values2 = Cache(), Cache()   

            if len(past_key_values2) > self.layer_idx:
                last_state2 = past_key_values2[self.layer_idx]
        batch_size, q_len, _ = hidden_states.size()
        cu_seqlens = kwargs.get('cu_seqlens', None)
        max_seqlen = kwargs.get('max_seqlen', q_len)
        if self.ttt:
            flag = True
            if self.layer_idx < self.num_prelude:  # first 2 layers
                if flag == True:
                    params = rnn_router.init_params_as_logits(batch_size, q_len)
                    flag = False
                # mask = torch.ones(batch_size, q_len, self.num_heads, 2, device=hidden_states.device).to(hidden_states.dtype)
                h_new = None
                # special_mask_attn = torch.zeros(batch_size, q_len, 1, device=hidden_states.device).to(hidden_states.dtype)
            else:
                concept_qkv = self.concept_proj(hidden_states)
                concept_q, concept_k, concept_v = concept_qkv.chunk(3, dim=-1)
                # lr_linear = F.sigmoid(self.lr1_proj(hidden_states)) * 1e-2
                # lr_ln = F.sigmoid(self.lr2_proj(hidden_states)) * 1e-2
                lr_linear = 1e-2
                lr_ln = 1e-2
                if rnn_router is not None:
                    params = rnn_router.learn(concept_k, concept_v, params, lr_linear, lr_ln)
                
                h_new = rnn_router.predict(concept_q, params)
                t = F.sigmoid(self.t_proj(h_new))
                # t_b = 1 - t
                t_b = F.sigmoid(self.t2_proj(h_new))

                # input_router = self.router2(h_new)
                # # input_router = nn.Softmax(dim=-1)(input_router) # [batch_size, seq_len, head_dim, 2]
                # input_router = F.sigmoid(input_router) # [batch_size, seq_len, head_dim * 2]
                # special_mask = self.router3(h_new)
                # # add bias to make the first position easier to be selected
                # bias = torch.zeros_like(special_mask)
                # bias[..., 0] = 2.0 
                # if self.use_bias:
                #     bias[..., 0] = 2.0 + self.learnable_bias0  # add positive bias to the first position, make it easier to be selected as 0
                # special_mask = F.gumbel_softmax(special_mask + bias, tau=0.1, hard=True)
                # special_mask_attn = special_mask[:, :, 1].unsqueeze(-1) # [batch_size, seq_len, 1]

                # mask = input_router
                # mask = mask.reshape(batch_size, q_len, self.num_heads, 2)
        # if self.layer_idx >= self.num_prelude:
        #     hidden_states = hidden_states + special_mask_gated_delta * self.special_mask_gated_delta.reshape(1, 1, -1)
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state2 is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state2['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            # position_ids = kwargs.get('position_ids', None)
            q_shared = self.q_proj(hidden_states)
            k_shared = self.k_proj(hidden_states)
            v_shared = self.v_proj(hidden_states)
            q, conv_state_q = self.q_conv1d(x=q_shared,
                                            mask=conv_mask,
                                            cache=conv_state_q,
                                            output_final_state=use_cache,
                                            cu_seqlens = cu_seqlens
                                            )
            k, conv_state_k = self.k_conv1d(x=k_shared,
                                            mask=conv_mask,
                                            cache=conv_state_k,
                                            output_final_state=use_cache,
                                            cu_seqlens = cu_seqlens
                                            )
            v, conv_state_v = self.v_conv1d(x=v_shared,
                                            mask=conv_mask,
                                            cache=conv_state_v,
                                            output_final_state=use_cache,
                                            cu_seqlens = cu_seqlens
                                            )
        else:
            q = self.silu(self.q_proj(hidden_states))
            k = self.silu(self.k_proj(hidden_states))
            v = self.silu(self.v_proj(hidden_states))

        # if self.layer_idx >= self.num_prelude:
        #     hidden_states_attn = hidden_states + special_mask_attn * self.special_mask.reshape(1, 1, -1)
        # else:
        #     hidden_states_attn = hidden_states
        if self.layer_idx >= self.num_prelude:
            q_attn = self.q_lora(hidden_states) + q_shared
            k_attn = self.k_lora(hidden_states) + k_shared
            v_attn = self.v_lora(hidden_states) + v_shared

            # q_attn = input_router[:, :, 1].unsqueeze(-1) * q_attn           
            q_attn, k_attn, v_attn = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (q_attn, k_attn, v_attn))
            # equivalent to cu_seqlens in `flash_attn`

            seqlen_offset = 0
            # seqlen_offset, max_seqlen = 0, q_len
            if all_past_key_values is not None:
                seqlen_offset = past_key_values1.get_seq_length(self.layer_idx)
                max_seqlen = q_attn.shape[1] + seqlen_offset

                if attention_mask is not None:
                    # to deliminate the offsets of padding tokens
                    seqlen_offset = (seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]).clamp(min=0)
                    max_seqlen = q_attn.shape[1] + max(seqlen_offset)

            # if self.max_position_embeddings is not None:
            #     max_seqlen_rotary = max(max_seqlen, self.max_position_embeddings)
            # else:
            #     max_seqlen_rotary = max_seqlen
            # q_attn, k_attn = self.rotary(q_attn, k_attn, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen_rotary, cu_seqlens=cu_seqlens)
            if all_past_key_values is not None:
                k_attn, v_attn = past_key_values1.update(
                    attn_state=(k_attn.flatten(-2, -1), v_attn.flatten(-2, -1)),
                    layer_idx=self.layer_idx,
                    offset=q_len,
                    cache_kwargs=dict(window_size=self.window_size)
                )['attn_state']
                k_attn = rearrange(k_attn, '... (h d) -> ... h d', h=self.num_heads)
                v_attn = rearrange(v_attn, '... (h d) -> ... h d', h=self.num_heads)
            if flash_attn_func is None:
                raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

            # Contains at least one padding token in the sequence
            if attention_mask is not None:
                q_attn, k_attn, v_attn, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(q_attn, k_attn, v_attn, attention_mask, q_len)
                cu_seqlens_q, cu_seqlens_k = cu_seq_lens
                max_seqlen_q, max_seqlen_k = max_seq_lens
                o_attn = flash_attn_varlen_func(
                    q_attn, k_attn, v_attn,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                )
                o_attn = pad_input(o_attn, indices_q, batch_size, q_len)
            elif cu_seqlens is not None:
                o_attn = flash_attn_varlen_func(
                    q_attn.squeeze(0), k_attn.squeeze(0), v_attn.squeeze(0),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                ).unsqueeze(0)
            else:
                o_attn = flash_attn_func(
                    q_attn, k_attn, v_attn,
                    causal=True,
                    window_size=(-1, -1) if self.window_size is None else (self.window_size-1, 0)
                ) # [total, num_heads, head_dim]   (total = batch_size * seq_len)
            if batch_size > 1:
                o_attn = o_attn.reshape(batch_size, q_len, self.num_heads, self.head_dim)

            # if self.layer_idx >= self.num_prelude:
            #     o_attn = torch.einsum("bnh,bnhd->bnhd", mask[:, :, :, 0], o_attn) # [batch_size, seq_len, num_heads, head_dim]
            
            o_attn = o_attn.reshape(batch_size, q_len, self.value_dim)
            # o_attn = self.o_proj_attention(o_attn)
            o_attn = self.o_proj_attn(o_attn) # + self.o_proj(o_attn)
        #################################################### end of attention ####################################################
        k, v = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (k, v))

        beta = self.b_proj(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias)
        # dealing with padding
        if attention_mask is not None:
            beta = beta.mul(attention_mask[:, -beta.shape[-2]:, None])
            g = g.mul(attention_mask[:, -g.shape[-2]:, None])
        recurrent_state = last_state2['recurrent_state'] if last_state2 is not None else None
        # if self.layer_idx >= self.num_prelude:
        #     # q_plus_feature = q.clone()
        #     q_safe_exp = safe_exp(q)
        #     q_plus_feature = q + q_safe_exp * if_feature_map
        #     # q_random_feature = random_proj(q, self.down_proj_matrix, self.up_proj_matrix, control_vec)
        #     # q_plus_feature = q_plus_feature + q_random_feature * if_feature_map2
        #     q_lora = lora_proj(q, self.down_proj_matrix, self.up_proj_matrix, torch.ones_like(control_vec)) # F.sigmoid(control_vec))  # F.sigmoid(control_vec)
        #     q_gaussian_feature = gaussian_basis(q_lora, basis_a, basis_c, basis_h)
        #     q_plus_feature = q_plus_feature + q_gaussian_feature * if_feature_map3

        #     q = q_plus_feature
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)

        if mode == 'chunk':
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True
            )
        elif mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                # head_first=False,
                use_qk_l2norm_in_kernel=True
            )
        if all_past_key_values is not None:
            past_key_values2.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q.shape[1]
            )
        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), '... (h d) -> ... h d', h=self.num_heads)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        # if self.layer_idx >= self.num_prelude:
        #     o = torch.einsum("bnh,bnhd->bnhd", mask[:, :, :, 1], o) # [batch_size, seq_len, num_heads, head_dim]
        o_gated_delta = rearrange(o, 'b t h d -> b t (h d)')
        o_gated_delta = self.o_proj(o_gated_delta)
        #################################################### end of delta rule ####################################################

        if self.layer_idx < self.num_prelude:
            o = o_gated_delta # + o_attn
        else:
            o = t_b * o_gated_delta + t * o_attn
            noise_std = t_b * t
            noise = self.condition_interpolation(o_gated_delta, o_attn, h_new) * noise_std
            o = o + noise

        if all_past_key_values is not None:
            all_past_key_values = (past_key_values1, past_key_values2)
        return o, None, None, all_past_key_values, h_new, params

    def _upad_input(self, q, k, v, attention_mask, q_len):
        seqlens = attention_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape

        k = index_first_axis(k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        v = index_first_axis(v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k)
        if q_len == seq_len:
            q = index_first_axis(q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k)
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return q, k, v, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)

if __name__ == "__main__":
    gated_delta_net_attention = Task_Aware_Delta_Net()
    q = torch.randn(1, 10, 6, 256)
    k = torch.randn(1, 10, 6, 256)
    v = torch.randn(1, 10, 6, 256)
    print(q.shape, k.shape, v.shape)
    # call forward function
    o, _, _, _ = gated_delta_net_attention.forward(hidden_states=torch.randn(2, 70, 128))
    print(o.shape)