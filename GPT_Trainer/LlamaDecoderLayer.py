from typing import Callable, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    is_torch_flex_attn_available,
    logging,
)
from transformers.models.llama.configuration_llama import LlamaConfig
import math
from torch.utils.checkpoint import checkpoint
# from transformers.models.mamba2.modeling_mamba2 import Mamba2Block as Mamba2Block_SM
# from transformers.models.mamba2.modeling_mamba2 import Mamba2Config

from einops import rearrange






class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Absolute Positional Encoding module.

        Args:
            d_model (int): The dimension of the embeddings.
            max_len (int): Maximum sequence length for which to compute positional encodings.
            dropout (float): Dropout rate applied to positional encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Not a parameter but persistent with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor: Input tensor with positional encoding added, same shape as input
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)






logger = logging.get_logger(__name__)
elu = torch.nn.functional.elu
relu = torch.nn.functional.relu


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, get_taylor_terms=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.attention_type = config.attention_type
        self.no_rope = False
        
        # Used to test the taylor values during inference
        self.get_taylor_terms = get_taylor_terms
        if get_taylor_terms:
            assert self.attention_type in ["softmax", "linear_elu", "linear_relu", "linear_cosine"]
            self.num_iters = 100
            self.avg_mags = [[] for i in range(0, self.num_iters)]

        self.abs_pos_enc = AbsolutePositionalEncoding(config.hidden_size, max_len=1024, dropout=0.0)

        assert self.attention_type in [
            "softmax", # Normal softmax
            "softmax_clamp_denom", # Normal softmax, clamp denominator to be at least one
            "softmax_detach_denom", # Normal softmax, detach denominator
            "softmax_detach_denom_gate", # Normal softmax, detach denominator, add a gate
            "softmax_divs", # Normal softmax, no denom divde by S
            "softmax_gate", # Normal softmax, no denom, add an output gate
            "softmax_divS_gate", # Normal softmax, no denom divde by S, add a gate
            "softmax_divS_gatev2", # Normal softmax, no denom divde by S, add a gate
            "softmax_divS_norm", # Normal softmax, no denom divde by S, add a gate

            "softmax_taylor_80terms", # 80 term taylor series exp approximation

            "gated_softmax",
            "gated_softmax_no_norm", # Gated softmax, no output norm
            "gated_softmax_no_in_gate", # Gated softmax, no input/value gate
            "gated_softmax_no_out_gate", # Gated softmax, no output gate
            "gated_softmax_no_out_gate_no_norm", # Gated softmax, no output gate, no norm
            "gated_softmax_no_in_gate_no_norm", # Gated softmax, no input gate, no norm
            "gated_relu_no_in_gate_no_norm",
            "gated_softmax_no_gate", # Gated softmax, no input or output gate
            "gated_softmax_no_gate_L2norm_nodivS", # Gated softmax, no input or output gate, change layer norm to no param L2 norm
            "gated_softmax_no_gate_L2norm_nodivS_noclamp", # Gated softmax, no input or output gate, change layer norm to no param L2 norm
            "gated_ReLU_no_gate_L2norm_nodivS_noclamp",
            "gated_softmax_out_gate_L2norm_nodivS_noclamp", # Gated softmax, no input gate, with output gate, change layer norm to no param L2 norm
            "gated_softmax_post_out_gate_L2norm_nodivS_noclamp",
            "gated_softmax_no_gate_rmsnorm", # Gated softmax, no input or output gate, change layer norm to rms norm
            "gated_softmax_no_gate_rmsnorm_nodivS", # Gated softmax, no div by S, no input or output gate, change layer norm to rms norm
            "gated_softmax_no_gate_no_norm", # Gated softmax, no input or output gate, no norm
            "gated_softmax_no_gate_customnorm",
            "gated_softmax_cumgate",
            
            "cumulative_gated_softmax", # Gated softmax but with a cumulative gate

            "gated_softmax_plusplus", # Expierment
            "gated_softmax_plusplus_extratoks",
            "gated_softmax_plusplus_mamba",
            "gated_softmax_plusplus_mamba2",

            "gated_softmax_decay",

            "linear_elu",
            "linear_relu",
            "linear_cosine",
            "linear_mamba"
        ]

        if self.attention_type == "softmax_taylor_80terms":
            self.num_terms = 80

        # Special case for mamba and rwkv
        if self.attention_type not in [
                "linear_mamba",
                "gated_softmax_plusplus_mamba",
                "gated_softmax_plusplus_mamba2"
            ]:

            if self.attention_type not in ["gated_softmax_plusplus", "gated_softmax_plusplus_extratoks"]:
                self.q_proj = nn.Linear(
                    config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
                )
                self.k_proj = nn.Linear(
                    config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
                )
                self.v_proj = nn.Linear(
                    config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
                )
            self.o_proj = nn.Linear(
                config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
            )

            # Normal softmax doesn't need anything extra

            # For softmax_detach_denom_gate, only a gate on the queries
            # as that acts like the denominator
            if self.attention_type in ["softmax_detach_denom_gate", "softmax_gate", "softmax_divS_gate", "softmax_divS_gatev2", "softmax_divS_norm"]:
                if self.attention_type in ["softmax_detach_denom_gate", "softmax_gate", "softmax_divS_gate", "softmax_divS_gatev2"]:
                    self.out_gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=True)
                elif self.attention_type in ["softmax_divS_norm"]:
                    self.out_norm = nn.RMSNorm(self.head_dim)

            # Gated softmax needs two gates and a norm
            elif self.attention_type in [
                    "gated_softmax", 
                    "gated_softmax_no_norm",
                    "gated_softmax_no_in_gate",
                    "gated_softmax_no_out_gate",
                    "gated_softmax_no_out_gate_no_norm",
                    "gated_softmax_no_in_gate_no_norm", 
                    "gated_relu_no_in_gate_no_norm",
                    "gated_softmax_no_gate",
                    "gated_softmax_no_gate_rmsnorm",
                    "gated_softmax_no_gate_L2norm_nodivS",
                    "gated_softmax_no_gate_L2norm_nodivS_noclamp",
                    "gated_ReLU_no_gate_L2norm_nodivS_noclamp",
                    "gated_softmax_out_gate_L2norm_nodivS_noclamp",
                    "gated_softmax_post_out_gate_L2norm_nodivS_noclamp",
                    "gated_softmax_no_gate_rmsnorm_nodivS",
                    "gated_softmax_no_gate_no_norm",
                    "gated_softmax_no_gate_customnorm",
                    "gated_softmax_plusplus",
                    "gated_softmax_plusplus_extratoks",
                    "gated_softmax_decay",
                    "gated_softmax_cumgate",
                ]:
                if self.attention_type not in ["gated_softmax_no_out_gate", "gated_softmax_no_out_gate_no_norm", "gated_softmax_no_gate", "gated_softmax_no_gate_no_norm", "gated_softmax_no_gate_rmsnorm", "gated_softmax_no_gate_L2norm_nodivS", "gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_ReLU_no_gate_L2norm_nodivS_noclamp", "gated_softmax_no_gate_rmsnorm_nodivS", "gated_softmax_no_gate_customnorm"]:
                    self.out_gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=True)
                if self.attention_type not in ["gated_softmax_no_in_gate", "gated_softmax_no_in_gate_no_norm", "gated_relu_no_in_gate_no_norm", "gated_softmax_no_gate", "gated_softmax_no_gate_no_norm", "gated_softmax_no_gate_rmsnorm", "gated_softmax_no_gate_L2norm_nodivS", "gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_ReLU_no_gate_L2norm_nodivS_noclamp", "gated_softmax_out_gate_L2norm_nodivS_noclamp", "gated_softmax_post_out_gate_L2norm_nodivS_noclamp", "gated_softmax_no_gate_rmsnorm_nodivS", "gated_softmax_no_gate_customnorm", "gated_softmax_cumgate"]:
                    self.in_gate_proj = nn.Linear(config.hidden_size, config.num_key_value_heads, bias=True)
                if self.attention_type in ["gated_softmax_no_out_gate_no_norm", "gated_softmax_no_in_gate_no_norm", "gated_relu_no_in_gate_no_norm", "gated_softmax_no_norm", "gated_softmax_no_gate_no_norm", "gated_softmax_cumgate"]:
                    self.out_norm = nn.Identity()
                elif self.attention_type in ["gated_softmax_no_gate_L2norm_nodivS", "gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_ReLU_no_gate_L2norm_nodivS_noclamp", "gated_softmax_out_gate_L2norm_nodivS_noclamp", "gated_softmax_post_out_gate_L2norm_nodivS_noclamp"]:
                    class L2Norm(nn.Module):
                        def __init__(self,):
                            super().__init__()
                        def forward(self, X):
                            return torch.nn.functional.normalize(X, dim=-1)
                    self.out_norm = L2Norm()
                elif self.attention_type in ["gated_softmax_no_gate_rmsnorm", "gated_softmax_no_gate_rmsnorm_nodivS"]:
                    self.out_norm = nn.RMSNorm(self.head_dim)
                elif self.attention_type == "gated_softmax_no_gate_customnorm":
                    class CustomNorm(nn.Module):
                        def __init__(self,):
                            super().__init__()

                        def forward(self, X):
                            return X.shape[-1] * torch.nn.functional.normalize(X, dim=-1)
                    self.out_norm = CustomNorm()
                else:
                    self.out_norm = nn.LayerNorm(self.head_dim)

                # Add a convolution and activation before
                if self.attention_type in ["gated_softmax_plusplus", "gated_softmax_plusplus_extratoks"]:
                    # Combine the QKV projections
                    all_dim = config.num_attention_heads * self.head_dim + 2 * config.num_key_value_heads * self.head_dim
                    self.q_size = config.num_attention_heads * self.head_dim
                    self.kv_size = config.num_key_value_heads * self.head_dim
                    self.qkv_proj = nn.Linear(
                        config.hidden_size, all_dim, bias=config.attention_bias
                    )

                    # Post convolution activation function
                    self.act = "silu"

                    # Input convolution
                    d_conv = 4
                    self.conv1d = nn.Conv1d(
                        in_channels=all_dim,
                        out_channels=all_dim,
                        bias=config.attention_bias,
                        kernel_size=d_conv,
                        groups=all_dim,
                        padding=d_conv - 1,
                    )

                    global causal_conv1d_fn
                    from causal_conv1d import causal_conv1d_fn

                if self.attention_type == "gated_softmax_decay":
                    self.no_rope = True
                    self.decay_gate_proj = nn.Linear(config.hidden_size, config.num_key_value_heads, bias=True)

            # Gated softmax needs two gates and a norm
            elif self.attention_type in [
                    "cumulative_gated_softmax", 
                ]:
                

                self.out_gate_proj = nn.Linear(config.hidden_size, config.num_attention_heads, bias=True)

            # Elu does not need anything special'

            # Cosine attention needs the power weights
            elif self.attention_type == "linear_cosine":
                self.norm_const = nn.Parameter(0.5*torch.ones(1, config.num_attention_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)

        else:
            # https://github.com/state-spaces/mamba/issues/706
            from mamba_ssm import Mamba2
            from mamba_test.Mamba2 import Mamba2_SM
            from mamba_test.mamba_softmax import Mamba2Block_SM, Mamba2Config

            if self.attention_type == "linear_mamba":
                self.mamba_layer = Mamba2(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=config.hidden_size, # Model dimension d_model
                    d_state=64,  # SSM state expansion factor, typically 64 or 128
                    d_conv=2,    # Local convolution width
                    expand=1,    # Block expansion factor
                    headdim=64,
                    D_has_hdim = True,
                    rmsnorm=True,
                )
            elif self.attention_type == "gated_softmax_plusplus_mamba":
                mamba_config = Mamba2Config(
                    head_dim=config.head_dim,
                    hidden_size=config.hidden_size,
                    state_size=config.head_dim*2,
                    num_heads=config.num_attention_heads*2,
                    n_groups=1,
                )
                self.mamba_layer = Mamba2Block_SM(
                    mamba_config,
                    layer_idx=self.layer_idx
                )
            elif self.attention_type == "gated_softmax_plusplus_mamba2":
                self.mamba_layer = Mamba2_SM(
                    # This module uses roughly 3 * expand * d_model^2 parameters
                    d_model=config.hidden_size, # Model dimension d_model
                    d_state=64,  # SSM state expansion factor, typically 64 or 128
                    d_conv=2,    # Local convolution width
                    headdim=64,
                    D_has_hdim = True,
                    use_mem_eff_path=False,
                    
                    A_proj=True,
                    no_dt=False,
                    no_D_gate=False,
                    no_z_norm=False,
                    no_in_conv=True,
                    rmsnorm=True,
                    expand=1,    # Block expansion factor
                )



    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)






        # Mamba and RWKV
        if self.attention_type in [
                "linear_mamba",
                "gated_softmax_plusplus_mamba",
                "gated_softmax_plusplus_mamba2"
            ]:
            # hidden_states = self.abs_pos_enc(hidden_states)

            def forwrd_mamba(hidden_states):
                return self.mamba_layer(hidden_states)
            
            attn_output = checkpoint(
                forwrd_mamba, hidden_states,
            )

            return attn_output, None






        # Q, K, V projections
        if self.attention_type in ["gated_softmax_plusplus", "gated_softmax_plusplus_extratoks"]:
            # Combined QKV
            QKV = self.qkv_proj(hidden_states)

            # Convolution
            QKV = causal_conv1d_fn(
                x=QKV.transpose(1, 2),
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.act,
            ).transpose(1, 2)

            # Get QKV tensors
            query_states = QKV[:, :, :self.q_size].view(hidden_shape).transpose(1, 2)
            key_states = QKV[:, :, self.q_size:self.q_size+self.kv_size].view(hidden_shape).transpose(1, 2)
            value_states = QKV[:, :, self.q_size+self.kv_size:].view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # RoPE
        if self.no_rope == False:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # if past_key_value is not None:
        #     # sin and cos are specific to RoPE models; cache_position needed for the static cache
        #     cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        #     key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Softmax attention. Just good ol flash attn
        if self.attention_type == "softmax":
            attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                **kwargs,
            )
            attn_output = attn_output.transpose(1, 2)


            if self.get_taylor_terms:
                attention_mask = torch.triu(torch.ones(1, 1, input_shape[-1], input_shape[-1])).mT.to(query_states.device).to(query_states.dtype)

                # Inner product
                attn_weights = query_states @ key_states.mT / math.sqrt(self.head_dim)

                # SUbtract max for stability
                # attn_weights = attn_weights - attn_weights.max(-1, keepdims=True).values
                # mask_ = (attention_mask == 0).to(query_states.device)
                attn_weights = (attn_weights * attention_mask).to(torch.float64)

                # Exponential
                attn_weights_ = attn_weights.clone() + 1
                cur = torch.tensor([1], dtype=torch.float64).to(query_states.device)
                self.avg_mags[0].append(torch.tensor(1, dtype=torch.float64))
                self.avg_mags[1].append((attn_weights * attention_mask).mean())
                for i in range(2, self.num_iters):
                    cur *= i
                    c = (attn_weights**i)/cur
                    attn_weights_ = attn_weights_ + c
                    self.avg_mags[i].append((c*attention_mask).abs().mean())


        elif self.attention_type in [
                "softmax_taylor_80terms",
            ]:
            def forwrd_gated(query_states, key_states, value_states, attention_mask):
                # Inner product
                attn_weights = (query_states @ key_states.mT) / math.sqrt(self.head_dim)

                # Causal mask
                causal_mask = attention_mask==0

                # Exponential polynomial expansion
                attn_weights = attn_weights.to(torch.float64)
                attn_weights_ = 1 + attn_weights.clone()
                cur = torch.tensor([1], dtype=torch.float64).to(attn_weights.device)
                iters = 80
                for i in range(2, iters):
                    cur = cur*i
                    attn_weights_ = attn_weights_ + (attn_weights**i)/cur

                # Mask
                attn_weights_ = (attn_weights_ * causal_mask)

                # Denominator
                attn_weights = (attn_weights_ / attn_weights_.sum(-1, keepdims=True)).to(query_states.dtype)

                # Output gate
                return attn_weights_.to(value_states.dtype) @ value_states

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, attention_mask
            )

        
        elif self.attention_type in [
                "softmax_clamp_denom",
                "softmax_detach_denom",
                "softmax_divs"
            ]:
            def forwrd_gated(query_states, key_states, value_states, attention_mask):
                # Inner product
                attn_weights = query_states @ key_states.mT / math.sqrt(self.head_dim)

                # Add mask
                attn_weights = attn_weights + attention_mask

                # Exponential
                attn_weights = attn_weights.clamp(max=5).exp()

                # Denominator
                denom = attn_weights.sum(-1, keepdim=True)
                if self.attention_type == "softmax_clamp_denom":
                    denom = denom.clamp(min=1)
                elif self.attention_type == "softmax_detach_denom":
                    denom = denom.clamp(min=1).detach()
                elif self.attention_type == "softmax_divs":
                    denom = (attention_mask==0).sum(-1, keepdim=True)
                attn_weights = attn_weights / denom

                # Output gate
                return attn_weights @ value_states

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, attention_mask
            )



        elif self.attention_type in ["softmax_detach_denom_gate", "softmax_gate", "softmax_divS_gate", "softmax_divS_gatev2"]:
            # Get extra gates
            out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()

            def forwrd_gated(query_states, key_states, value_states, out_gate, attention_mask):
                dtype = query_states.dtype
                
                # Inner product
                attn_weights = query_states.float() @ key_states.mT.float() / math.sqrt(self.head_dim)

                # Exponential
                attn_weights = attn_weights.clamp(max=5).exp()

                # Add mask
                causal_mask = (attention_mask.clone()==0)
                attn_weights = attn_weights * causal_mask

                # Denominator
                if self.attention_type == "softmax_detach_denom_gate":
                    denom = attn_weights.sum(-1, keepdim=True).detach() + 1e-4
                elif self.attention_type == "softmax_gate":
                    denom = torch.ones_like(attn_weights)
                elif self.attention_type == "softmax_divS_gate":
                    denom = causal_mask.sum(-1, keepdim=True)
                elif self.attention_type == "softmax_divS_gatev2":
                    # This just applies the gate directly on the weights
                    # instead of after.
                    attn_weights = (attn_weights * out_gate.float()) / causal_mask.sum(-1, keepdim=True)
                    return (attn_weights @ value_states.float()).to(dtype)
                attn_weights = attn_weights / denom#.clamp(min=1)

                # Output and gate
                return ((attn_weights @ value_states.float()) * out_gate.float()).to(dtype)
                # return (attn_weights @ value_states) * out_gate

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, out_gate, attention_mask
            )



        elif self.attention_type == "softmax_divS_norm":
            def forwrd_gated(query_states, key_states, value_states, attention_mask):
                # Inner product
                attn_weights = query_states @ key_states.mT / math.sqrt(self.head_dim)

                # Add mask
                attn_weights = attn_weights + attention_mask

                # Exponential
                attn_weights = attn_weights.clamp(max=5).exp()

                # Denominator
                attn_weights = attn_weights / (attention_mask==0).sum(-1, keepdim=True)

                # Output and gate
                return self.out_norm(attn_weights @ value_states)

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, attention_mask
            )


        elif self.attention_type in [
                "gated_softmax", 
                "gated_softmax_no_norm",
                "gated_softmax_no_in_gate",
                "gated_softmax_no_out_gate", 
                "gated_softmax_no_out_gate_no_norm",
                "gated_softmax_no_in_gate_no_norm",
                "gated_relu_no_in_gate_no_norm",
                "gated_softmax_no_gate",
                "gated_softmax_no_gate_rmsnorm",
                "gated_softmax_no_gate_rmsnorm_nodivS",
                "gated_softmax_no_gate_L2norm_nodivS",
                "gated_softmax_no_gate_L2norm_nodivS_noclamp",
                "gated_ReLU_no_gate_L2norm_nodivS_noclamp",
                "gated_softmax_out_gate_L2norm_nodivS_noclamp",
                "gated_softmax_post_out_gate_L2norm_nodivS_noclamp",
                "gated_softmax_no_gate_customnorm",
                "gated_softmax_no_gate_no_norm",
                "gated_softmax_plusplus",
                "gated_softmax_plusplus_extratoks",
            ]:
            """
            # Get extra gates
            if self.attention_type not in ["gated_softmax_no_out_gate", "gated_softmax_no_gate", "gated_softmax_no_gate_no_norm"]:
                out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            else:
                out_gate = torch.ones_like(query_states)[:, :, :, :1]
            if self.attention_type not in ["gated_softmax_no_in_gate", "gated_softmax_no_gate", "gated_softmax_no_gate_no_norm"]:
                in_gate = self.in_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            else:
                in_gate = torch.ones_like(value_states)[:, :, :, :1]

            # Mask mask binary and multiplicative instead of addative
            causal_mask = attention_mask.clone()==0

            # Attention operation
            attn_output = GatedAttention.apply(query_states.float(), key_states.float(), value_states.float(), out_gate.float(), in_gate.float(), causal_mask, self.scaling)

            # Divide output by the sequence length
            attn_output = attn_output / causal_mask.sum(-1, keepdim=True)

            # Output norm
            attn_output = self.out_norm(attn_output)
            """


            """
            # Get extra gates
            out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            in_gate = self.in_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()

            def forwrd_gated(query_states, key_states, value_states, out_gate, in_gate, attention_mask):
                dtype = query_states.dtype
                attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
                # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
                attn_weights = attn_weights.clamp(max=5).exp()  * in_gate.mT * out_gate

                causal_mask = (attention_mask==0)
                attn_weights = attn_weights * causal_mask

                attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)

                # Output gate
                # attn_output = self.out_norm(attn_output)
                return torch.matmul(attn_weights, value_states.float()).to(dtype)

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, out_gate, in_gate, attention_mask
            )

            attn_output = self.out_norm(attn_output)
            """



            #"""
            # Get extra gates
            if self.attention_type not in ["gated_softmax_no_out_gate", "gated_softmax_no_out_gate_no_norm", "gated_softmax_no_gate", "gated_softmax_no_gate_rmsnorm", "gated_softmax_no_gate_L2norm_nodivS", "gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_ReLU_no_gate_L2norm_nodivS_noclamp", "gated_softmax_no_gate_rmsnorm_nodivS", "gated_softmax_no_gate_no_norm", "gated_softmax_no_gate_customnorm"]:
                out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            else:
                out_gate = torch.ones_like(query_states)[:, :, :, :1]
            if self.attention_type not in ["gated_softmax_no_in_gate", "gated_softmax_no_in_gate_no_norm", "gated_relu_no_in_gate_no_norm", "gated_softmax_no_gate", "gated_softmax_no_gate_rmsnorm", "gated_softmax_no_gate_L2norm_nodivS", "gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_ReLU_no_gate_L2norm_nodivS_noclamp", "gated_softmax_out_gate_L2norm_nodivS_noclamp", "gated_softmax_post_out_gate_L2norm_nodivS_noclamp", "gated_softmax_no_gate_rmsnorm_nodivS", "gated_softmax_no_gate_no_norm", "gated_softmax_no_gate_customnorm"]:
                in_gate = self.in_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            else:
                in_gate = torch.ones_like(value_states)[:, :, :, :1]


            def forwrd_gated(query_states, key_states, value_states, out_gate, in_gate, attention_mask):
                # Values in gate
                value_states = value_states * in_gate

                dtype = query_states.dtype
                if self.attention_type in ["gated_ReLU_no_gate_L2norm_nodivS_noclamp", "gated_relu_no_in_gate_no_norm"]:
                    attn_weights = torch.matmul(query_states.float().relu(), key_states.float().transpose(2, 3).relu()) / math.sqrt(self.head_dim)
                else:
                    attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
                # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
                if self.attention_type in ["gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_softmax_out_gate_L2norm_nodivS_noclamp", "gated_softmax_post_out_gate_L2norm_nodivS_noclamp"]:
                    attn_weights = attn_weights.exp()
                elif self.attention_type in ["gated_ReLU_no_gate_L2norm_nodivS_noclamp"]:
                    pass
                elif self.attention_type in ["gated_relu_no_in_gate_no_norm"]:
                    attn_weights = attn_weights.clamp(max=5)
                else:
                    attn_weights = attn_weights.clamp(max=5).exp()
                # attn_weights = torch.nn.functional.silu(attn_weights.clamp(max=5))

                causal_mask = (attention_mask==0)
                attn_weights = attn_weights * causal_mask

                # attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)

                # Output gate
                # attn_output = self.out_norm(attn_output)
                if self.attention_type in ["gated_softmax_no_gate_L2norm_nodivS", "gated_softmax_no_gate_rmsnorm_nodivS", "gated_softmax_no_gate_L2norm_nodivS_noclamp", "gated_ReLU_no_gate_L2norm_nodivS_noclamp"]:
                    return out_gate * torch.matmul(attn_weights, value_states.float()).to(dtype)
                return out_gate * torch.matmul(attn_weights, value_states.float()).to(dtype) / causal_mask.sum(-1, keepdim=True)

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, out_gate, in_gate, attention_mask
            )

            attn_output = self.out_norm(attn_output)

            # Apply after
            if self.attention_type == "gated_softmax_post_out_gate_L2norm_nodivS_noclamp":
                attn_output = attn_output * out_gate
            #"""










        elif self.attention_type in [
                "gated_softmax_cumgate", 
            ]:
            #"""
            # Get extra gates
            out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None].float().exp().cumsum(-2).clamp(min=1)


            def forwrd_gated(query_states, key_states, value_states, out_gate, attention_mask):
                dtype = query_states.dtype
                attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
                # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
                attn_weights = attn_weights.clamp(max=5).exp()
                # attn_weights = torch.nn.functional.silu(attn_weights.clamp(max=5))

                causal_mask = (attention_mask==0)
                attn_weights = attn_weights * causal_mask

                # attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)

                # Output gate
                # attn_output = self.out_norm(attn_output)
                return torch.matmul(attn_weights/out_gate, value_states.float()).to(dtype)# / causal_mask.sum(-1, keepdim=True)

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, out_gate, attention_mask
            )

            attn_output = self.out_norm(attn_output)
            #"""












        elif self.attention_type in [
                "cumulative_gated_softmax", 
            ]:



            # Get extra gates
            out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None]
            # Exponentiate and cumsum
            out_gate = 1/out_gate.exp().cumsum(-1)


            def forwrd_gated(query_states, key_states, value_states, out_gate, attention_mask):
                # Values in gate
                value_states = value_states

                dtype = query_states.dtype
                attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
                # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
                attn_weights = attn_weights.clamp(max=5).exp()
                # attn_weights = torch.nn.functional.silu(attn_weights.clamp(max=5))

                causal_mask = (attention_mask==0)
                attn_weights = attn_weights * causal_mask

                # attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)

                # Output gate
                # attn_output = self.out_norm(attn_output)
                return out_gate * torch.matmul(attn_weights, value_states.float()).to(dtype) / causal_mask.sum(-1, keepdim=True)

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, out_gate, attention_mask
            )









            


        elif self.attention_type in ["gated_softmax_decay"]:
            # Get extra gates
            out_gate = self.out_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            in_gate = self.in_gate_proj(hidden_states).mT[:, :, :, None].sigmoid()
            decay_gate = self.decay_gate_proj(hidden_states).mT


            def forwrd_gated(query_states, key_states, value_states, out_gate, in_gate, decay_gate, attention_mask):
                # Move the decay gate over by 1
                decay_gate = torch.cat([
                        torch.zeros(decay_gate.shape[0], decay_gate.shape[1], 1).to(decay_gate.device).to(decay_gate.dtype),
                        decay_gate[:, :, :-1]
                    ],
                    dim=-1
                )
                # Cumulative sums for decay gates
                decay_gate_cum = decay_gate.cumsum(-1) / 1024
                # Outer summation
                decay_gate_mat = decay_gate_cum[:, :, :, None] - decay_gate_cum[:, :, None, :]
                # Exponentiate
                decay_gate_mat = decay_gate_mat.exp()
                
                # Values in gate
                value_states = value_states * in_gate

                dtype = query_states.dtype
                attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
                # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
                attn_weights = attn_weights.clamp(max=5).exp()
                # attn_weights = torch.nn.functional.silu(attn_weights.clamp(max=5))

                causal_mask = (attention_mask==0)
                attn_weights = attn_weights * causal_mask

                attn_weights = attn_weights * decay_gate_mat

                # attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)

                # Output gate
                # attn_output = self.out_norm(attn_output)
                return out_gate * torch.matmul(attn_weights, value_states.float()).to(dtype) / causal_mask.sum(-1, keepdim=True)

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, out_gate, in_gate, decay_gate, attention_mask
            )

            attn_output = self.out_norm(attn_output)


        elif self.attention_type == "linear_elu":
            # https://arxiv.org/abs/2006.16236

            def forwrd_gated(query_states, key_states, value_states, attention_mask):
                # Elu(X) + 1
                query_states = elu(query_states) + 1
                key_states = elu(key_states) + 1

                # Inner product
                attn_weights = query_states @ key_states.mT

                # Mask
                if self.training:
                    causal_mask = (attention_mask==0)
                else:
                    causal_mask = torch.triu(torch.ones(1, 1, input_shape[-1], input_shape[-1])).mT.to(query_states.device).to(query_states.dtype)
                attn_weights = attn_weights * causal_mask



                if self.get_taylor_terms:
                    attn_weights_ = (attn_weights * causal_mask).to(torch.float64)
                    # Exponential
                    attn_weights_ = attn_weights_.clone() + 1
                    cur = torch.tensor([1], dtype=torch.float64).to(query_states.device)
                    self.avg_mags[0].append(torch.tensor(1, dtype=torch.float64))
                    self.avg_mags[1].append((attn_weights * causal_mask).mean())
                    for i in range(2, self.num_iters):
                        cur *= i
                        c = (attn_weights**i)/cur
                        attn_weights_ = attn_weights_ + c
                        self.avg_mags[i].append((c*causal_mask).abs().mean())

                # Denominator
                attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)

                # Output
                return attn_weights @ value_states

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, attention_mask
            )




        elif self.attention_type == "linear_relu":
            # https://arxiv.org/abs/2410.10629

            def forwrd_gated(query_states, key_states, value_states, attention_mask):
                # Relu
                query_states = relu(query_states)
                key_states = relu(key_states)

                # Inner product
                attn_weights = query_states @ key_states.mT

                # Mask
                if self.training:
                    causal_mask = (attention_mask==0)
                else:
                    causal_mask = torch.triu(torch.ones(1, 1, input_shape[-1], input_shape[-1])).mT.to(query_states.device).to(query_states.dtype)
                attn_weights = attn_weights * causal_mask

                if self.get_taylor_terms:
                    attn_weights_ = (attn_weights * causal_mask).to(torch.float64)
                    # Exponential
                    attn_weights_ = attn_weights_.clone() + 1
                    cur = torch.tensor([1], dtype=torch.float64).to(query_states.device)
                    self.avg_mags[0].append(torch.tensor(1, dtype=torch.float64))
                    self.avg_mags[1].append((attn_weights * causal_mask).mean())
                    for i in range(2, self.num_iters):
                        cur *= i
                        c = (attn_weights**i)/cur
                        attn_weights_ = attn_weights_ + c
                        self.avg_mags[i].append((c*causal_mask).abs().mean())

                # Denominator
                attn_weights = attn_weights / (attn_weights.sum(-1, keepdim=True) + 1e-8)

                # Output
                return attn_weights @ value_states

            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, attention_mask
            )


        
        elif self.attention_type == "linear_cosine":
            def forwrd_gated(query_states, key_states, value_states, attention_mask):
                # Normalize query and key
                query_states = torch.nn.functional.normalize(query_states)
                key_states = torch.nn.functional.normalize(key_states)

                # Get binary mask
                if self.training:
                    causal_mask = (attention_mask==0)
                else:
                    causal_mask = torch.triu(torch.ones(1, 1, input_shape[-1], input_shape[-1])).mT.to(query_states.device).to(query_states.dtype)

                # Scale value
                value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())

                # Inner product
                attn_weights = (query_states @ key_states.mT) * causal_mask

                if self.get_taylor_terms:
                    attn_weights_ = (attn_weights * causal_mask).to(torch.float64)
                    # Exponential
                    attn_weights_ = attn_weights_.clone() + 1
                    cur = torch.tensor([1], dtype=torch.float64).to(query_states.device)
                    self.avg_mags[0].append(torch.tensor(1, dtype=torch.float64))
                    self.avg_mags[1].append((attn_weights * causal_mask).mean())
                    for i in range(2, self.num_iters):
                        cur *= i
                        c = (attn_weights**i)/cur
                        attn_weights_ = attn_weights_ + c
                        self.avg_mags[i].append((c*causal_mask).abs().mean())

                # Output gate
                return attn_weights @ value_states
            
            attn_output = checkpoint(
                forwrd_gated, query_states, key_states, value_states, attention_mask
            )



        else:
            assert NotImplementedError

        # Remove heads, output projection
        #### NOTE: For some reason huggingface decided to put the transpose in
        ####       the function (I think due to flash attn). I put it back here
        ####       but this means normal huggingface functions will have to be transposed :/
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)


        return attn_output, None
    














class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int, get_taylor_terms=False):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx, get_taylor_terms=get_taylor_terms)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs