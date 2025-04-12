import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv
from transformers.models.llama.modeling_llama import ACT2FN
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from kernel.Gated_Attn import GatedAttention

try:
    from GPT_Trainer.multi_gpu_helpers import is_main_process
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
import wandb

import numpy as np
import torch
import sys
import warnings


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
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











class LlamaMLPGeLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.hidden_inner = int(self.intermediate_size + (self.hidden_size * self.intermediate_size)**0.5)

        self.up_proj = nn.Linear(self.hidden_size, self.hidden_inner, bias=False)
        self.down_proj = nn.Linear(self.hidden_inner, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(torch.nn.functional.gelu(torch.nn.functional.normalize(self.up_proj(x) * self.hidden_inner**0.5, p=2, dim=-1)))


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









class LlamaCosAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_type = config.attention_type

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # TODO (joao): remove in v4.46 (RoPE is computed in the model, not in the decoder layers)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)


        # Learnable constant for each head for norm
        # self.norm_const = nn.Parameter(0.5*torch.ones(1, self.num_key_value_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)
        # self.norm_const = nn.Parameter(torch.ones(1, self.num_key_value_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)
        # self.base = nn.Parameter(0.2*torch.ones(1, self.num_key_value_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)
        # self.norm_const2 = nn.Parameter(2*torch.ones(1, self.num_key_value_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)

        # self.scalars = nn.Parameter(torch.ones([80], dtype=torch.float32))

        # self.lp_const = nn.Parameter(2*torch.ones(1, self.num_key_value_heads, 1, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)

        # self.learned_weight = nn.Linear(self.head_dim, 1).to(self.q_proj.weight.device)

        # self.base = nn.Parameter(torch.e*torch.ones(1, self.num_key_value_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)



        # Softmax V2
        # self.weight_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.weight_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.weight_proj2 = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        # self.norm_const = nn.Parameter(0.5*torch.ones(1, self.num_key_value_heads, 1, 1, dtype=self.q_proj.weight.dtype)).to(self.q_proj.weight.device)
        self.out_norm = nn.LayerNorm(self.head_dim)


        # # value conv and key conv with their masks
        # self.value_conv = nn.Conv1d(self.head_dim, self.head_dim, 3, padding=1).to(self.q_proj.weight.device)
        # self.key_conv = nn.Conv1d(self.head_dim, self.head_dim, 3, padding=1).to(self.q_proj.weight.device)
        # # key mask looks one into the past (1, 1, 0)
        # self.register_buffer("key_mask", torch.cat((torch.ones(self.head_dim, self.head_dim, 2), torch.zeros(self.head_dim, self.head_dim, 1)), dim=-1))
        # # value mask looks one into the future (0, 1, 1)
        # self.register_buffer("value_mask", torch.cat((torch.zeros(self.head_dim, self.head_dim, 1), torch.ones(self.head_dim, self.head_dim, 2)), dim=-1))







    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        
        
        attn_mode = "softmaxV2"
        attention_mask = attention_mask.clone()

        # if attention_mask is None:
        #     attention_mask = torch.where(~torch.tril(torch.ones(q_len, q_len, device=key_states.device, dtype=torch.bool)), -torch.inf, 0)[None, None, :, :]

        if attn_mode == "cosine":
            """ Normal cosine attention
            # Normalize query, and keys
            query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            # Divisor
            value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """




            """ Normal cosine attention Div
            # Get query and key norms
            query_norm = torch.norm(query_states, dim=-1, p=2, keepdim=True)
            key_norm = torch.norm(key_states, dim=-1, p=2, keepdim=True)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights_denom = query_norm * key_norm.transpose(2, 3)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask
                attn_weights_denom = attn_weights_denom * causal_mask

            # Weight normalization
            attn_weights = attn_weights / attn_weights_denom.sum(-1, keepdim=True)

            # Divisor
            # value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """



            """ Normal cosine attention Div exp
            # Get query and key norms
            query_norm = torch.norm(query_states, dim=-1, p=2, keepdim=True)
            key_norm = torch.norm(key_states, dim=-1, p=2, keepdim=True)

            # Normalize query, and keys
            query_states = query_states / query_norm
            key_states = key_states / key_norm

            # Attention weights
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights_denom = query_norm * key_norm.transpose(2, 3)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask
                attn_weights_denom = attn_weights_denom * causal_mask

            # Weight normalization
            attn_weights = (attn_weights_denom * attn_weights) / attn_weights_denom.sum(-1, keepdim=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """


            
            """ # Normalize query, and keys
            query_mag = torch.norm(query_states, dim=-1, p=2, keepdim=True)
            query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            key_mag = torch.norm(key_states, dim=-1, p=2, keepdim=True)
            key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            coef = (query_mag / math.sqrt(self.head_dim)).exp() * (key_mag / math.sqrt(self.head_dim)).exp() * (torch.tensor(-1.0, device=query_mag.device)).exp()
            # coef = (query_mag*key_mag.mT)/math.sqrt(self.head_dim)

            attn_weights = coef * (torch.matmul(query_states, key_states.transpose(2, 3)) + 1)  # + coef # + 1/(query_mag*key_mag)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights * (causal_mask==0).detach()

            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)
            # attn_weights = attn_weights / attn_weights.norm(dim=-1, keepdim=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """




            """ Only mag
            # Get query and key norms
            query_norm = torch.norm(query_states, dim=-1, p=2, keepdim=True)
            key_norm = torch.norm(key_states, dim=-1, p=2, keepdim=True)

            # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights_denom = (query_norm * key_norm.transpose(2, 3)).exp()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                # attn_weights = attn_weights * causal_mask
                attn_weights_denom = attn_weights_denom * causal_mask

            # Weight normalization
            attn_weights = attn_weights_denom / attn_weights_denom.sum(-1, keepdim=True)

            # Divisor
            # value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """







            """ Cosine softmax hybrid
            # Get query and key norms
            # query_norm = torch.norm(query_states, dim=-1, p=2, keepdim=True)
            # key_norm = torch.norm(key_states, dim=-1, p=2, keepdim=True)
            query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            # attn_weights_denom = (query_norm * key_norm.transpose(2, 3)).exp()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                # attn_weights = attn_weights * causal_mask
                # attn_weights_denom = attn_weights_denom * causal_mask

            # Weight normalization
            # attn_weights = attn_weights_denom / attn_weights_denom.sum(-1, keepdim=True)

            # Where are the weights high?
            mask = (torch.diagonal(attn_weights, dim1=-2, dim2=-1).unsqueeze(-2) > self.base).repeat(1, 1, attn_weights.shape[-2], 1)
            # Exponentiate the weights
            attn_weights = torch.where(mask, attn_weights.exp(), attn_weights) * self.base


            # Divisor
            value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            # def forward_pass(query_states, key_states, value_states, attention_mask):
            #     # Normalize query and key states
            #     query_states = F.normalize(query_states, dim=-1, p=2)
            #     key_states = F.normalize(key_states, dim=-1, p=2)

            #     # Compute attention weights
            #     attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            #     if attention_mask is not None:
            #         causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            #         causal_mask = (causal_mask == 0).detach()

            #     # Apply base scaling
            #     mask = (
            #         torch.diagonal(attn_weights, dim1=-2, dim2=-1)
            #         .unsqueeze(-2)
            #         > self.base
            #     ).repeat(1, 1, attn_weights.shape[-2], 1)
            #     attn_weights = torch.where(mask, attn_weights.exp(), attn_weights) * self.base

            #     # Normalize value states based on causal mask
            #     if attention_mask is not None:
            #         value_states = value_states / (
            #             causal_mask.sum(-1, keepdim=True) ** self.norm_const.sigmoid()
            #         )

            #     # Mask attention weights
            #     attn_weights = attn_weights * causal_mask

            #     # Compute attention output
            #     attn_output = torch.matmul(attn_weights, value_states)

            #     return attn_output

            # # Use checkpoint to recompute forward pass during the backward pass
            # attn_output = checkpoint(
            #     forward_pass, query_states, key_states, value_states, attention_mask
            # )

            bsz, q_len = query_states.size(0), query_states.size(2)
            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """






            """ Softmax norm scale
            # Get query and key norms
            query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).exp()
            # attn_weights_denom = (query_norm * key_norm.transpose(2, 3)).exp()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()

            # Divisor
            value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid() * torch.e)

            attn_weights = attn_weights * causal_mask

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            """





            """ Gaussian Attention
            # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            # attn_weights = 1-torch.exp(-(self.norm_const2*attn_weights)**2)

            # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            # attn_weights = 1.1**attn_weights

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            # Divisor
            value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """


            """ Gaussian Attention V2
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))

            attn_weights = 1.1**attn_weights

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)
            attn_weights = attn_weights.to(query_states.dtype)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """









            """1+elu(x) with denominator
            power = 1
            dtype = torch.float64
            cosine_attn = False
            denom_add = 0.1
            if cosine_attn:
                denom = False
                norm = True
                def act(x):
                    # return torch.nn.functional.elu(x) + 1
                    return torch.nn.functional.normalize(x, dim=-1)
            else:
                denom = True
                norm = False
                def act(x):
                    # return torch.nn.functional.elu(x) + 1
                    return x.relu()

            # Send the query and keys through the activation
            query_states = act(query_states)
            key_states = act(key_states)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).to(dtype)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / (attn_weights.shape[-1]**0.5)

            # Divisor
            if norm:
                value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())
            # Power
            attn_weights = attn_weights**power
            # Norm
            if denom:
                attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states).to(query_states.dtype)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """








            """1+elu(x) with denominator as a power series (no checkpoint)
            power = 80
            power_series = True
            dtype = torch.float64
            cosine_attn = False
            denom_add = 0.00001
            if cosine_attn:
                denom = False
                norm = True
                def act(x):
                    # return torch.nn.functional.elu(x) + 1
                    return torch.nn.functional.normalize(x, dim=-1, p=2)
            else:
                denom = True
                norm = False
                def act(x):
                    # return torch.nn.functional.elu(x) + 1
                    return x.relu()

            # Send the query and keys through the activation
            query_states = act(query_states)
            key_states = act(key_states)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).to(dtype)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / (attn_weights.shape[-1]**0.5)

            # Power series
            if power_series:
                attn_weights_ = 1 + attn_weights.clone()
                cur = torch.tensor([1], dtype=dtype).to(attn_weights.device)
                for i in range(2, power+1):
                    cur = cur * i
                    attn_weights_ = attn_weights_ + (1/cur) * attn_weights**i
            else:
                attn_weights_ = attn_weights ** power

            attn_weights_ = attn_weights_ + denom_add
            attn_weights_ = attn_weights_ * causal_mask

            # Norm
            if norm:
                value_states = value_states / ((causal_mask).sum(-1, keepdims=True)**self.norm_const.sigmoid())
            if denom:
                attn_weights = attn_weights_ / attn_weights_.sum(-1, keepdim=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """


            """ Cosine and sin
            # Inner product
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).float()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / (attn_weights.shape[-1]**0.5)

            # Cosine and sin
            attn_weights = attn_weights.sin() + attn_weights.cos()
            attn_weights = attn_weights + 1.42
            attn_weights = attn_weights * causal_mask

            # Denom
            attn_weights = attn_weights / attn_weights.sum(-1, keepdims=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """



            """ No activation
            # Inner product
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).float()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / (attn_weights.shape[-1]**0.5)

            # Add min
            attn_weights = attn_weights - attn_weights.min(-1, keepdims=True).values + 0.00001
            attn_weights = attn_weights * causal_mask

            # Denom
            attn_weights = attn_weights / attn_weights.sum(-1, keepdims=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            """
        
        
        elif attn_mode == "softmax":
            #""" Normal softmax
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            #"""




            """ Normal softmax with a learnable constant
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Exponential
            attn_weights = self.base ** attn_weights

            # Divisor
            value_states = value_states / (self.base**((causal_mask==0).sum(-1, keepdims=True)*self.norm_const))

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            """



            """ Normal softmax, kinda
            n = 10
            attn_weights = torch.matmul(query_states[:, :, :, :-n], key_states[:, :, :, :-n].transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights_mul = torch.matmul(query_states[:, :, :, -n:], key_states[:, :, :, -n:].transpose(2, 3)).sigmoid()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
                attn_weights_mul = attn_weights_mul * (causal_mask==0)

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_weights = attn_weights * attn_weights_mul
            attn_output = torch.matmul(attn_weights, value_states)
            """


            """ Normal softmax, kinda - inner
            n = 10
            normalize = torch.nn.functional.normalize
            attn_weights = torch.matmul(query_states[:, :, :, :-n], key_states[:, :, :, :-n].transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_weights_mul = torch.matmul(normalize(query_states[:, :, :, -n:], dim=-1), normalize(key_states[:, :, :, -n:], dim=-1).transpose(2, 3))

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
                attn_weights_mul = attn_weights_mul * (causal_mask==0)

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_weights = attn_weights * attn_weights_mul
            attn_output = torch.matmul(attn_weights, value_states)
            """



            """ Magnitude not in the exponential
            # Get query and key norms
            query_norm = torch.norm(query_states, dim=-1, p=2, keepdim=True)
            key_norm = torch.norm(key_states, dim=-1, p=2, keepdim=True)

            # Normalize query, and keys
            query_states = query_states / query_norm
            key_states = key_states / key_norm

            # ||a|| * ||b|| * e^cos(theta)
            attn_weights = ((torch.matmul(query_states, key_states.transpose(2, 3))-1) * (query_norm * key_norm.transpose(2, 3))).exp()

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask == 0).detach()
                attn_weights = attn_weights * causal_mask

            # Weight normalization
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)

            # upcast attention to fp32
            # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)
            """



            """ Normal softmax - decomposed
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = ~attention_mask[:, :, :, : key_states.shape[-2]].bool()
                # attn_weights = attn_weights + causal_mask

            def attn(query_states, key_states, value_states, causal_mask):
                attn_weights = (query_states.relu().unsqueeze(-2) * key_states.relu().unsqueeze(-3)).prod(-1) + 0.00001
                attn_weights = attn_weights * causal_mask

                # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                # upcast attention to fp32
                attn_weights = attn_weights / (attn_weights.sum(-1, keepdims=True))
                attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                return torch.matmul(attn_weights, value_states)
            
            attn_output = torch.utils.checkpoint.checkpoint(
                attn, query_states, key_states, value_states, causal_mask,
                use_reentrant=True
            )
            """









            """ Iterative softmax of polynomials via expansion
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = ~attention_mask[:, :, :, : key_states.shape[-2]].bool()
                # attn_weights = attn_weights + causal_mask
            else:
                L, S = query_states.size(-2), key_states.size(-2)
                attn_bias = torch.zeros(L, S, dtype=query_states.dtype, device=query_states.device)
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).cuda()
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias = attn_bias.to(query_states.dtype)
                causal_mask = attn_bias == 0

            def attn(query_states, key_states, value_states, causal_mask):
                attn_weights = query_states @ key_states.mT
                attn_weights = attn_weights * causal_mask

                attn_weights = attn_weights.to(torch.float64)
                #attn_weights = attn_weights - attn_weights.max(-1, keepdims=True).values
                attn_weights_ = 1 + attn_weights.clone()
                cur = torch.tensor([1], dtype=torch.float64).to(attn_weights.device)
                iters = 80
                for i in range(2, iters):
                    cur = cur*i
                    attn_weights_ = attn_weights_ + (attn_weights**i)/cur
                attn_weights_ = (attn_weights_ * causal_mask)
                attn_weights = (attn_weights_ / attn_weights_.sum(-1, keepdims=True)).to(query_states.dtype)

                # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

                # upcast attention to fp32
                # attn_weights = attn_weights / (attn_weights.sum(-1, keepdims=True))
                # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                return torch.matmul(attn_weights, value_states)
            
            attn_output = torch.utils.checkpoint.checkpoint(
                attn, query_states, key_states, value_states, causal_mask,
                use_reentrant=True
            )
            """




            """Softmax stop grad norm
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).detach()
            attn_output = torch.matmul(attn_weights, value_states)
            """



            """ learnable base
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / self.head_dim

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = self.base**attn_weights
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)
            attn_output = torch.matmul(attn_weights, value_states)
            """




            """ (1/(1-x)) instead of exp
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / self.head_dim

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = (1/(1-attn_weights.clamp(max=0.9))) * (causal_mask==0)
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)
            attn_output = torch.matmul(attn_weights, value_states)
            """




            """Softmax - divide by varaince
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            # attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights.exp()
            mean = (attn_weights.sum(-1, keepdim=True) / (causal_mask==0).sum(-1, keepdim=True))
            mean_sq = ((attn_weights**2).sum(-1, keepdim=True) / (causal_mask==0).sum(-1, keepdim=True))
            var = mean_sq - mean**2
            var[:, :, 0] = var[:, :, 0] + 1
            attn_weights = (attn_weights-mean) / torch.where(var==0, var+1, var)
            attn_output = torch.matmul(attn_weights, value_states)
            """






            """Softmax - L2 distance instead of inner product
            def dist(query, key):
                return -(((query[:, :, :, None, :] - key[:, :, None, :, :])**2).sum(-1)**0.5)
            attn_weights = torch.utils.checkpoint.checkpoint(
                dist, query_states, key_states,
                use_reentrant=True
            )

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            # attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights.softmax(-1)
            attn_output = torch.matmul(attn_weights, value_states)
            """







            """Softmax - LP distance instead of inner product
            def dist(query, key, power):
                return -(((query[:, :, :, None, :] - key[:, :, None, :, :])**power).sum(-1))**(1/power.squeeze(-1))
            attn_weights = torch.utils.checkpoint.checkpoint(
                dist, query_states, key_states, self.lp_const,
                use_reentrant=True
            )

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            # attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights.softmax(-1)
            attn_output = torch.matmul(attn_weights, value_states)
            """






            """Softmax - Covariance instead of inner product
            def dist(query, key):
                query_key_var = (query[:, :, :, None, :] * key[:, :, None, :, :]).mean(-1)
                Q_var = query[:, :, :, None, :].mean(-1)
                K_var = key[:, :, None, :, :].mean(-1)
                return query_key_var - Q_var*K_var
            attn_weights = torch.utils.checkpoint.checkpoint(
                dist, query_states, key_states,
                use_reentrant=True
            )

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            # attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights.softmax(-1)
            attn_output = torch.matmul(attn_weights, value_states)
            """










            """Softmax - Learned
            def dist(query, key, linear):
                return linear(query[:, :, :, None, :] * key[:, :, None, :, :]).squeeze(-1)
            attn_weights = torch.utils.checkpoint.checkpoint(
                dist, query_states, key_states, self.learned_weight,
                use_reentrant=True
            )

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            # attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights.softmax(-1)
            attn_output = torch.matmul(attn_weights, value_states)
            """






            """Coshmax
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            attn_weights = attn_weights.float()

            # upcast attention to fp32
            attn_weights = torch.cosh(attn_weights)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights * (causal_mask==0)

            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True)
            attn_output = torch.matmul(attn_weights.to(value_states.dtype), value_states)
            """

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            















        elif attn_mode == "softmaxV2":
            """
            # Weight projection
            decay_weights = self.weight_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).sigmoid()

            def attn(query_states, key_states, value_states, decay_weights, norm):
                dtype = query_states.dtype
                query_states = query_states.float()
                key_states = key_states.float()
                value_states = value_states.float()
                decay_weights = decay_weights.float()
                norm = norm.float()

                # Output tensor
                attn_output = []

                # Iterate down the query outputs
                for t in range(0, q_len):
                    decay_weight = decay_weights[:, :, :t+1]
                    key_states_ = key_states[:, :, :t+1] * decay_weight#.cumsum(-2).flip(-2)
                    w = (query_states[:, :, t:t+1] @ key_states_.mT / math.sqrt(self.head_dim)).exp()
                    attn_output.append(w @ value_states[:, :, :t+1])

                return norm(torch.stack(attn_output, dim=-2)).to(dtype)
            attn_output = torch.utils.checkpoint.checkpoint(
                attn, query_states, key_states, value_states, decay_weights, self.out_norm,
                use_reentrant=True
            )
            """







            """ Weight gate attn, upper bound denominator, out norm
            # Weight projection
            # decay_weights = self.weight_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).sigmoid()
            decay_weights = self.weight_proj(hidden_states).sigmoid()

            dtype = query_states.dtype
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
            # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
            attn_weights = attn_weights.clamp(max=5).exp() * decay_weights.mT[:, :, None, :]

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask==0)
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / (torch.e**5 * causal_mask.sum(-1, keepdim=True))
            # attn_weights = attn_weights / (torch.e**5 * (causal_mask.sum(-1, keepdims=True))**self.norm_const.sigmoid())

            attn_output = self.out_norm(torch.matmul(attn_weights, value_states.float())).to(dtype)
            """





            #""" Double weight gate attn, upper bound denominator, out norm
            # Weight projection
            # decay_weights = self.weight_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2).sigmoid()
            # out_gate = self.weight_proj(hidden_states).mT[:, :, :, None].sigmoid()
            # in_gate = self.weight_proj2(hidden_states).mT[:, :, :, None].sigmoid()

            # dtype = query_states.dtype
            # attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
            # # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
            # attn_weights = attn_weights.clamp(max=5).exp()  * decay_weights_k.mT[:, :, None, :] * decay_weights_q.mT[:, :, :, None]

            # if attention_mask is not None:  # no matter the length, we just slice it
            #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            #     causal_mask = (causal_mask==0)
            #     attn_weights = attn_weights * causal_mask

            # attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)
            # # attn_weights = attn_weights / (torch.e**5 * (causal_mask.sum(-1, keepdims=True))**self.norm_const.sigmoid())

            # attn_output = self.out_norm(torch.matmul(attn_weights, value_states.float())).to(dtype)

            # if attention_mask is not None:  # no matter the length, we just slice it
            #     causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            #     causal_mask = (causal_mask==0)

            # attn_output = self.out_norm(GatedAttention.apply(query_states, key_states, value_states, out_gate, in_gate, causal_mask))
            
            decay_weights_q = self.weight_proj(hidden_states).sigmoid()
            decay_weights_k = self.weight_proj2(hidden_states).sigmoid()

            dtype = query_states.dtype
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
            # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
            attn_weights = attn_weights.clamp(max=5).exp()  * decay_weights_k.mT[:, :, None, :] * decay_weights_q.mT[:, :, :, None]

            if attention_mask is not None:  # no matter the length, we just slice it
                # causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = attention_mask
                causal_mask = (causal_mask==0)
                attn_weights = attn_weights * causal_mask

            attn_weights = attn_weights / causal_mask.sum(-1, keepdim=True)
            # attn_weights = attn_weights / (torch.e**5 * (causal_mask.sum(-1, keepdims=True))**self.norm_const.sigmoid())

            attn_output = self.out_norm(torch.matmul(attn_weights, value_states.float())).to(dtype)
            #"""







            """ Weight attn, upper bound denom, learnable power
            # Weight projection
            decay_weights = self.weight_proj(hidden_states).sigmoid()

            dtype = query_states.dtype
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)
            # attn_weights = attn_weights * decay_weights.mT[:, :, None, :]
            attn_weights = attn_weights.clamp(max=5).exp()
            # attn_weights = attn_weights.clamp(max=5).exp() * decay_weights.mT[:, :, None, :]

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = (causal_mask==0)
                attn_weights = attn_weights * causal_mask

            # attn_weights = attn_weights / (torch.e**5 * causal_mask.sum(-1, keepdim=True))
            attn_weights = attn_weights / (torch.e**5 * (causal_mask.sum(-1, keepdims=True))**self.norm_const.sigmoid())

            # attn_output = torch.matmul(attn_weights, decay_weights.mT[:, :, :, None] * value_states.float()).to(dtype)
            attn_output = torch.matmul(attn_weights, decay_weights.mT[:, :, :, None] * value_states.float()).to(dtype)
            """
















            """ Softmax with gate
            # Weight projection
            decay_weights = self.weight_proj(hidden_states).sigmoid()

            dtype = query_states.dtype
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            attn_weights = attn_weights.exp()
            attn_weights = (attn_weights * decay_weights.mT[:, :, None, :]) / attn_weights.sum(-1, keepdim=True)

            attn_output = torch.matmul(attn_weights, value_states.float()).to(dtype)
            # attn_output = torch.matmul(attn_weights, decay_weights.mT[:, :, :, None] * value_states.float()).to(dtype)
            """




            """ Softmax with gate V2
            # Weight projection
            decay_weights = self.weight_proj(hidden_states).sigmoid()

            dtype = query_states.dtype
            attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # attn_weights = attn_weights.clamp(max=5).exp()
            # attn_weights = attn_weights / (attn_weights.sum(-2, keepdim=True).detach() + 1.0)
            attn_weights = attn_weights.clamp(max=5).exp()
            attn_weights = attn_weights / (attn_weights.sum(-1, keepdim=True).detach())

            # attn_output = torch.matmul(attn_weights, value_states.float()).to(dtype)
            attn_output = self.out_norm(torch.matmul(attn_weights, decay_weights.mT[:, :, :, None] * value_states.float())).to(dtype)
            """
                



































        elif attn_mode == "other":
            # Mask the convolutions
            self.key_conv.weight.data = self.key_conv.weight.data * self.key_mask
            self.value_conv.weight.data = self.value_conv.weight.data * self.value_mask
            # Convolutions
            key_states = self.key_conv(key_states.reshape(bsz*self.num_heads, -1, self.head_dim).mT).mT.reshape(bsz, self.num_heads, -1, self.head_dim) # look 1 into past
            value_states = self.value_conv(value_states.reshape(bsz*self.num_heads, -1, self.head_dim).mT).mT.reshape(bsz, self.num_heads, -1, self.head_dim) # look 1 into future



            """Normal softmax
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = (attn_weights - attn_weights.max(-1, keepdim=True).values).exp()
            attn_weights = attn_weights / attn_weights.sum(-1, keepdim=True).detach()
            attn_output = torch.matmul(attn_weights, value_states)
            """



            #"""Normal softmax - no diag
            # query_states = torch.nn.functional.normalize(query_states, dim=-1, p=2)
            # key_states = torch.nn.functional.normalize(key_states, dim=-1, p=2)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                causal_mask = causal_mask + torch.diag(-torch.inf*torch.ones(q_len))[None, None, :, :].to(causal_mask.device)
                causal_mask[:, :, 0, 0] = 0.0
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = attn_weights.softmax(-1)
            attn_output = torch.matmul(attn_weights, value_states)
            #"""








            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            




        else:
            assert False






            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
            





        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value














        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value





    
    
    
    
    
LLAMA_ATTENTION_CLASSES = {
    "soft": LlamaCosAttention,
    "cos": LlamaCosAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaCosAttention(config=config, layer_idx=layer_idx)

        custom = False
        if custom:
            self.mlp = LlamaMLPGeLU(config)
        else:
            self.mlp = LlamaMLP(config)
        
        
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
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

        if use_cache:
            outputs += (present_key_value,)

        return outputs
