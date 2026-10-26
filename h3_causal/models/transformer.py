"""MiniMax-H3 Omni-Transformer for chunk-causal training and KV-cached sampling.

A standalone port of diffusers' `MiniMaxH3Transformer3DModel` (0.40). Parameter
names match the diffusers checkpoint under `transformer/`, so its safetensors
load as-is. Three things differ from the release forward:

* attention takes a mask: a FlexAttention `BlockMask` (training), a dense bool
  mask (tests) or nothing (bidirectional, bit-compatible with the release);
* the token refiner runs per caption segment, so a chunk caption never sees
  another caption (the release refines one prompt, which is the 1-segment case);
* blocks can read cached K/V of earlier rows and return the K/V of their own
  rows, which is what the chunk-causal sampler is built on.

Keys are cached *before* RoPE together with their positions, so a long rollout
can re-rotate history (position recycling) without re-running the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

MODALITY_NUM = 3  # AdaLN table rows per timestep: video 0, text 1, audio 2
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2

# Small float32 modules of the mixed-precision release checkpoint.
FP32_MODULES = ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")

_compiled_flex_attention = None


def _flex_attention(q, k, v, block_mask):
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        # Packed lengths differ per clip: one dynamic-shape compile instead of a recompile per length.
        _compiled_flex_attention = torch.compile(flex_attention, dynamic=True)
    return _compiled_flex_attention(q, k, v, block_mask=block_mask)


@dataclass
class H3Config:
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    num_layers: int = 50
    num_refiner_layers: int = 2
    ffn_dim: int = 14336
    in_channels: int = 24
    audio_in_channels: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    freq_dim: int = 256
    time_embed_hidden_dim: int = 5376
    time_embed_dim: int = 2688
    rope_freq_dim: int = 16
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    @classmethod
    def from_dict(cls, config: dict) -> "H3Config":
        known = {name: config[name] for name in cls.__dataclass_fields__ if name in config}
        if "patch_size" in known:
            known["patch_size"] = tuple(known["patch_size"])
        return cls(**known)


@dataclass
class LayerKV:
    """Cached rows of one block: pre-RoPE keys, values and their (t, h, w) positions."""

    key: torch.Tensor  # [rows, heads, head_dim]
    value: torch.Tensor
    position_ids: torch.Tensor  # [rows, 3] float64


@dataclass
class KVContext:
    """Per-forward cache I/O.

    `prefix[i]` holds the rows block `i` reads ahead of the current rows (None =
    no history). `prefix_position_shift` is subtracted from the time axis of the
    cached rows that have `shiftable` set, i.e. position recycling. With
    `collect`, every block appends the pre-RoPE K/V of the current rows to
    `collected`.
    """

    prefix: list[LayerKV | None] | None = None
    prefix_shiftable: torch.Tensor | None = None  # [prefix_rows] bool
    prefix_position_shift: float = 0.0
    collect: bool = False
    collect_position_ids: torch.Tensor | None = None  # canonical positions to store (unshifted)
    collected: list[LayerKV] = field(default_factory=list)


def get_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """diffusers `Timesteps(dim, flip_sin_to_cos=True, downscale_freq_shift=0)`, timesteps in [0, 1] unscaled."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, hidden_dim, bias=True)
        self.linear_2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim * 2, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.silu(gate)


class FeedForward(nn.Module):
    """diffusers `FeedForward(activation_fn="swiglu", bias=False)`: `net.0.proj`, `net.2`."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.net = nn.ModuleList([SwiGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim, bias=False)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class RotaryPosEmbed(nn.Module):
    """3-axis RoPE over (t, h, w); one `inv_freq` shared by the axes, 96 of 128 head channels rotated."""

    def __init__(self, rope_freq_dim: int, rope_theta: float):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.float().view(1, 1, -1)
        freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`hidden_states` is [B, S, heads, head_dim]; cos/sin are [S, rotary_dim]."""
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotary = rotary * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((rotary, passthrough), dim=-1)


class Attention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dim_head: int, qk_norm_eps: float):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, hidden_size, bias=False), nn.Dropout(0.0)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | BlockMask | None = None,
        rope: "RotaryPosEmbed | None" = None,
        kv_prefix: LayerKV | None = None,
        prefix_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        kv_collect: list[LayerKV] | None = None,
    ) -> torch.Tensor:
        batch = hidden_states.shape[0]
        query = self.norm_q(self.to_q(hidden_states).unflatten(-1, (self.heads, -1)))
        key = self.norm_k(self.to_k(hidden_states).unflatten(-1, (self.heads, -1)))
        value = self.to_v(hidden_states).unflatten(-1, (self.heads, -1))

        if kv_collect is not None:
            if batch != 1:
                raise ValueError("KV collection runs on one sample")
            kv_collect.append(LayerKV(key[0].detach(), value[0].detach(), position_ids))

        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        if kv_prefix is not None:
            prefix_key = apply_rotary_emb(kv_prefix.key[None].to(key.dtype), *prefix_rotary_emb)
            key = torch.cat([prefix_key.expand(batch, -1, -1, -1), key], dim=1)
            value = torch.cat([kv_prefix.value[None].to(value.dtype).expand(batch, -1, -1, -1), value], dim=1)

        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        if isinstance(attention_mask, BlockMask):
            out = _flex_attention(query, key, value, attention_mask)
        else:
            out = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
        out = out.transpose(1, 2).flatten(2, 3).type_as(hidden_states)
        return self.to_out[1](self.to_out[0](out))


class TokenRefinerBlock(nn.Module):
    def __init__(self, config: H3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config.hidden_size, config.num_attention_heads, config.attention_head_dim, config.qk_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ff = FeedForward(config.hidden_size, config.ffn_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        return hidden_states + self.ff(self.norm2(hidden_states))


class TokenRefiner(nn.Module):
    """Two plain pre-norm blocks over the projected text, no AdaLN and no RoPE."""

    def __init__(self, config: H3Config):
        super().__init__()
        self.refiner_blocks = nn.ModuleList([TokenRefinerBlock(config) for _ in range(config.num_refiner_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        # Text rows of different segments (global caption, chunk captions) are
        # refined independently: block-diagonal attention over segment ids.
        if hidden_states.shape[1] == 0:
            return self.final_norm(hidden_states)
        mask = None
        if segment_ids is not None and int(segment_ids.max()) != int(segment_ids.min()):
            mask = segment_ids[:, None] == segment_ids[None, :]
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(block, hidden_states, mask, use_reentrant=False)
            else:
                hidden_states = block(hidden_states, mask)
        return self.final_norm(hidden_states)


class AdaLayerNormModulation(nn.Module):
    """`(timesteps, time_embed_dim)` -> six `(timesteps * 3, hidden)` tables, one row per (timestep, modality)."""

    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MODALITY_NUM, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(F.silu(temb).to(self.linear.weight.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class AdaLayerNormOut(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor, timestep_indices: torch.Tensor) -> torch.Tensor:
        shift, scale = self.linear(F.silu(temb).to(self.linear.weight.dtype)).chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(0, timestep_indices)


class TransformerBlock(nn.Module):
    def __init__(self, config: H3Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(config.hidden_size, config.num_attention_heads, config.attention_head_dim, config.qk_norm_eps)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ff = FeedForward(config.hidden_size, config.ffn_dim)
        self.adaln_proj = AdaLayerNormModulation(config.time_embed_dim, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | BlockMask | None = None,
        kv_prefix: LayerKV | None = None,
        prefix_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_ids: torch.Tensor | None = None,
        kv_collect: list[LayerKV] | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_msa.index_select(0, adaln_indices)) + shift_msa.index_select(
            0, adaln_indices
        )
        attn_output = self.attn(
            norm_hidden_states,
            rotary_emb,
            attention_mask,
            kv_prefix=kv_prefix,
            prefix_rotary_emb=prefix_rotary_emb,
            position_ids=position_ids,
            kv_collect=kv_collect,
        )
        hidden_states = hidden_states + gate_msa.index_select(0, adaln_indices) * attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp.index_select(0, adaln_indices)) + shift_mlp.index_select(
            0, adaln_indices
        )
        return hidden_states + gate_mlp.index_select(0, adaln_indices) * self.ff(norm_hidden_states)


class H3Transformer(nn.Module):
    """One packed sequence of text, video and audio rows; see `h3_causal.dataset.causal_packing` for the layout."""

    def __init__(self, config: H3Config):
        super().__init__()
        self.config = config
        video_patch_dim = config.in_channels * math.prod(config.patch_size)
        self.proj_in = nn.Linear(video_patch_dim, config.hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(config.audio_in_channels, config.hidden_size, bias=True)
        self.context_embedder = nn.Linear(config.text_dim, config.hidden_size, bias=True)
        self.time_embedder = TimestepEmbedding(config.freq_dim, config.time_embed_hidden_dim, config.time_embed_dim)
        self.rope = RotaryPosEmbed(config.rope_freq_dim, config.rope_theta)
        self.token_refiner = TokenRefiner(config)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm_out = AdaLayerNormOut(config.hidden_size, config.time_embed_dim, config.final_norm_eps)
        self.proj_out = nn.Linear(config.hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(config.hidden_size, config.audio_in_channels, bias=True)
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True
        self.token_refiner.gradient_checkpointing = True

    def embed_timesteps(self, timestep: torch.Tensor) -> torch.Tensor:
        temb = get_timestep_embedding(timestep, self.config.freq_dim)
        return self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        text_segment_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | BlockMask | None = None,
        kv_context: KVContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same contract as diffusers' forward plus the causal extras.

        Returns the video and audio velocity (`x0 - noise`, data-ward) of the
        rows in `video_indices` / `audio_indices` order.
        """
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError("token_tags and timestep_indices must be (seq_len,) like position_ids")
        rotary_emb = self.rope(position_ids)

        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds, text_segment_ids)

        packed = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        temb = self.embed_timesteps(timestep)
        adaln_indices = timestep_indices * MODALITY_NUM + token_tags

        prefix_rotary = [None] * len(self.transformer_blocks)
        prefix = None
        if kv_context is not None and kv_context.prefix is not None:
            prefix = kv_context.prefix
            # Every block caches the same rows, so one rotation table serves all.
            first = next((layer for layer in prefix if layer is not None), None)
            if first is not None:
                prefix_positions = first.position_ids
                if kv_context.prefix_position_shift and kv_context.prefix_shiftable is not None:
                    prefix_positions = prefix_positions.clone()
                    prefix_positions[kv_context.prefix_shiftable, 0] -= kv_context.prefix_position_shift
                prefix_rotary = [self.rope(prefix_positions)] * len(self.transformer_blocks)
        collect = kv_context is not None and kv_context.collect
        collect_positions = position_ids
        if collect:
            if torch.is_grad_enabled():
                raise RuntimeError("KV collection is an inference-only path")
            kv_context.collected = []
            if kv_context.collect_position_ids is not None:
                collect_positions = kv_context.collect_position_ids

        for index, block in enumerate(self.transformer_blocks):
            block_kwargs = dict(
                attention_mask=attention_mask,
                kv_prefix=None if prefix is None else prefix[index],
                prefix_rotary_emb=prefix_rotary[index],
                position_ids=collect_positions,
                kv_collect=kv_context.collected if collect else None,
            )
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                packed = torch.utils.checkpoint.checkpoint(
                    block, packed, temb, adaln_indices, rotary_emb, use_reentrant=False, **block_kwargs
                )
            else:
                packed = block(packed, temb, adaln_indices, rotary_emb, **block_kwargs)

        packed = self.norm_out(packed, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(packed.index_select(1, video_indices))
        audio_output = self.audio_proj_out(packed.index_select(1, audio_indices))
        return video_output, audio_output
