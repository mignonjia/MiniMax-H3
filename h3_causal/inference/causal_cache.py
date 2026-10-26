"""KV cache and incremental forwards for chunk-causal MiniMax-H3 sampling.

Mirrors the training mask exactly: group 0 (global text + anchor) is prefilled
once; chunk k's forward reads group 0, the committed audio/video of the
previous `window_chunks - 1` chunks and the committed captions inside the
caption window, plus its own rows. A chunk is committed by one extra forward
of its final latents at `commit_timestep`, collecting every block's K/V.

Keys are stored pre-RoPE with their positions. `position_shift` re-rotates the
history and the current chunk onto earlier rotary times (position recycling)
once a rollout runs past the trained length; group 0 never moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from h3_causal.dataset.causal_packing import ROLE_ANCHOR, ROLE_GLOBAL_TEXT, CausalPlan
from h3_causal.models.transformer import H3Transformer, KVContext, LayerKV


@dataclass
class RowSelection:
    """Model inputs for a contiguous run of rows of a plan (one attention group)."""

    rows: torch.Tensor
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    text_indices: torch.Tensor
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_segment_ids: torch.Tensor
    text_slice: slice  # into plan.text_indices order
    video_slice: slice  # into plan.video_indices order
    audio_slice: slice
    shiftable: torch.Tensor  # rows that move under position recycling


def _ordinal_slice(order: torch.Tensor, start: int, end: int) -> tuple[slice, torch.Tensor]:
    inside = ((order >= start) & (order < end)).nonzero().flatten()
    if inside.numel() == 0:
        return slice(0, 0), order.new_zeros(0)
    first, last = int(inside[0]), int(inside[-1]) + 1
    if last - first != inside.numel():
        raise ValueError("rows of one group must be contiguous in each modality's order")
    return slice(first, last), order[first:last] - start


def select_group(plan: CausalPlan, group: int) -> RowSelection:
    members = (plan.row_group == group).nonzero().flatten()
    start, end = int(members[0]), int(members[-1]) + 1
    if end - start != members.numel():
        raise ValueError(f"group {group} is not contiguous")
    text_slice, text_local = _ordinal_slice(plan.text_indices, start, end)
    video_slice, video_local = _ordinal_slice(plan.video_indices, start, end)
    audio_slice, audio_local = _ordinal_slice(plan.audio_indices, start, end)
    role = plan.row_role[start:end]
    return RowSelection(
        rows=members,
        position_ids=plan.position_ids[start:end],
        token_tags=plan.token_tags[start:end],
        text_indices=text_local,
        video_indices=video_local,
        audio_indices=audio_local,
        text_segment_ids=plan.text_segment_ids[text_slice],
        text_slice=text_slice,
        video_slice=video_slice,
        audio_slice=audio_slice,
        shiftable=(role != ROLE_GLOBAL_TEXT) & (role != ROLE_ANCHOR),
    )


@dataclass
class ChunkEntry:
    media: list[LayerKV]  # audio + video rows of the chunk, every block
    caption: list[LayerKV] | None


def _cat_layers(parts: list[list[LayerKV]]) -> list[LayerKV]:
    return [
        LayerKV(
            torch.cat([p[i].key for p in parts]),
            torch.cat([p[i].value for p in parts]),
            torch.cat([p[i].position_ids for p in parts]),
        )
        for i in range(len(parts[0]))
    ]


def _split_layers(layers: list[LayerKV], mask: torch.Tensor) -> list[LayerKV]:
    return [LayerKV(layer.key[mask], layer.value[mask], layer.position_ids[mask]) for layer in layers]


@dataclass
class CausalKVCache:
    window_chunks: int
    caption_window_chunks: int
    group0: list[LayerKV] | None = None
    chunks: dict[int, ChunkEntry] = field(default_factory=dict)

    def visible_prefix(self, chunk: int) -> tuple[list[LayerKV], torch.Tensor]:
        """Cached rows chunk `chunk` reads, and which of them are shiftable."""
        if self.group0 is None:
            raise RuntimeError("prefill group 0 first")
        parts = [self.group0]
        shiftable = [torch.zeros(self.group0[0].key.shape[0], dtype=torch.bool)]
        for past in range(max(0, chunk - self.window_chunks + 1), chunk):
            entry = self.chunks[past]
            if entry.caption is not None and past > chunk - self.caption_window_chunks:
                parts.append(entry.caption)
                shiftable.append(torch.ones(entry.caption[0].key.shape[0], dtype=torch.bool))
            parts.append(entry.media)
            shiftable.append(torch.ones(entry.media[0].key.shape[0], dtype=torch.bool))
        return _cat_layers(parts), torch.cat(shiftable)

    def commit(self, chunk: int, collected: list[LayerKV], caption_rows: int) -> None:
        rows = collected[0].key.shape[0]
        is_caption = torch.zeros(rows, dtype=torch.bool)
        is_caption[:caption_rows] = True
        self.chunks[chunk] = ChunkEntry(
            media=_split_layers(collected, ~is_caption),
            caption=_split_layers(collected, is_caption) if caption_rows else None,
        )
        # Ring: nothing older than the window is ever read again.
        for old in [c for c in self.chunks if c < chunk - self.window_chunks + 2]:
            del self.chunks[old]


def group_forward(
    model: H3Transformer,
    selection: RowSelection,
    *,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    text_rows: torch.Tensor,
    row_t: torch.Tensor,
    kv_context: KVContext | None,
    position_shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One forward over a group's rows; `row_t` is the per-row H3 timestep of these rows."""
    device = video_rows.device if video_rows.numel() else text_rows.device
    position_ids = selection.position_ids.to(device)
    if position_shift:
        position_ids = position_ids.clone()
        position_ids[selection.shiftable.to(device), 0] -= position_shift
    timestep, timestep_indices = torch.unique(row_t.float(), sorted=True, return_inverse=True)
    if kv_context is not None:
        kv_context.collect_position_ids = selection.position_ids.to(device)
        kv_context.prefix_position_shift = position_shift
    return model(
        hidden_states=video_rows[None],
        audio_hidden_states=audio_rows[None],
        encoder_hidden_states=text_rows[None],
        timestep=timestep.to(device),
        timestep_indices=timestep_indices.to(device),
        token_tags=selection.token_tags.to(device),
        position_ids=position_ids,
        video_indices=selection.video_indices.to(device),
        audio_indices=selection.audio_indices.to(device),
        text_indices=selection.text_indices.to(device),
        text_segment_ids=selection.text_segment_ids.to(device),
        kv_context=kv_context,
    )
