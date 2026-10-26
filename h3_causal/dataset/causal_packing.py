"""Packed chunk-causal layout of one MiniMax-H3 sample.

Rows are laid out in attention-group order (the transformer scatters rows by
index, so no separate FFN order is needed):

    group 0      [ global text | anchor (first-frame keyframe) ]
    group k + 1  [ caption_k | audio_k (ch0, ch1) | video_k (5 latent frames) ]

Visibility (`window_chunks` = W, `caption_window_chunks` = Wc):
    group 0 sees group 0 only;
    chunk group g sees group 0, the audio/video of groups g-W+1..g, and the
    captions of groups g-Wc+1..g (Wc = 1: own caption only).

RoPE follows the release: global text on t = 0..Lg-1, media from origin Lg on
the shared 40-units/s clock. A chunk caption sits on the time axis right before
its chunk's first latent frame, mirroring the release's "text, then media".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from h3_causal.dataset import geometry as geo
from h3_causal.models.transformer import AUDIO_TAG, TEXT_TAG, VIDEO_TAG

ROLE_GLOBAL_TEXT, ROLE_ANCHOR, ROLE_CAPTION, ROLE_AUDIO, ROLE_VIDEO = range(5)


@dataclass
class CausalPlan:
    num_chunks: int
    latent_height: int
    latent_width: int
    rows_per_frame: int
    sequence_length: int
    position_ids: torch.Tensor  # [S, 3] float64
    token_tags: torch.Tensor  # [S]
    row_group: torch.Tensor  # [S] 0 = global group, k + 1 = chunk k
    row_role: torch.Tensor  # [S] ROLE_*
    text_indices: torch.Tensor  # global text rows, then each chunk caption
    text_segment_ids: torch.Tensor  # [num_text_rows] 0 = global, k + 1 = caption k
    video_indices: torch.Tensor  # anchor rows, then chunk rows in chunk order
    audio_indices: torch.Tensor  # per chunk: ch0 rows then ch1 rows
    num_anchor_rows: int
    video_row_chunk: torch.Tensor  # [num_video_rows] -1 = anchor
    audio_row_chunk: torch.Tensor  # [num_audio_rows]
    audio_row_latent: torch.Tensor  # [num_audio_rows] (channel, latent) flattened as channel * Ta + latent
    group_rows: tuple[int, ...]  # rows per group, in sequence order

    def to(self, device) -> "CausalPlan":
        moved = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return CausalPlan(**moved)


def build_causal_plan(
    *,
    global_text_tags: torch.Tensor,
    caption_lengths: list[int],
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
) -> CausalPlan:
    """Lay out one sample. `caption_lengths[k]` is chunk k's caption rows (0 = no caption)."""
    num_chunks = len(caption_lengths)
    if num_chunks < 1:
        raise ValueError("a causal sample needs at least one chunk")
    _, patch_h, patch_w = geo.PATCH_SIZE
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    chunk_video_rows = geo.LATENTS_PER_CHUNK * rows_per_frame
    if geo.chunk_audio_range(num_chunks - 1)[1] > num_audio_latents:
        raise ValueError(f"{num_audio_latents} audio latents do not cover {num_chunks} chunks")

    num_global = int(global_text_tags.numel())
    origin = float(num_global)
    frame_grid, width_grid = geo.frame_position_grid(latent_height, latent_width)
    frame_times = geo.latent_frame_rotary_times(geo.LATENTS_PER_CHUNK * num_chunks, origin)

    positions, tags, groups, roles = [], [], [], []
    text_rows, text_segments, video_rows, audio_rows = [], [], [], []
    video_row_chunk, audio_row_chunk, audio_row_latent = [], [], []
    group_rows = []
    cursor = 0

    def take(count: int) -> torch.Tensor:
        nonlocal cursor
        rows = torch.arange(cursor, cursor + count)
        cursor += count
        return rows

    # Group 0: global text, then the anchor frame.
    rows = take(num_global)
    pos = torch.zeros(num_global, 3, dtype=torch.float64)
    pos[:, 0] = torch.arange(num_global, dtype=torch.float64)
    positions.append(pos)
    tags.append(global_text_tags.to(torch.long))
    roles.append(torch.full((num_global,), ROLE_GLOBAL_TEXT))
    text_rows.append(rows)
    text_segments.append(torch.zeros(num_global, dtype=torch.long))

    rows = take(rows_per_frame)
    pos = torch.empty(rows_per_frame, 3, dtype=torch.float64)
    pos[:, 0] = origin
    pos[:, 1:] = frame_grid
    positions.append(pos)
    tags.append(torch.full((rows_per_frame,), VIDEO_TAG))
    roles.append(torch.full((rows_per_frame,), ROLE_ANCHOR))
    video_rows.append(rows)
    video_row_chunk.append(torch.full((rows_per_frame,), -1))
    groups.append(torch.zeros(num_global + rows_per_frame, dtype=torch.long))
    group_rows.append(num_global + rows_per_frame)

    for chunk, caption_length in enumerate(caption_lengths):
        start_time = float(frame_times[chunk * geo.LATENTS_PER_CHUNK])
        group_size = 0

        if caption_length:
            rows = take(caption_length)
            pos = torch.zeros(caption_length, 3, dtype=torch.float64)
            pos[:, 0] = start_time - caption_length + torch.arange(caption_length, dtype=torch.float64)
            positions.append(pos)
            tags.append(torch.full((caption_length,), TEXT_TAG))
            roles.append(torch.full((caption_length,), ROLE_CAPTION))
            text_rows.append(rows)
            text_segments.append(torch.full((caption_length,), chunk + 1))
            group_size += caption_length

        audio_start, audio_end = geo.chunk_audio_range(chunk)
        per_channel = audio_end - audio_start
        rows = take(per_channel * geo.AUDIO_CHANNELS)
        pos = torch.zeros(per_channel * geo.AUDIO_CHANNELS, 3, dtype=torch.float64)
        latent = torch.arange(audio_start, audio_end)
        pos[:, 0] = (origin + latent.to(torch.float64)).repeat(geo.AUDIO_CHANNELS)
        pos[:per_channel, 2] = float(width_grid[0])
        pos[per_channel:, 2] = float(width_grid[-1])
        positions.append(pos)
        tags.append(torch.full((rows.numel(),), AUDIO_TAG))
        roles.append(torch.full((rows.numel(),), ROLE_AUDIO))
        audio_rows.append(rows)
        audio_row_chunk.append(torch.full((rows.numel(),), chunk))
        audio_row_latent.append(torch.cat([c * num_audio_latents + latent for c in range(geo.AUDIO_CHANNELS)]))
        group_size += rows.numel()

        rows = take(chunk_video_rows)
        pos = torch.empty(geo.LATENTS_PER_CHUNK, rows_per_frame, 3, dtype=torch.float64)
        chunk_frames = slice(chunk * geo.LATENTS_PER_CHUNK, (chunk + 1) * geo.LATENTS_PER_CHUNK)
        pos[:, :, 0] = frame_times[chunk_frames][:, None]
        pos[:, :, 1:] = frame_grid[None]
        positions.append(pos.reshape(-1, 3))
        tags.append(torch.full((chunk_video_rows,), VIDEO_TAG))
        roles.append(torch.full((chunk_video_rows,), ROLE_VIDEO))
        video_rows.append(rows)
        video_row_chunk.append(torch.full((chunk_video_rows,), chunk))
        group_size += chunk_video_rows

        groups.append(torch.full((group_size,), chunk + 1))
        group_rows.append(group_size)

    return CausalPlan(
        num_chunks=num_chunks,
        latent_height=latent_height,
        latent_width=latent_width,
        rows_per_frame=rows_per_frame,
        sequence_length=cursor,
        position_ids=torch.cat(positions),
        token_tags=torch.cat(tags),
        row_group=torch.cat(groups),
        row_role=torch.cat(roles),
        text_indices=torch.cat(text_rows),
        text_segment_ids=torch.cat(text_segments),
        video_indices=torch.cat(video_rows),
        audio_indices=torch.cat(audio_rows),
        num_anchor_rows=rows_per_frame,
        video_row_chunk=torch.cat(video_row_chunk),
        audio_row_chunk=torch.cat(audio_row_chunk),
        audio_row_latent=torch.cat(audio_row_latent),
        group_rows=tuple(group_rows),
    )


def _check_windows(window_chunks: int, caption_window_chunks: int) -> None:
    if window_chunks < 1:
        raise ValueError("window_chunks must be positive")
    if not 1 <= caption_window_chunks <= window_chunks:
        raise ValueError(f"caption_window_chunks must be in [1, {window_chunks}], got {caption_window_chunks}")


def visibility(
    query_group: torch.Tensor,
    key_group: torch.Tensor,
    key_is_caption: torch.Tensor,
    window_chunks: int,
    caption_window_chunks: int,
) -> torch.Tensor:
    """The structural mask, elementwise over broadcast query/key tensors."""
    in_window = (key_group <= query_group) & (key_group > query_group - window_chunks)
    in_caption_window = key_group > query_group - caption_window_chunks
    chunk_visible = (query_group > 0) & (key_group > 0) & in_window & (~key_is_caption | in_caption_window)
    return (key_group == 0) | chunk_visible


def build_dense_mask(plan: CausalPlan, window_chunks: int, caption_window_chunks: int) -> torch.Tensor:
    """[S, S] bool, True = attend. Test oracle for the block mask."""
    _check_windows(window_chunks, caption_window_chunks)
    group = plan.row_group
    is_caption = plan.row_role == ROLE_CAPTION
    return visibility(group[:, None], group[None, :], is_caption[None, :], window_chunks, caption_window_chunks)


def build_block_mask(
    plan: CausalPlan, window_chunks: int, caption_window_chunks: int, device: torch.device | str
) -> BlockMask:
    _check_windows(window_chunks, caption_window_chunks)
    group = plan.row_group.to(device)
    is_caption = (plan.row_role == ROLE_CAPTION).to(device)

    def mask_mod(batch, head, query, key):
        return visibility(group[query], group[key], is_caption[key], window_chunks, caption_window_chunks)

    length = plan.sequence_length
    return create_block_mask(mask_mod, None, None, length, length, device=device)
