"""Diffusion forcing for MiniMax-H3: per-chunk noise levels on the packed causal layout.

H3 convention: `t` in [0, 1] with t = 1 clean, `x_t = t * x0 + (1 - t) * noise`,
the model predicts the data-ward velocity `v = x0 - noise`. Video and audio
share one schedule index per chunk and map it through their own release shifts
(12 and 3), which is how the release couples the two modalities at inference.

Text rows are pinned to a constant timestep so their K/V never depend on the
noise of a chunk (the release lets text inherit the video timestep, which would
invalidate a text KV cache under per-chunk noise). The anchor stays at the
release's keyframe level, 0.999.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from h3_causal.dataset import geometry as geo
from h3_causal.dataset.causal_packing import ROLE_ANCHOR, ROLE_AUDIO, ROLE_CAPTION, ROLE_GLOBAL_TEXT, ROLE_VIDEO, CausalPlan

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def build_bsmntw_weights(num_train_timesteps: int, *, device: torch.device | str | None = None) -> torch.Tensor:
    """Mean-one BSMNTW weights over discrete schedule indices (PAN DF recipe)."""
    if num_train_timesteps < 2:
        raise ValueError("num_train_timesteps must be at least 2")
    n = float(num_train_timesteps)
    indices = torch.arange(num_train_timesteps, dtype=torch.float32, device=device)
    values = torch.exp(-2 * ((indices - n / 2) / n).square())
    shifted = values - values.min()
    return shifted * (n / shifted.sum())


@dataclass
class ChunkTimesteps:
    index: torch.Tensor  # [C] schedule index, 0 = noisiest
    video: torch.Tensor  # [C] H3 t of the video rows
    audio: torch.Tensor  # [C] H3 t of the audio rows
    weight: torch.Tensor  # [C] loss weight


def sample_chunk_timesteps(
    num_chunks: int,
    *,
    num_train_timesteps: int,
    rng: torch.Generator,
    bsmntw_weights: torch.Tensor | None,
) -> ChunkTimesteps:
    """One discrete-uniform schedule index per chunk; sigma_u = (T - i) / T in (0, 1]."""
    index = torch.randint(0, num_train_timesteps, (num_chunks,), generator=rng)
    sigma = (num_train_timesteps - index).to(torch.float64) / num_train_timesteps
    video = 1.0 - shift_sigma(sigma, VIDEO_SHIFT)
    audio = 1.0 - shift_sigma(sigma, AUDIO_SHIFT)
    weight = bsmntw_weights.index_select(0, index) if bsmntw_weights is not None else torch.ones(num_chunks)
    return ChunkTimesteps(index, video.float(), audio.float(), weight.float())


def row_timesteps(
    plan: CausalPlan,
    video_t: torch.Tensor,
    audio_t: torch.Tensor,
    *,
    text_timestep: float,
    anchor_timestep: float = geo.KEYFRAME_NOISE_AUG,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row t reduced to the transformer's `(timestep, timestep_indices)` pair."""
    row_t = torch.empty(plan.sequence_length, dtype=torch.float32)
    role = plan.row_role
    text = (role == ROLE_GLOBAL_TEXT) | (role == ROLE_CAPTION)
    row_t[text] = text_timestep
    row_t[role == ROLE_ANCHOR] = anchor_timestep
    chunk_of_row = plan.row_group - 1
    video = role == ROLE_VIDEO
    audio = role == ROLE_AUDIO
    row_t[video] = video_t.float().index_select(0, chunk_of_row[video])
    row_t[audio] = audio_t.float().index_select(0, chunk_of_row[audio])
    return torch.unique(row_t, sorted=True, return_inverse=True)


def noise_rows(clean: torch.Tensor, t: torch.Tensor, rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `x_t` and the velocity target `x0 - noise`, noise drawn in fp32 on the CPU generator."""
    noise = torch.randn(clean.shape, generator=rng, dtype=torch.float32).to(clean.device)
    clean = clean.float()
    t = t.float().to(clean.device)[:, None]
    return t * clean + (1.0 - t) * noise, clean - noise


@dataclass
class DFBatch:
    """One sample, ready for the transformer, with its loss weights."""

    plan: CausalPlan
    video_rows: torch.Tensor  # [Nv, 96] noised (anchor rows first)
    audio_rows: torch.Tensor  # [Na, 32]
    text_rows: torch.Tensor  # [Nt, 5120]
    timestep: torch.Tensor
    timestep_indices: torch.Tensor
    video_target: torch.Tensor  # [Nv - anchor, 96]
    audio_target: torch.Tensor  # [Na, 32]
    video_weight: torch.Tensor  # [Nv - anchor] per-row loss weight
    audio_weight: torch.Tensor  # [Na]
    chunk_timesteps: ChunkTimesteps


def build_df_batch(
    plan: CausalPlan,
    *,
    video_latents: torch.Tensor,
    keyframe_latents: torch.Tensor,
    audio_latents: torch.Tensor,
    text_rows: torch.Tensor,
    rng: torch.Generator,
    num_train_timesteps: int,
    bsmntw_weights: torch.Tensor | None,
    text_timestep: float,
) -> DFBatch:
    """Noise every chunk at its own level; the anchor is the release-noised keyframe.

    `video_latents` [24, >=5C, H, W], `keyframe_latents` [24, 1, H, W] and
    `audio_latents` [2, 32, Ta] are normalized latents.
    """
    num_chunks = plan.num_chunks
    steps = sample_chunk_timesteps(
        num_chunks, num_train_timesteps=num_train_timesteps, rng=rng, bsmntw_weights=bsmntw_weights
    )

    anchor = geo.patchify_video_latents(keyframe_latents.float())
    anchor_noise = torch.randn(anchor.shape, generator=rng, dtype=torch.float32).to(anchor.device)
    anchor = geo.KEYFRAME_NOISE_AUG * anchor + (1.0 - geo.KEYFRAME_NOISE_AUG) * anchor_noise

    chunk_rows = geo.patchify_video_latents(video_latents[:, : num_chunks * geo.LATENTS_PER_CHUNK].float())
    video_chunk = plan.video_row_chunk[plan.num_anchor_rows :].to(chunk_rows.device)
    video_noised, video_target = noise_rows(chunk_rows, steps.video.to(chunk_rows.device)[video_chunk], rng)

    audio_flat = audio_latents.float().permute(0, 2, 1).reshape(-1, audio_latents.shape[1])
    audio_clean = audio_flat.index_select(0, plan.audio_row_latent.to(audio_flat.device))
    audio_chunk = plan.audio_row_chunk.to(audio_clean.device)
    audio_noised, audio_target = noise_rows(audio_clean, steps.audio.to(audio_clean.device)[audio_chunk], rng)

    timestep, timestep_indices = row_timesteps(plan, steps.video, steps.audio, text_timestep=text_timestep)
    weight = steps.weight.to(chunk_rows.device)
    return DFBatch(
        plan=plan,
        video_rows=torch.cat([anchor, video_noised]),
        audio_rows=audio_noised,
        text_rows=text_rows,
        timestep=timestep,
        timestep_indices=timestep_indices,
        video_target=video_target,
        audio_target=audio_target,
        video_weight=weight[video_chunk],
        audio_weight=weight[audio_chunk],
        chunk_timesteps=steps,
    )


def df_loss(
    video_pred: torch.Tensor,
    audio_pred: torch.Tensor,
    batch: DFBatch,
    *,
    audio_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted MSE over generated rows; the anchor rows are excluded. Each modality is a row mean."""
    num_anchor = batch.plan.num_anchor_rows
    video_err = (video_pred[num_anchor:].float() - batch.video_target).square().mean(dim=-1)
    audio_err = (audio_pred.float() - batch.audio_target).square().mean(dim=-1)
    video_loss = (video_err * batch.video_weight).sum() / video_err.numel()
    audio_loss = (audio_err * batch.audio_weight).sum() / audio_err.numel()
    loss = video_loss + audio_loss_weight * audio_loss
    stats = {
        "loss": loss.detach(),
        "loss_video": video_loss.detach(),
        "loss_audio": audio_loss.detach(),
        "mse_video_unweighted": video_err.detach().mean(),
        "mse_audio_unweighted": audio_err.detach().mean(),
    }
    return loss, stats
