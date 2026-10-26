"""Chunk-wise autoregressive sampling for causal MiniMax-H3.

Each chunk runs the release's Euler schedule (`num_inference_steps` sigma grid
points, video shift 12, audio shift 3, one forward per step: the checkpoint is
guidance-distilled) against the cached history, then commits its final latents.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from h3_causal.dataset import geometry as geo
from h3_causal.dataset.causal_packing import CausalPlan
from h3_causal.inference.causal_cache import CausalKVCache, group_forward, select_group
from h3_causal.models.transformer import H3Transformer, KVContext
from h3_causal.train.diffusion_forcing import AUDIO_SHIFT, VIDEO_SHIFT, shift_sigma


def release_sigmas(num_inference_steps: int, shift: float) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, num_inference_steps, dtype=torch.float32)
    return torch.unique_consecutive(shift_sigma(base, shift))


@dataclass
class SamplerConfig:
    num_inference_steps: int = 50
    window_chunks: int = 8
    caption_window_chunks: int = 1
    text_timestep: float = 1.0
    commit_timestep: float = 1.0
    # Position recycling: once chunk k > recycle_after_chunk, every shiftable row
    # moves back so chunk k sits where chunk `recycle_after_chunk` sat in training.
    recycle_after_chunk: int | None = None


@dataclass
class SampleOutput:
    video_rows: torch.Tensor  # [5C * rows_per_frame, 96], chunk order
    audio_rows: torch.Tensor  # [Na, 32], plan.audio_indices order


def _euler(x: torch.Tensor, velocity: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    denoised = x + sigma * velocity.float()
    ratio = sigma_next / sigma
    return ratio * x + (1.0 - ratio) * denoised


@torch.no_grad()
def sample_causal(
    model: H3Transformer,
    plan: CausalPlan,
    *,
    keyframe_latents: torch.Tensor,
    text_rows: torch.Tensor,
    config: SamplerConfig,
    generator: torch.Generator,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
) -> SampleOutput:
    """`keyframe_latents` [24, 1, H, W] normalized; `text_rows` in `plan.text_indices` order."""
    cache = CausalKVCache(config.window_chunks, config.caption_window_chunks)
    text_rows = text_rows.to(device, dtype)
    sigmas_video = release_sigmas(config.num_inference_steps, VIDEO_SHIFT)
    sigmas_audio = release_sigmas(config.num_inference_steps, AUDIO_SHIFT)
    if sigmas_video.numel() != sigmas_audio.numel():
        raise ValueError("video and audio schedules must have the same length")

    anchor = geo.patchify_video_latents(keyframe_latents.float())
    anchor_noise = torch.randn(anchor.shape, generator=generator, dtype=torch.float32)
    anchor = (geo.KEYFRAME_NOISE_AUG * anchor + (1.0 - geo.KEYFRAME_NOISE_AUG) * anchor_noise).to(device)

    group0 = select_group(plan, 0)
    row_t = torch.full((group0.rows.numel(),), config.text_timestep)
    row_t[group0.video_indices] = geo.KEYFRAME_NOISE_AUG
    context = KVContext(collect=True)
    group_forward(
        model,
        group0,
        video_rows=anchor.to(dtype),
        audio_rows=anchor.new_zeros(0, model.config.audio_in_channels).to(dtype),
        text_rows=text_rows[group0.text_slice],
        row_t=row_t,
        kv_context=context,
    )
    cache.group0 = context.collected

    video_out, audio_out = [], []
    rows_shape_video = (geo.LATENTS_PER_CHUNK * plan.rows_per_frame, model.proj_in.in_features)
    for chunk in range(plan.num_chunks):
        selection = select_group(plan, chunk + 1)
        num_audio = selection.audio_slice.stop - selection.audio_slice.start
        caption_rows = selection.text_slice.stop - selection.text_slice.start
        prefix, shiftable = cache.visible_prefix(chunk)
        shift = 0.0
        if config.recycle_after_chunk is not None and chunk > config.recycle_after_chunk:
            times = geo.latent_frame_rotary_times(geo.LATENTS_PER_CHUNK * (chunk + 1), 0.0)
            shift = float(
                times[chunk * geo.LATENTS_PER_CHUNK] - times[config.recycle_after_chunk * geo.LATENTS_PER_CHUNK]
            )

        video = torch.randn(rows_shape_video, generator=generator, dtype=torch.float32).to(device)
        audio = torch.randn((num_audio, model.config.audio_in_channels), generator=generator, dtype=torch.float32).to(device)
        text = text_rows[selection.text_slice]
        role_video = selection.token_tags == 0
        role_audio = selection.token_tags == 2

        def forward(t_video: float, t_audio: float, collect: bool):
            row_t = torch.full((selection.rows.numel(),), config.text_timestep)
            row_t[role_video] = t_video
            row_t[role_audio] = t_audio
            context = KVContext(prefix=prefix, prefix_shiftable=shiftable.to(device), collect=collect)
            out = group_forward(
                model,
                selection,
                video_rows=video.to(dtype),
                audio_rows=audio.to(dtype),
                text_rows=text,
                row_t=row_t,
                kv_context=context,
                position_shift=shift,
            )
            return out, context

        for step in range(sigmas_video.numel() - 1):
            sv, sv_next = float(sigmas_video[step]), float(sigmas_video[step + 1])
            sa, sa_next = float(sigmas_audio[step]), float(sigmas_audio[step + 1])
            (v_video, v_audio), _ = forward(1.0 - sv, 1.0 - sa, collect=False)
            video = _euler(video, v_video[0], sv, sv_next)
            audio = _euler(audio, v_audio[0], sa, sa_next)

        video_out.append(video.cpu())
        audio_out.append(audio.cpu())
        if config.commit_timestep < 1.0:
            noise_v = torch.randn(video.shape, generator=generator, dtype=torch.float32).to(device)
            noise_a = torch.randn(audio.shape, generator=generator, dtype=torch.float32).to(device)
            video = config.commit_timestep * video + (1 - config.commit_timestep) * noise_v
            audio = config.commit_timestep * audio + (1 - config.commit_timestep) * noise_a
        _, context = forward(config.commit_timestep, config.commit_timestep, collect=True)
        cache.commit(chunk, context.collected, caption_rows)

    return SampleOutput(torch.cat(video_out), torch.cat(audio_out))


def assemble_latents(plan: CausalPlan, output: SampleOutput, num_audio_latents: int, channels: int = 24):
    """Rows back to normalized latents: video [24, 5C, H, W], audio [2, 32, Ta] (Ta covers the chunks)."""
    frames = geo.LATENTS_PER_CHUNK * plan.num_chunks
    video = geo.unpatchify_video_latents(output.video_rows, channels, frames, plan.latent_height, plan.latent_width)
    audio_channels = output.audio_rows.shape[-1]
    audio = torch.zeros(geo.AUDIO_CHANNELS * num_audio_latents, audio_channels)
    audio[plan.audio_row_latent] = output.audio_rows
    audio = audio.view(geo.AUDIO_CHANNELS, num_audio_latents, audio_channels).permute(0, 2, 1)
    covered = geo.chunk_audio_range(plan.num_chunks - 1)[1]
    return video, audio[:, :, :covered]
