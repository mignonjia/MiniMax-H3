"""MiniMax-H3 latent geometry for chunk-causal training.

The video VAE encodes independent 17-frame clips into 5 latent frames (the first
covers one pixel frame, the others four each); `17n + 5` pixel frames give
`5n + 2` latents. One causal chunk is one VAE clip: 5 latents, 17 frames,
0.708 s at 24 fps. The 2-latent tail of an encode is dropped.

Audio runs at 40 latents/s per stereo channel on the same rotary clock as video
(one unit per audio latent, 5/3 units per pixel frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

FPS = 24
VAE_CLIP_FRAMES = 17
LATENTS_PER_CHUNK = 5
SPATIAL_COMPRESSION = 16
PATCH_SIZE = (1, 2, 2)
CANVAS_MULTIPLE = SPATIAL_COMPRESSION * PATCH_SIZE[2]
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
AUDIO_SAMPLE_RATE = 32000
KEYFRAME_NOISE_AUG = 0.999

ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32


def aligned_num_frames(num_chunks: int) -> int:
    """Pixel frames to encode for `num_chunks` causal chunks (`17n + 5`, the VAE's encodable length)."""
    return VAE_CLIP_FRAMES * num_chunks + LATENTS_PER_CHUNK


def max_chunks_for_frames(num_frames: int) -> int:
    return max(0, (num_frames - LATENTS_PER_CHUNK) // VAE_CLIP_FRAMES)


def audio_latent_boundary(pixel_frame: int) -> int:
    """Audio latent index at a pixel-frame boundary, rounded on the latent grid like the release."""
    return int(round(pixel_frame / FPS * AUDIO_LATENTS_PER_SECOND))


def chunk_audio_range(chunk: int) -> tuple[int, int]:
    return audio_latent_boundary(VAE_CLIP_FRAMES * chunk), audio_latent_boundary(VAE_CLIP_FRAMES * (chunk + 1))


def chunk_time_range_sec(chunk: int) -> tuple[float, float]:
    return VAE_CLIP_FRAMES * chunk / FPS, VAE_CLIP_FRAMES * (chunk + 1) / FPS


def latent_frame_rotary_times(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of each latent frame: spacing `5/3 * (1, 4, 4, 4, 4)`, first frame at `origin`."""
    spans = torch.tensor(
        [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def frame_position_grid(latent_height: int, latent_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h, w) rotary coordinates of one latent frame's patches, and the width axis."""
    _, patch_h, patch_w = PATCH_SIZE
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([g.reshape(-1) for g in grids], dim=-1), width_grid


def patchify_video_latents(latents: torch.Tensor) -> torch.Tensor:
    """[C, T, H, W] -> [T * H/2 * W/2, C * 4], frame-major then row-major (release order)."""
    patch_t, patch_h, patch_w = PATCH_SIZE
    channels, frames, height, width = latents.shape
    latents = latents.reshape(channels, frames // patch_t, patch_t, height // patch_h, patch_h, width // patch_w, patch_w)
    latents = latents.permute(1, 3, 5, 0, 2, 4, 6)
    return latents.reshape(-1, channels * patch_t * patch_h * patch_w)


def unpatchify_video_latents(rows: torch.Tensor, channels: int, frames: int, height: int, width: int) -> torch.Tensor:
    patch_t, patch_h, patch_w = PATCH_SIZE
    rows = rows.reshape(frames // patch_t, height // patch_h, width // patch_w, channels, patch_t, patch_h, patch_w)
    return rows.permute(3, 0, 4, 1, 5, 2, 6).reshape(channels, frames, height, width)


@dataclass(frozen=True)
class Canvas:
    height: int
    width: int

    @property
    def latent_height(self) -> int:
        return self.height // SPATIAL_COMPRESSION

    @property
    def latent_width(self) -> int:
        return self.width // SPATIAL_COMPRESSION

    @property
    def rows_per_frame(self) -> int:
        return (self.latent_height // PATCH_SIZE[1]) * (self.latent_width // PATCH_SIZE[2])


def resolve_canvas(aspect_width: float, aspect_height: float, short_edge: int, max_pixels: int) -> Canvas:
    """Release canvas rule (short edge, area cap, 32-multiple) at an arbitrary short edge, e.g. 480."""
    ratio = aspect_width / aspect_height
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    m = CANVAS_MULTIPLE
    return Canvas(max(m, round(height / m) * m), max(m, round(width / m) * m))
