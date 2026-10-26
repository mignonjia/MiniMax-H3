"""Tiny-model fixtures shared by the causal H3 tests."""

from __future__ import annotations

import torch

from h3_causal.dataset import geometry as geo
from h3_causal.dataset.causal_packing import build_causal_plan
from h3_causal.models.transformer import TEXT_TAG, VIDEO_TAG, H3Config, H3Transformer

TINY = dict(
    num_attention_heads=2,
    attention_head_dim=32,
    hidden_size=48,
    num_layers=3,
    num_refiner_layers=2,
    ffn_dim=64,
    in_channels=24,
    audio_in_channels=32,
    text_dim=40,
    freq_dim=32,
    time_embed_hidden_dim=48,
    time_embed_dim=24,
    rope_freq_dim=4,
)


def tiny_model(seed: int = 0) -> H3Transformer:
    torch.manual_seed(seed)
    model = H3Transformer(H3Config(**TINY)).float().eval()
    # Zero-initialised AdaLN would hide timestep bugs; make every parameter non-trivial.
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.2)
    return model


def tiny_sample(num_chunks: int = 4, caption_lengths=None, seed: int = 0, latent_hw=(4, 6)):
    gen = torch.Generator().manual_seed(seed)
    if caption_lengths is None:
        caption_lengths = [3 + (k % 2) for k in range(num_chunks)]
    global_tags = torch.full((7,), TEXT_TAG)
    global_tags[2:4] = VIDEO_TAG  # a keyframe vision block inside the global text
    num_audio = geo.chunk_audio_range(num_chunks - 1)[1] + 2
    plan = build_causal_plan(
        global_text_tags=global_tags,
        caption_lengths=list(caption_lengths),
        latent_height=latent_hw[0],
        latent_width=latent_hw[1],
        num_audio_latents=num_audio,
    )
    text = torch.randn(plan.text_indices.numel(), TINY["text_dim"], generator=gen)
    video = torch.randn(24, geo.LATENTS_PER_CHUNK * num_chunks, *latent_hw, generator=gen)
    keyframe = torch.randn(24, 1, *latent_hw, generator=gen)
    audio = torch.randn(2, 32, num_audio, generator=gen)
    return plan, text, video, keyframe, audio
