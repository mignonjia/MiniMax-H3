"""Exact-resume invariants of the sampler and the per-sample corruption."""

import itertools

import torch

from h3_causal.dataset.causal_parquet_dataset import ResumableSampler
from h3_causal.train.diffusion_forcing import build_bsmntw_weights, build_df_batch
from tests.helpers import tiny_sample


def test_sampler_resumes_from_consumed_count():
    full = ResumableSampler(10, rank=1, world_size=3, seed=7)
    reference = list(itertools.islice(iter(full), 11))  # crosses two epoch boundaries (3 per rank)
    resumed = ResumableSampler(10, rank=1, world_size=3, seed=7)
    resumed.set_consumed(4)
    assert list(itertools.islice(iter(resumed), 7)) == reference[4:]
    # Ranks partition each epoch.
    epoch0 = [list(itertools.islice(iter(ResumableSampler(10, rank=r, world_size=3, seed=7)), 3)) for r in range(3)]
    indices = [i for ranks in epoch0 for _, i in ranks]
    assert len(set(indices)) == 9


def test_corruption_is_a_function_of_the_seed():
    plan, text, video, keyframe, audio = tiny_sample(num_chunks=3)

    def batch(seed):
        return build_df_batch(
            plan, video_latents=video, keyframe_latents=keyframe, audio_latents=audio, text_rows=text,
            rng=torch.Generator().manual_seed(seed), num_train_timesteps=1000,
            bsmntw_weights=build_bsmntw_weights(1000), text_timestep=1.0,
        )

    a, b, c = batch(3), batch(3), batch(4)
    assert torch.equal(a.video_rows, b.video_rows) and torch.equal(a.audio_rows, b.audio_rows)
    assert not torch.equal(a.video_rows, c.video_rows)
