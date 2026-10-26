"""Layout, mask and DF invariants of the packed causal H3 sample."""

import pytest
import torch

from h3_causal.dataset import geometry as geo
from h3_causal.dataset.causal_packing import (
    ROLE_ANCHOR,
    ROLE_AUDIO,
    ROLE_CAPTION,
    ROLE_GLOBAL_TEXT,
    ROLE_VIDEO,
    build_block_mask,
    build_dense_mask,
)
from h3_causal.train.diffusion_forcing import build_bsmntw_weights, build_df_batch, row_timesteps
from tests.helpers import tiny_sample


def test_geometry_chunks_tile_the_audio_clock():
    ranges = [geo.chunk_audio_range(k) for k in range(30)]
    assert ranges[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
    # 17 frames at 24 fps = 28.33 audio latents per chunk.
    assert {end - start for start, end in ranges} == {28, 29}
    times = geo.latent_frame_rotary_times(10, 0.0)
    # One VAE clip spans 17 frames = 17 * 5/3 rotary units = its audio span in latents.
    assert float(times[5] - times[0]) == pytest.approx(17 * 5 / 3)


def test_patchify_roundtrip():
    latents = torch.randn(24, 10, 6, 8)
    rows = geo.patchify_video_latents(latents)
    assert rows.shape == (10 * 3 * 4, 96)
    torch.testing.assert_close(geo.unpatchify_video_latents(rows, 24, 10, 6, 8), latents)


def test_canvas_480p():
    canvas = geo.resolve_canvas(16, 9, 480, 480 * 864)
    assert (canvas.height, canvas.width) == (480, 864)
    assert canvas.rows_per_frame == 15 * 27


def test_layout_groups_and_positions():
    plan, *_ = tiny_sample(num_chunks=3, caption_lengths=[3, 0, 4])
    assert plan.sequence_length == plan.position_ids.shape[0] == sum(plan.group_rows)
    # Every row is exactly one of text / video / audio.
    covered = torch.cat([plan.text_indices, plan.video_indices, plan.audio_indices]).sort().values
    assert torch.equal(covered, torch.arange(plan.sequence_length))
    # Groups are contiguous and increasing.
    assert torch.all(plan.row_group[1:] >= plan.row_group[:-1])
    origin = float((plan.row_role == ROLE_GLOBAL_TEXT).sum())
    anchor = plan.position_ids[plan.row_role == ROLE_ANCHOR]
    assert torch.all(anchor[:, 0] == origin)
    first_video = plan.position_ids[plan.row_role == ROLE_VIDEO][0]
    assert first_video[0] == origin
    # A caption ends right before its chunk's first latent frame.
    for chunk, length in enumerate([3, 0, 4]):
        if not length:
            continue
        rows = (plan.row_group == chunk + 1) & (plan.row_role == ROLE_CAPTION)
        video = (plan.row_group == chunk + 1) & (plan.row_role == ROLE_VIDEO)
        assert float(plan.position_ids[rows, 0].max()) == float(plan.position_ids[video, 0].min()) - 1
    # Audio rows share the video clock.
    audio = plan.position_ids[plan.row_role == ROLE_AUDIO]
    assert float(audio[:, 0].min()) == origin


def test_dense_mask_semantics():
    plan, *_ = tiny_sample(num_chunks=5)
    mask = build_dense_mask(plan, window_chunks=3, caption_window_chunks=1)
    group, role = plan.row_group, plan.row_role

    def rows(g, r=None):
        sel = group == g
        if r is not None:
            sel &= role == r
        return sel.nonzero().flatten()

    q = rows(0)[0]
    assert mask[q, rows(0)].all() and not mask[q, group > 0].any()
    q = rows(4, ROLE_VIDEO)[0]  # chunk 3
    assert mask[q, rows(0)].all()
    assert mask[q, rows(4)].all()  # own group incl. own caption
    for g in (2, 3):  # window 3: chunks 1, 2 visible without their captions
        assert mask[q, rows(g, ROLE_VIDEO)].all() and mask[q, rows(g, ROLE_AUDIO)].all()
        assert not mask[q, rows(g, ROLE_CAPTION)].any()
    assert not mask[q, rows(1)].any()  # outside the window
    assert not mask[q, rows(5)].any()  # future
    # Caption queries read exactly what their chunk's video reads.
    caption_q = rows(4, ROLE_CAPTION)[0]
    assert torch.equal(mask[caption_q], mask[q])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention runs on GPU")
def test_block_mask_matches_dense_mask():
    from torch.nn.attention.flex_attention import flex_attention

    plan, *_ = tiny_sample(num_chunks=6, latent_hw=(8, 8))
    dense = build_dense_mask(plan, 3, 2).cuda()
    block = build_block_mask(plan, 3, 2, "cuda")
    q, k, v = (torch.randn(1, 2, plan.sequence_length, 32, device="cuda") for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense)
    actual = flex_attention(q, k, v, block_mask=block)
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_df_batch_rows_and_timesteps():
    plan, text, video, keyframe, audio = tiny_sample(num_chunks=4)
    rng = torch.Generator().manual_seed(0)
    batch = build_df_batch(
        plan,
        video_latents=video,
        keyframe_latents=keyframe,
        audio_latents=audio,
        text_rows=text,
        rng=rng,
        num_train_timesteps=1000,
        bsmntw_weights=build_bsmntw_weights(1000),
        text_timestep=1.0,
    )
    assert batch.video_rows.shape[0] == plan.video_indices.numel()
    assert batch.audio_rows.shape[0] == plan.audio_indices.numel()
    row_t = batch.timestep[batch.timestep_indices]
    text = (plan.row_role == ROLE_GLOBAL_TEXT) | (plan.row_role == ROLE_CAPTION)
    assert torch.all(row_t[text] == 1.0)
    assert torch.allclose(row_t[plan.row_role == ROLE_ANCHOR], torch.tensor(0.999))
    for chunk in range(plan.num_chunks):
        video_rows = (plan.row_group == chunk + 1) & (plan.row_role == ROLE_VIDEO)
        audio_rows = (plan.row_group == chunk + 1) & (plan.row_role == ROLE_AUDIO)
        assert torch.unique(row_t[video_rows]).numel() == 1
        assert torch.unique(row_t[audio_rows]).numel() == 1
        # Audio uses the gentler shift: never noisier than video at the same index.
        assert float(row_t[audio_rows][0]) >= float(row_t[video_rows][0]) - 1e-6
    # x_t = t x0 + (1 - t) noise and target = x0 - noise => x0 = x_t + (1 - t) * target.
    clean = geo.patchify_video_latents(video)
    t_rows = batch.chunk_timesteps.video[plan.video_row_chunk[plan.num_anchor_rows :]][:, None]
    recon = batch.video_rows[plan.num_anchor_rows :] + (1 - t_rows) * batch.video_target
    torch.testing.assert_close(recon, clean, rtol=1e-4, atol=1e-4)


def test_row_timesteps_text_independent_of_chunk_noise():
    plan, *_ = tiny_sample(num_chunks=3)
    a = row_timesteps(plan, torch.tensor([0.1, 0.5, 0.9]), torch.tensor([0.2, 0.6, 0.95]), text_timestep=1.0)
    b = row_timesteps(plan, torch.tensor([0.7, 0.2, 0.3]), torch.tensor([0.8, 0.3, 0.4]), text_timestep=1.0)
    text = (plan.row_role == ROLE_GLOBAL_TEXT) | (plan.row_role == ROLE_CAPTION)
    assert torch.equal(a[0][a[1]][text], b[0][b[1]][text])
