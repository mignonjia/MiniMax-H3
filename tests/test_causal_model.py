"""End-to-end causality of the masked model and KV-cache equivalence with the full forward."""

import pytest
import torch

from h3_causal.dataset.causal_packing import ROLE_CAPTION, ROLE_VIDEO, build_dense_mask
from h3_causal.inference.causal_cache import CausalKVCache, group_forward, select_group
from h3_causal.models.transformer import KVContext
from h3_causal.train.diffusion_forcing import row_timesteps
from h3_causal.dataset import geometry as geo
from tests.helpers import tiny_model, tiny_sample

WINDOW, CAPTION_WINDOW = 2, 1


def _full_forward(model, plan, video_rows, audio_rows, text, video_t, audio_t, text_t=1.0):
    timestep, timestep_indices = row_timesteps(plan, video_t, audio_t, text_timestep=text_t)
    mask = build_dense_mask(plan, WINDOW, CAPTION_WINDOW)
    with torch.no_grad():
        return model(
            hidden_states=video_rows[None],
            audio_hidden_states=audio_rows[None],
            encoder_hidden_states=text[None],
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=plan.token_tags,
            position_ids=plan.position_ids,
            video_indices=plan.video_indices,
            audio_indices=plan.audio_indices,
            text_indices=plan.text_indices,
            text_segment_ids=plan.text_segment_ids,
            attention_mask=mask,
        )


def _inputs(seed=0, num_chunks=4):
    plan, text, video, keyframe, audio = tiny_sample(num_chunks=num_chunks, seed=seed)
    video_rows = torch.cat([geo.patchify_video_latents(keyframe), geo.patchify_video_latents(video)])
    audio_flat = audio.permute(0, 2, 1).reshape(-1, 32)
    audio_rows = audio_flat[plan.audio_row_latent]
    gen = torch.Generator().manual_seed(seed + 1)
    video_t = torch.rand(num_chunks, generator=gen)
    audio_t = torch.rand(num_chunks, generator=gen)
    return plan, text, video_rows, audio_rows, video_t, audio_t


def _group_outputs(plan, out_video, out_audio, group):
    vrows = plan.video_indices
    in_group = plan.row_group[vrows] == group
    arows = plan.audio_indices
    return out_video[0][in_group], out_audio[0][plan.row_group[arows] == group]


@pytest.mark.parametrize("perturb", ["future_video", "future_caption"])
def test_masked_model_is_causal(perturb):
    """Nothing in a later chunk reaches an earlier one.

    Older captions and chunks outside the window still reach a chunk indirectly,
    through the hidden states of the in-window chunks that read them one layer
    earlier; the KV cache reproduces exactly that (see the equivalence test).
    """
    model = tiny_model()
    plan, text, video_rows, audio_rows, video_t, audio_t = _inputs()
    base = _full_forward(model, plan, video_rows, audio_rows, text, video_t, audio_t)
    text2, video2 = text.clone(), video_rows.clone()
    # Chunk index 2 (group 3) is the observer.
    observer = 3
    if perturb == "future_video":
        video2[plan.row_group[plan.video_indices] == 4] += 1.0
    elif perturb == "future_caption":
        text2[plan.text_segment_ids == 4] += 1.0
    new = _full_forward(model, plan, video2, audio_rows, text2, video_t, audio_t)
    for a, b in zip(_group_outputs(plan, *base, observer), _group_outputs(plan, *new, observer)):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_old_caption_is_visible_with_wider_caption_window():
    model = tiny_model()
    plan, text, video_rows, audio_rows, video_t, audio_t = _inputs()
    timestep, timestep_indices = row_timesteps(plan, video_t, audio_t, text_timestep=1.0)

    def run(text_rows):
        with torch.no_grad():
            return model(
                hidden_states=video_rows[None], audio_hidden_states=audio_rows[None],
                encoder_hidden_states=text_rows[None], timestep=timestep, timestep_indices=timestep_indices,
                token_tags=plan.token_tags, position_ids=plan.position_ids, video_indices=plan.video_indices,
                audio_indices=plan.audio_indices, text_indices=plan.text_indices,
                text_segment_ids=plan.text_segment_ids, attention_mask=build_dense_mask(plan, 2, 2),
            )

    text2 = text.clone()
    text2[plan.text_segment_ids == 2] += 1.0
    a = _group_outputs(plan, *run(text), 3)[0]
    b = _group_outputs(plan, *run(text2), 3)[0]
    assert (a - b).abs().max() > 1e-4


def test_kv_cache_matches_full_forward():
    """History committed at `commit_t` through the cache == full masked forward with history at `commit_t`."""
    model = tiny_model()
    num_chunks = 4
    plan, text, video_rows, audio_rows, _, _ = _inputs(num_chunks=num_chunks)
    commit_t, current_t, current_audio_t = 1.0, 0.37, 0.61
    observer_chunk = num_chunks - 1
    video_t = torch.full((num_chunks,), commit_t)
    audio_t = torch.full((num_chunks,), commit_t)
    video_t[observer_chunk], audio_t[observer_chunk] = current_t, current_audio_t
    full = _full_forward(model, plan, video_rows, audio_rows, text, video_t, audio_t)

    cache = CausalKVCache(WINDOW, CAPTION_WINDOW)
    group0 = select_group(plan, 0)
    row_t = torch.ones(group0.rows.numel())
    row_t[group0.video_indices] = geo.KEYFRAME_NOISE_AUG
    with torch.no_grad():
        context = KVContext(collect=True)
        group_forward(
            model, group0, video_rows=video_rows[group0.video_slice], audio_rows=audio_rows[:0],
            text_rows=text[group0.text_slice], row_t=row_t, kv_context=context,
        )
        cache.group0 = context.collected
        for chunk in range(num_chunks):
            selection = select_group(plan, chunk + 1)
            prefix, shiftable = cache.visible_prefix(chunk)
            row_t = torch.ones(selection.rows.numel())
            row_t[selection.token_tags == 0] = float(video_t[chunk])
            row_t[selection.token_tags == 2] = float(audio_t[chunk])
            context = KVContext(prefix=prefix, prefix_shiftable=shiftable, collect=chunk != observer_chunk)
            out = group_forward(
                model, selection, video_rows=video_rows[selection.video_slice],
                audio_rows=audio_rows[selection.audio_slice], text_rows=text[selection.text_slice],
                row_t=row_t, kv_context=context,
            )
            if chunk != observer_chunk:
                caption_rows = selection.text_slice.stop - selection.text_slice.start
                cache.commit(chunk, context.collected, caption_rows)
        # Earlier chunks fell out of the ring.
        assert set(cache.chunks) == {observer_chunk - 1}

    expected = _group_outputs(plan, *full, observer_chunk + 1)
    torch.testing.assert_close(out[0][0], expected[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out[1][0], expected[1], rtol=1e-4, atol=1e-4)


def test_position_shift_is_translation_invariant_for_chunk_rows():
    """RoPE is relative: shifting the whole visible chunk history leaves outputs unchanged when group 0 is masked out."""
    model = tiny_model()
    plan, text, video_rows, audio_rows, video_t, audio_t = _inputs()
    selection = select_group(plan, 2)
    row_t = torch.full((selection.rows.numel(),), 0.5)
    with torch.no_grad():
        a = group_forward(
            model, selection, video_rows=video_rows[selection.video_slice], audio_rows=audio_rows[selection.audio_slice],
            text_rows=text[selection.text_slice], row_t=row_t, kv_context=None, position_shift=0.0,
        )
        b = group_forward(
            model, selection, video_rows=video_rows[selection.video_slice], audio_rows=audio_rows[selection.audio_slice],
            text_rows=text[selection.text_slice], row_t=row_t, kv_context=None, position_shift=13.0,
        )
    torch.testing.assert_close(a[0], b[0], rtol=1e-4, atol=1e-4)
