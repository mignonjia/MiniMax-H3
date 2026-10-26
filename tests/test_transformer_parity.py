"""The standalone transformer reproduces diffusers' MiniMaxH3Transformer3DModel on the release layout."""

import pytest
import torch

from tests.helpers import TINY, tiny_model

diffusers = pytest.importorskip("diffusers")
if not hasattr(diffusers, "MiniMaxH3Transformer3DModel"):
    pytest.skip("needs diffusers >= 0.40 (set H3_DIFFUSERS_PATH)", allow_module_level=True)

from diffusers.modular_pipelines.minimax_h3.before_denoise import (  # noqa: E402
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)


def _reference(model):
    reference = diffusers.MiniMaxH3Transformer3DModel(
        num_attention_heads=TINY["num_attention_heads"],
        attention_head_dim=TINY["attention_head_dim"],
        hidden_size=TINY["hidden_size"],
        num_layers=TINY["num_layers"],
        num_refiner_layers=TINY["num_refiner_layers"],
        ffn_dim=TINY["ffn_dim"],
        text_dim=TINY["text_dim"],
        freq_dim=TINY["freq_dim"],
        time_embed_hidden_dim=TINY["time_embed_hidden_dim"],
        time_embed_dim=TINY["time_embed_dim"],
        rope_freq_dim=TINY["rope_freq_dim"],
    ).float()
    missing, unexpected = reference.load_state_dict(model.state_dict(), strict=True)
    assert not missing and not unexpected
    return reference.eval()


def test_state_dict_keys_match_release_checkpoint_names():
    model = tiny_model()
    keys = set(model.state_dict())
    assert "transformer_blocks.0.adaln_proj.linear.weight" in keys
    assert "token_refiner.refiner_blocks.1.ff.net.0.proj.weight" in keys
    assert "norm_out.linear.weight" in keys


@pytest.mark.parametrize("keyframe", [False, True])
def test_bidirectional_forward_matches_diffusers(keyframe):
    model = tiny_model()
    reference = _reference(model)
    gen = torch.Generator().manual_seed(1)
    text_tags = torch.ones(9, dtype=torch.long)
    if keyframe:
        text_tags[3:5] = 0
    layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_tags, 7, 4, 6, 12, (1, 2, 2), 2, 2, 0, ("first",) if keyframe else ()
    )
    position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond, _ = layout
    timestep, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, num_cond, 0, text_indices.numel(), 0.3, 0.6, 0.999, 1.0
    )
    inputs = dict(
        hidden_states=torch.randn(1, video_indices.numel(), 96, generator=gen),
        audio_hidden_states=torch.randn(1, audio_indices.numel(), 32, generator=gen),
        encoder_hidden_states=torch.randn(1, text_indices.numel(), TINY["text_dim"], generator=gen),
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
    )
    with torch.no_grad():
        expected = reference(**inputs, return_dict=False)
        actual = model(**inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-4)
