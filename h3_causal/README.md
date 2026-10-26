# Causal H3

Diffusion-forcing (DF) fine-tune of the released MiniMax-H3 (FL2VA, 33B single-stream, joint
video + stereo audio) into a chunk-causal model, ported from PAN's causal recipe in
`WM-Game-Training` (main + `mingjia/df-own-caption-runs`).

## Recipe

| | PAN (game v7) | H3 |
|---|---|---|
| chunk | 4 latents = 1 s at 16 fps | 5 latents = one VAE clip = 17 frames = 0.708 s at 24 fps |
| anchor | clean first latent, t = 0 | release keyframe rows (single-frame encode), t = 0.999 |
| per-chunk noise | discrete-uniform index, shift 5, BSMNTW weights | same index per chunk, video shift 12 / audio shift 3, BSMNTW |
| text timestep | timestep-free text modulation | text rows pinned to `text_timestep` (1.0) |
| captions | global + own chunk caption (`caption_window_chunks: 1`), times dropped | same |
| mask kernel | FA4 block-sparse on 128-token tiles | FlexAttention `BlockMask` over group ids |
| guidance | CFG 4 | none: the checkpoint is guidance-distilled |

Layout (rows in attention-group order; the transformer scatters by index):

```
group 0      [ global text (+ "<Picture 1>:" vision block) | anchor ]
group k + 1  [ caption_k | audio_k ch0, ch1 | video_k (5 latent frames) ]
```

Group 0 sees itself; chunk group g sees group 0, audio/video of groups g-W+1..g and its own caption.
RoPE: global text at t = 0..L-1, media from origin L on the release's 40-units/s clock; a chunk
caption sits right before its chunk's first latent frame.

## H3-specific pitfalls found while porting

- **Text inherits the video timestep in the release** (`build_row_timesteps`). Under per-chunk noise that
  makes text K/V depend on chunk noise and breaks the KV cache, so text is pinned (`row_timesteps`).
- **The token refiner is full self-attention over all text**: it runs per caption segment
  (block-diagonal), otherwise later captions leak into earlier ones.
- **VAE tiling is part of the latent distribution.** Tiled (256 px) vs untiled encode differ by 47% relative;
  tiled latents normalise to std 1.0 under the release `latents_mean/std`, untiled to ~0.77. Untiled
  *decode* drops reconstruction from 30 dB to 21 dB. Keep the release's tiled encode and decode.
- **FSDP2 must not cast forward inputs**: the default would cast float64 RoPE positions and the
  timesteps to bf16. Policies use `cast_forward_inputs=False`; fp32 release heads keep an fp32 policy.
- **`x0 = x_t + sigma * v`**: H3 predicts the data-ward velocity; target is `x0 - noise`, t = 1 is clean.
- **VAE geometry**: `17n + 5` frames -> `5n + 2` latents, each 17-frame clip encoded independently (latent 0 of
  a clip is one frame), so a 5-latent chunk is encoder-causal; the decoder cross-fades with 2 latents of
  look-ahead (stream-decode must lag one chunk).
- **Indirect history flow is inherent**: older captions / out-of-window chunks still reach chunk k through
  in-window hidden states one layer earlier; the KV cache reproduces exactly this (tested).

## Run

Environment: `/mnt/weka/shrd/wm/.conda/wm-production-h200-cu129` plus diffusers 0.40 and pyarrow from
`H3_PYLIBS=/mnt/weka/shrd/wm/mingjia/pylibs/h3` (`pip install --target`; the shared env is untouched).
Weights: `/mnt/weka/shrd/wm/areeb/H3` (diffusers layout).

```bash
# manifests: game raw windows (on weka) or Loki CSVs + locally fetched mp4s
python -m h3_causal.preprocess.manifest game --num-clips 3000 --out game.jsonl
python -m h3_causal.preprocess.manifest loki --csv-glob '/mnt/weka/shrd/wm/zhiyang/loki_h3_joint_training_csvs_60n_cp4_120shards/*.csv' --local-root /path/to/loki --out loki.jsonl
# encode (restartable)
sbatch [--nodes N] h3_causal/scripts/encode.sh game.jsonl /mnt/weka/shrd/wm/mingjia/data/h3_causal/game_pilot3k_480p
# train: 2-node smoke, 8-node DF
sbatch --nodes 2 h3_causal/scripts/train.sh h3_causal/configs/smoke_2n.yaml
sbatch --nodes 8 h3_causal/scripts/train.sh h3_causal/configs/df_8n.yaml
# generate held-out clips (one GPU per sample); --mode bidirectional = release baseline
torchrun --nproc_per_node 8 -m h3_causal.inference.generate --weights RUN/exports/step-XXXXXXX-ema \
    --data-dir DATA --out OUT --mode causal --num-samples 16 --save-reference [--extend-chunks 28]
# tests (diffusers parity needs H3_DIFFUSERS_PATH)
H3_DIFFUSERS_PATH=$H3_PYLIBS python -m pytest tests/
```

Storage: ~15 MB per 10 s clip at 480p (bf16 video latents ~5 MB, global + per-event Qwen3-VL
embeddings ~10 MB).
