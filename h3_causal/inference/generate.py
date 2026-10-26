"""Generate held-out clips with causal (KV-cached) or bidirectional H3 and write mp4 with audio.

    torchrun --nproc_per_node 8 -m h3_causal.inference.generate --weights DIR --data-dir D --out OUT \
        --mode causal|bidirectional [--num-samples 16] [--num-chunks 13] [--extend-chunks 0]

`bidirectional` reproduces the release FL2VA sampler on the same packed rows (global caption only, text
following the video timestep, no mask): the Gate-A baseline at a given resolution. `causal` runs the
chunk-wise sampler; `--extend-chunks` rolls out past the clip length (captions repeat the last chunk's,
positions recycle past `--recycle-after`).
"""

from __future__ import annotations

import argparse
import json
import os

import av
import numpy as np
import torch

from h3_causal.dataset import geometry as geo
from h3_causal.dataset.causal_packing import ROLE_ANCHOR, ROLE_AUDIO, ROLE_VIDEO, build_causal_plan
from h3_causal.dataset.causal_parquet_dataset import ParquetIndex, clip_text_rows, read_clip
from h3_causal.inference.sampler import SamplerConfig, assemble_latents, release_sigmas, sample_causal
from h3_causal.models.loading import build_model
from h3_causal.train.diffusion_forcing import AUDIO_SHIFT, VIDEO_SHIFT

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


@torch.no_grad()
def sample_bidirectional(model, plan, *, keyframe_latents, text_rows, num_inference_steps, generator, device, dtype):
    """Release FL2VA sampling on the packed rows: one shared t, text follows video t, anchor at max(t, 0.999)."""
    anchor = geo.patchify_video_latents(keyframe_latents.float())
    anchor_noise = torch.randn(anchor.shape, generator=generator, dtype=torch.float32)
    anchor = geo.KEYFRAME_NOISE_AUG * anchor + (1.0 - geo.KEYFRAME_NOISE_AUG) * anchor_noise
    num_video = plan.video_indices.numel() - plan.num_anchor_rows
    video = torch.randn((num_video, model.proj_in.in_features), generator=generator, dtype=torch.float32).to(device)
    audio = torch.randn((plan.audio_indices.numel(), model.config.audio_in_channels), generator=generator).to(device)
    anchor = anchor.to(device)
    sigmas_v = release_sigmas(num_inference_steps, VIDEO_SHIFT)
    sigmas_a = release_sigmas(num_inference_steps, AUDIO_SHIFT)
    role = plan.row_role
    plan_device = plan.to(device)
    for step in range(sigmas_v.numel() - 1):
        sv, sv_next, sa, sa_next = (float(x) for x in (sigmas_v[step], sigmas_v[step + 1], sigmas_a[step], sigmas_a[step + 1]))
        row_t = torch.full((plan.sequence_length,), 1.0 - sv)
        row_t[role == ROLE_AUDIO] = 1.0 - sa
        row_t[role == ROLE_ANCHOR] = max(1.0 - sv, geo.KEYFRAME_NOISE_AUG)
        timestep, timestep_indices = torch.unique(row_t, sorted=True, return_inverse=True)
        v_video, v_audio = model(
            hidden_states=torch.cat([anchor, video]).to(dtype)[None],
            audio_hidden_states=audio.to(dtype)[None],
            encoder_hidden_states=text_rows.to(device, dtype)[None],
            timestep=timestep.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=plan_device.token_tags,
            position_ids=plan_device.position_ids,
            video_indices=plan_device.video_indices,
            audio_indices=plan_device.audio_indices,
            text_indices=plan_device.text_indices,
            text_segment_ids=plan_device.text_segment_ids,
        )
        video_velocity = v_video[0, plan.num_anchor_rows :].float()
        video = (sv_next / sv) * video + (1 - sv_next / sv) * (video + sv * video_velocity)
        audio = (sa_next / sa) * audio + (1 - sa_next / sa) * (audio + sa * v_audio[0].float())
    return video.cpu(), audio.cpu()


class Decoder:
    def __init__(self, h3_root: str, device):
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        self.device = device
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(f"{h3_root}/vae").to(device).eval()
        self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(f"{h3_root}/audio_vae").to(device).eval()

    @torch.no_grad()
    def __call__(self, video_latents: torch.Tensor, audio_latents: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        d = self.device
        mean = torch.tensor(self.vae.config.latents_mean, device=d).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=d).view(1, -1, 1, 1, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            video = self.vae.decode(video_latents[None].to(d) * std + mean).sample
        pm = torch.tensor(PIXEL_MEAN, device=d).view(1, -1, 1, 1, 1)
        ps = torch.tensor(PIXEL_STD, device=d).view(1, -1, 1, 1, 1)
        video = ((video.float() * ps + pm).clamp(0, 1) * 255).round().byte()[0].permute(1, 2, 3, 0).cpu().numpy()
        amean = torch.tensor(self.audio_vae.config.latents_mean, device=d).view(1, -1, 1)
        astd = torch.tensor(self.audio_vae.config.latents_std, device=d).view(1, -1, 1)
        audio = self.audio_vae.decode(audio_latents.to(d) * astd + amean).sample.float()[:, 0].cpu().numpy()
        return video, audio


def write_mp4(path: str, video: np.ndarray, audio: np.ndarray | None) -> None:
    with av.open(path, "w") as out:
        stream = out.add_stream("libx264", rate=geo.FPS)
        stream.width, stream.height, stream.pix_fmt = video.shape[2], video.shape[1], "yuv420p"
        stream.options = {"crf": "18"}
        audio_stream = None
        if audio is not None:
            audio_stream = out.add_stream("aac", rate=geo.AUDIO_SAMPLE_RATE)
            audio_stream.layout = "stereo"
        for frame in video:
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        if audio_stream is not None:
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(audio.astype(np.float32)), format="fltp", layout="stereo")
            frame.sample_rate = geo.AUDIO_SAMPLE_RATE
            for packet in audio_stream.encode(frame):
                out.mux(packet)
            for packet in audio_stream.encode():
                out.mux(packet)


def extend_clip_captions(clip, num_chunks: int):
    """Past the clip, every new chunk reuses the last chunk's caption (rollout beyond the data)."""
    index = list(clip.caption_index[: clip.num_chunks])
    while len(index) < num_chunks:
        index.append(index[-1])
    clip.caption_index = index
    return clip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="transformer dir (release or exported)")
    parser.add_argument("--h3-root", default="/mnt/weka/shrd/wm/areeb/H3")
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--holdout", type=int, default=64, help="samples are taken from the last `holdout` rows")
    parser.add_argument("--num-chunks", type=int, default=0, help="0 = the clip's chunks")
    parser.add_argument("--extend-chunks", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--window-chunks", type=int, default=8)
    parser.add_argument("--caption-window-chunks", type=int, default=1)
    parser.add_argument("--text-timestep", type=float, default=1.0)
    parser.add_argument("--commit-timestep", type=float, default=1.0)
    parser.add_argument("--recycle-after", type=int, default=-1, help="-1 = trained chunk count - 1")
    parser.add_argument("--no-chunk-captions", action="store_true")
    parser.add_argument("--save-reference", action="store_true", help="also decode the ground-truth latents")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    os.makedirs(args.out, exist_ok=True)

    index = ParquetIndex(args.data_dir)
    rows = list(range(max(0, len(index) - args.holdout), len(index)))[: args.num_samples]
    model = build_model(args.weights, device=device)
    decoder = Decoder(args.h3_root, device)

    for row_index in rows[rank::world]:
        clip = read_clip(index.read(row_index))
        num_chunks = args.num_chunks or clip.num_chunks
        if args.mode == "causal":
            num_chunks += args.extend_chunks
            clip = extend_clip_captions(clip, num_chunks)
        use_captions = args.mode == "causal" and not args.no_chunk_captions
        text_rows, caption_lengths = clip_text_rows(clip, num_chunks, use_captions)
        _, _, height, width = clip.video_latents.shape
        num_audio = geo.chunk_audio_range(num_chunks - 1)[1]
        plan = build_causal_plan(
            global_text_tags=clip.global_text_tags,
            caption_lengths=caption_lengths,
            latent_height=height,
            latent_width=width,
            num_audio_latents=num_audio,
        )
        generator = torch.Generator().manual_seed(args.seed + row_index)
        if args.mode == "causal":
            recycle = args.recycle_after if args.recycle_after >= 0 else clip.num_chunks - 1
            config = SamplerConfig(
                num_inference_steps=args.steps,
                window_chunks=args.window_chunks,
                caption_window_chunks=args.caption_window_chunks,
                text_timestep=args.text_timestep,
                commit_timestep=args.commit_timestep,
                recycle_after_chunk=recycle,
            )
            output = sample_causal(
                model, plan, keyframe_latents=clip.keyframe_latents, text_rows=text_rows,
                config=config, generator=generator, device=device,
            )
            video_latents, audio_latents = assemble_latents(plan, output, num_audio)
        else:
            video_rows, audio_rows = sample_bidirectional(
                model, plan, keyframe_latents=clip.keyframe_latents, text_rows=text_rows,
                num_inference_steps=args.steps, generator=generator, device=device, dtype=torch.bfloat16,
            )
            from h3_causal.inference.sampler import SampleOutput

            video_latents, audio_latents = assemble_latents(plan, SampleOutput(video_rows, audio_rows), num_audio)

        video, audio = decoder(video_latents, audio_latents)
        name = f"{row_index:06d}_{clip.clip_id}"
        write_mp4(os.path.join(args.out, f"{name}_{args.mode}.mp4"), video, audio)
        if args.save_reference:
            reference_chunks = min(num_chunks, clip.num_chunks)
            frames = reference_chunks * geo.LATENTS_PER_CHUNK
            ref_audio = clip.audio_latents[:, :, : geo.chunk_audio_range(reference_chunks - 1)[1]].float()
            ref_video, ref_wave = decoder(clip.video_latents[:, :frames].float(), ref_audio)
            write_mp4(os.path.join(args.out, f"{name}_reference.mp4"), ref_video, ref_wave)
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(
                {"clip": clip.clip_id, "mode": args.mode, "num_chunks": num_chunks, "global_caption": clip.global_caption,
                 "chunk_captions": [clip.caption_texts[i] if i >= 0 else "" for i in clip.caption_index[:num_chunks]],
                 "args": vars(args)},
                f, indent=1,
            )
        print(f"[rank {rank}] wrote {name}", flush=True)


if __name__ == "__main__":
    main()
