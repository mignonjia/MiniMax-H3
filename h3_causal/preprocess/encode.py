"""Encode a clip manifest into causal-H3 parquet shards (one GPU per rank).

Per clip: 24 fps frames on an aspect-preserving canvas at the requested short
edge, `17n + 5` frames -> H3 video latents (first `5n` kept), the first frame
as the keyframe anchor (release recipe), 32 kHz stereo audio latents, the
Qwen3-VL layer-50 global caption (with the keyframe's vision block, as FL2VA
presents it) and the deduplicated per-chunk captions.

The row schema extends FastVideo's `pyarrow_schema_t2va` (same `*_bytes/_shape/_dtype`
triplets) with keyframe, caption-chunk and token-tag columns. Large tensors are bf16.

    torchrun --nproc_per_node 8 -m h3_causal.preprocess.encode --manifest m.jsonl --out DIR --h3-root /mnt/weka/shrd/wm/areeb/H3
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import zlib

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from h3_causal.dataset import geometry as geo
from h3_causal.preprocess.captions import Event, build_clip_captions

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
TEXT_ENCODER_LAYER = 50
TEXT_TAG, VIDEO_TAG = 1, 0


def tensor_columns(prefix: str, tensor: torch.Tensor) -> dict:
    tensor = tensor.contiguous().cpu()
    dtype = str(tensor.dtype).removeprefix("torch.")
    raw = tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor
    return {
        f"{prefix}_bytes": raw.numpy().tobytes(),
        f"{prefix}_shape": list(tensor.shape),
        f"{prefix}_dtype": dtype,
    }


SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("source", pa.string()),
        *[
            f
            for prefix in ("vae_latent", "audio_latent", "text_embedding", "keyframe_latent")
            for f in (
                pa.field(f"{prefix}_bytes", pa.binary()),
                pa.field(f"{prefix}_shape", pa.list_(pa.int64())),
                pa.field(f"{prefix}_dtype", pa.string()),
            )
        ],
        pa.field("text_token_tags", pa.list_(pa.int8())),
        pa.field("chunk_caption_bytes", pa.list_(pa.binary())),
        pa.field("chunk_caption_shapes", pa.list_(pa.list_(pa.int64()))),
        pa.field("chunk_caption_dtype", pa.string()),
        pa.field("chunk_caption_texts", pa.list_(pa.string())),
        pa.field("chunk_caption_index", pa.list_(pa.int64())),
        pa.field("num_chunks", pa.int64()),
        pa.field("file_name", pa.string()),
        pa.field("caption", pa.string()),
        pa.field("media_type", pa.string()),
        pa.field("width", pa.int64()),
        pa.field("height", pa.int64()),
        pa.field("num_frames", pa.int64()),
        pa.field("duration_sec", pa.float64()),
        pa.field("fps", pa.float64()),
        pa.field("audio_sample_rate", pa.int64()),
    ]
)


# --------------------------------------------------------------------------- decoding


def decode_clip(path: str, start_sec: float, num_frames: int) -> tuple[np.ndarray, np.ndarray | None]:
    """`num_frames` frames resampled to 24 fps from `start_sec` ([T, H, W, 3] uint8) and the 32 kHz stereo audio.

    Video: for each 24 fps instant, the decoded frame with the nearest timestamp.
    """
    duration = num_frames / geo.FPS
    wanted = [start_sec + i / geo.FPS for i in range(num_frames)]
    with av.open(path) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0.0, start_sec - 1.0) / stream.time_base), stream=stream, backward=True)
        frames, cursor, previous = [], 0, None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            current = (frame.time, frame.to_ndarray(format="rgb24"))
            while cursor < num_frames and wanted[cursor] <= current[0]:
                if previous is not None and wanted[cursor] - previous[0] < current[0] - wanted[cursor]:
                    frames.append(previous[1])
                else:
                    frames.append(current[1])
                cursor += 1
            previous = current
            if cursor >= num_frames:
                break
        if len(frames) < num_frames:
            raise ValueError(f"{path}: {len(frames)} of {num_frames} frames from {start_sec:.2f}s")
        video = np.stack(frames)

    audio = None
    with av.open(path) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=geo.AUDIO_SAMPLE_RATE)
            container.seek(int(max(0.0, start_sec - 1.0) / stream.time_base), stream=stream, backward=True)
            chunks, first_time = [], None
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                if first_time is None:
                    first_time = float(frame.time)
                chunks.extend(out.to_ndarray() for out in resampler.resample(frame))
                if frame.time > start_sec + duration + 0.5:
                    break
            if chunks:
                samples = np.concatenate(chunks, axis=1)
                offset = int(round((start_sec - first_time) * geo.AUDIO_SAMPLE_RATE))
                length = int(round(duration * geo.AUDIO_SAMPLE_RATE))
                if offset >= 0 and samples.shape[1] >= offset + length:
                    audio = np.ascontiguousarray(samples[:, offset : offset + length], dtype=np.float32)
    return video, audio


def to_canvas(video: np.ndarray, canvas: geo.Canvas) -> torch.Tensor:
    """Resize the short side and centre-crop to the canvas; [T, H, W, 3] uint8 -> [1, 3, T, h, w] uint8."""
    frames = torch.from_numpy(video).permute(0, 3, 1, 2).float()
    _, _, height, width = frames.shape
    scale = max(canvas.height / height, canvas.width / width)
    size = (max(canvas.height, round(height * scale)), max(canvas.width, round(width * scale)))
    frames = F.interpolate(frames, size=size, mode="bicubic", antialias=True, align_corners=False)
    top, left = (size[0] - canvas.height) // 2, (size[1] - canvas.width) // 2
    frames = frames[:, :, top : top + canvas.height, left : left + canvas.width]
    return frames.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)[None]


# --------------------------------------------------------------------------- models


class Encoders:
    def __init__(self, h3_root: str, device: torch.device):
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        self.device = device
        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(f"{h3_root}/vae").to(device).eval()
        self.audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(f"{h3_root}/audio_vae").to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(f"{h3_root}/tokenizer")
        self.processor = AutoProcessor.from_pretrained(f"{h3_root}/processor")
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(f"{h3_root}/text_encoder", dtype=torch.bfloat16)
        # hidden_states[50] is the un-normalized output of decoder layer 50; layers past 51 never feed it.
        layers = text_encoder.model.language_model.layers
        del layers[TEXT_ENCODER_LAYER + 1 :]
        self.text_encoder = text_encoder.to(device).eval()
        self.vae_mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        self.vae_std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        self.audio_mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        self.audio_std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)

    def _pixels(self, uint8: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(PIXEL_MEAN, device=self.device).view(1, -1, 1, 1, 1)
        std = torch.tensor(PIXEL_STD, device=self.device).view(1, -1, 1, 1, 1)
        return (uint8.to(self.device, torch.float32).div(255.0) - mean) / std

    @torch.no_grad()
    def video(self, uint8: torch.Tensor, seed: int) -> torch.Tensor:
        posterior = self.vae.encode(self._pixels(uint8)).latent_dist
        latents = posterior.sample(generator=torch.Generator(self.device).manual_seed(seed))
        return ((latents - self.vae_mean) / self.vae_std)[0]

    @torch.no_grad()
    def keyframe(self, uint8: torch.Tensor) -> torch.Tensor:
        """Release keyframe recipe: single-frame spatial encode, posterior sample at seed 42, fp16 rounding."""
        posterior = self.vae.encode(self._pixels(uint8)).latent_dist
        latents = posterior.sample(generator=torch.Generator().manual_seed(42)).to(torch.float16).float()
        return ((latents - self.vae_mean) / self.vae_std)[0]

    @torch.no_grad()
    def audio(self, waveform: np.ndarray) -> torch.Tensor:
        wave = torch.from_numpy(waveform).to(self.device)[:, None]
        latents = self.audio_vae.encode(wave).latent_dist.mode()
        return (latents - self.audio_mean) / self.audio_std

    @torch.no_grad()
    def text(self, prompt: str, keyframe: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """FL2VA presentation: optional `<Picture 1>: ` + vision block, then the prompt verbatim; no template."""
        token_ids, tags, vision = [], [], {}
        if keyframe is not None:
            features = self.processor.image_processor(images=[keyframe], return_tensors="pt")
            grid = features["image_grid_thw"]
            vision = {"pixel_values": features["pixel_values"].to(self.device, torch.bfloat16), "image_grid_thw": grid.to(self.device)}
            num_image_tokens = int(grid[0].prod()) // self.processor.image_processor.merge_size**2
            label = self.tokenizer("<Picture 1>: ", add_special_tokens=False)["input_ids"]
            block = (
                [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label + block
            tags += [TEXT_TAG] * len(label) + [VIDEO_TAG] * len(block)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        tags += [TEXT_TAG] * len(prompt_ids)
        input_ids = torch.tensor([token_ids], device=self.device)
        mm_token_type_ids = torch.tensor(self.processor.create_mm_token_type_ids([token_ids]), device=self.device)
        outputs = self.text_encoder.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_token_type_ids,
            use_cache=False,
            output_hidden_states=True,
            **vision,
        )
        return outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.bfloat16), torch.tensor(tags, dtype=torch.int8)


# --------------------------------------------------------------------------- per clip


def encode_clip(encoders: Encoders, clip: dict, args) -> dict:
    available = int(clip["max_duration_sec"] * geo.FPS)
    num_chunks = min(geo.max_chunks_for_frames(available), args.max_chunks)
    if num_chunks < args.min_chunks:
        raise ValueError(f"only {num_chunks} chunks in {clip['max_duration_sec']:.2f}s")
    num_frames = geo.aligned_num_frames(num_chunks)
    video, audio = decode_clip(clip["video_path"], clip["start_sec"], num_frames)
    if audio is None and not args.allow_silent:
        raise ValueError("no audio stream")
    if audio is None:
        audio = np.zeros((2, int(round(num_frames / geo.FPS * geo.AUDIO_SAMPLE_RATE))), dtype=np.float32)

    canvas = geo.resolve_canvas(video.shape[2], video.shape[1], args.short_edge, args.max_pixels)
    pixels = to_canvas(video, canvas)
    seed = zlib.crc32(clip["id"].encode())
    latents = encoders.video(pixels, seed)[:, : num_chunks * geo.LATENTS_PER_CHUNK]
    keyframe = encoders.keyframe(pixels[:, :, :1])
    audio_latents = encoders.audio(audio)
    needed = geo.chunk_audio_range(num_chunks - 1)[1]
    if audio_latents.shape[-1] < needed:
        raise ValueError(f"{audio_latents.shape[-1]} audio latents < {needed}")
    audio_latents = audio_latents[:, :, :needed]

    events = [Event(start, end, fields) for start, end, fields in clip["events"]]
    captions = build_clip_captions(clip["caption"], events, num_chunks)
    keyframe_image = pixels[0, :, 0].permute(1, 2, 0).cpu().numpy() if args.keyframe_in_text else None
    global_embedding, tags = encoders.text(captions.global_text, keyframe_image)
    if global_embedding.shape[0] > args.max_global_tokens:
        raise ValueError(f"global caption {global_embedding.shape[0]} > {args.max_global_tokens} tokens")
    chunk_embeddings = []
    for text in captions.chunk_texts:
        embedding, _ = encoders.text(text)
        if embedding.shape[0] > args.max_caption_tokens:
            raise ValueError(f"chunk caption {embedding.shape[0]} > {args.max_caption_tokens} tokens")
        chunk_embeddings.append(embedding.cpu())

    row = {"id": clip["id"], "source": clip["source"]}
    row |= tensor_columns("vae_latent", latents.to(torch.bfloat16))
    row |= tensor_columns("audio_latent", audio_latents.float())
    row |= tensor_columns("text_embedding", global_embedding)
    row |= tensor_columns("keyframe_latent", keyframe.float())
    row |= {
        "text_token_tags": tags.tolist(),
        "chunk_caption_bytes": [e.view(torch.int16).numpy().tobytes() for e in chunk_embeddings],
        "chunk_caption_shapes": [list(e.shape) for e in chunk_embeddings],
        "chunk_caption_dtype": "bfloat16",
        "chunk_caption_texts": captions.chunk_texts,
        "chunk_caption_index": captions.chunk_index,
        "num_chunks": num_chunks,
        "file_name": clip["video_path"],
        "caption": captions.global_text,
        "media_type": "video",
        "width": canvas.width,
        "height": canvas.height,
        "num_frames": num_frames,
        "duration_sec": num_frames / geo.FPS,
        "fps": float(geo.FPS),
        "audio_sample_rate": geo.AUDIO_SAMPLE_RATE,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--h3-root", default="/mnt/weka/shrd/wm/areeb/H3")
    parser.add_argument("--short-edge", type=int, default=480)
    parser.add_argument("--max-pixels", type=int, default=480 * 864)
    parser.add_argument("--max-chunks", type=int, default=14)
    parser.add_argument("--min-chunks", type=int, default=7)
    parser.add_argument("--max-global-tokens", type=int, default=1536)
    parser.add_argument("--max-caption-tokens", type=int, default=384)
    parser.add_argument("--rows-per-file", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-silent", action="store_true")
    parser.add_argument("--no-keyframe-in-text", dest="keyframe_in_text", action="store_false")
    args = parser.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    os.makedirs(args.out, exist_ok=True)

    with open(args.manifest) as f:
        clips = [json.loads(line) for line in f]
    if args.limit:
        clips = clips[: args.limit]
    clips = clips[rank::world]
    done_path = f"{args.out}/_done.rank{rank:04d}.txt"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    clips = [c for c in clips if c["id"] not in done]
    print(f"[rank {rank}] {len(clips)} clips to encode ({len(done)} already done)", flush=True)

    encoders = Encoders(args.h3_root, device)
    existing = [n for n in os.listdir(args.out) if n.startswith(f"rank{rank:04d}-")]
    file_index = len(existing)
    buffer, buffer_ids, failures = [], [], 0
    started = time.time()

    def flush():
        nonlocal buffer, buffer_ids, file_index
        if not buffer:
            return
        path = f"{args.out}/rank{rank:04d}-{file_index:05d}.parquet"
        # One row per row group: the loader reads single samples.
        pq.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA), path + ".tmp", row_group_size=1)
        os.replace(path + ".tmp", path)
        with open(done_path, "a") as f:
            f.write("\n".join(buffer_ids) + "\n")
        file_index += 1
        buffer, buffer_ids = [], []

    for index, clip in enumerate(clips):
        try:
            buffer.append(encode_clip(encoders, clip, args))
            buffer_ids.append(clip["id"])
        except Exception as error:  # a bad clip must not stop the shard
            failures += 1
            with open(f"{args.out}/_failed.rank{rank:04d}.tsv", "a") as f:
                f.write(f"{clip['id']}\t{type(error).__name__}: {error}\n")
            if failures <= 3:
                traceback.print_exc()
        if len(buffer) >= args.rows_per_file:
            flush()
        if index % 20 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"[rank {rank}] {index + 1}/{len(clips)} clips, {failures} failed, {rate:.2f} clips/s", flush=True)
    flush()
    print(f"[rank {rank}] finished: {failures} failures", flush=True)


if __name__ == "__main__":
    main()
