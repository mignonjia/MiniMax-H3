"""Per-chunk captions for clips that only carry one whole-clip prompt (e.g. H3-generated sets).

A Qwen3-VL instruct model sees the clip's source prompt and two frames per causal chunk and
rewrites them into the own-caption layout: an H3-register global caption that only states what
holds from the first frame on (no later events), and one self-contained caption per chunk
(visual, camera, sound). Sound comes from the prompt: the captioner does not hear the audio.

    torchrun --nproc_per_node 8 -m h3_causal.preprocess.chunk_captioner --manifest IN.jsonl --out OUT_DIR

Each rank appends finished records to OUT_DIR/captions.rank*.jsonl (restartable); merge with
`--merge OUT.jsonl`. Records become manifest lines with `caption.global_text` and per-chunk events.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import traceback

import numpy as np
import torch
from PIL import Image

from h3_causal.dataset import geometry as geo
from h3_causal.preprocess.encode import decode_clip

DEFAULT_MODEL = "/mnt/weka/shrd/wm/.cache/huggingface/hub/models--Qwen--Qwen3-VL-32B-Instruct/snapshots/0cfaf48183f594c314753d30a4c4974bc75f3ccb"
FRAMES_PER_CHUNK = 2
FRAME_LONG_EDGE = 448

INSTRUCTION = """You are annotating a {duration:.1f}-second AI-generated video for training a streaming video-and-audio generator.
Below is the exact prompt that produced the video (it describes both picture and sound), followed by {num_frames} frames: \
{per_chunk} per segment, {num_chunks} consecutive segments of {chunk_sec:.2f} s each.

PROMPT:
{prompt}

Return JSON with two keys.
"global": what holds for the whole clip and its opening state, never what happens later, as three fields:
  "integrated_multimodal_description": style and medium, opening framing, setting, lighting, each main subject's appearance, and what the first frame shows. No later actions, no later shots, no timestamps.
  "overall_soundscape": the persistent ambient sound bed only, no one-off sound events.
  "non_diegetic_music": the music's style and instrumentation from the prompt, without how it develops over time ("N/A" if none).
"segments": exactly {num_chunks} objects in order, each {{"visual": what is visible and happening in this segment (1-2 sentences, present tense), \
"camera": the camera motion in this segment ("static" if none), "sound": what is heard in this segment, from the prompt's sound descriptions and the action}}.
Every segment must read on its own: name subjects by description instead of pronouns, and never mention segment numbers, times, or other segments. \
If a shot cut happens inside a segment, describe the new shot. Output only the JSON."""


def chunk_frame_indices(num_chunks: int) -> list[list[int]]:
    """Frames at 1/4 and 3/4 of each chunk's 17-frame span."""
    out = []
    for chunk in range(num_chunks):
        start = chunk * geo.VAE_CLIP_FRAMES
        out.append([start + round((k + 0.5) * geo.VAE_CLIP_FRAMES / FRAMES_PER_CHUNK) for k in range(FRAMES_PER_CHUNK)])
    return out


def _resize(frame: np.ndarray) -> Image.Image:
    height, width = frame.shape[:2]
    scale = FRAME_LONG_EDGE / max(height, width)
    size = (max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32))
    return Image.fromarray(frame).resize(size, Image.BICUBIC)


def build_conversation(clip: dict, num_chunks: int) -> list[dict]:
    frames, _ = decode_clip(clip["video_path"], clip["start_sec"], geo.aligned_num_frames(num_chunks))
    content = [{
        "type": "text",
        "text": INSTRUCTION.format(
            duration=geo.aligned_num_frames(num_chunks) / geo.FPS, num_frames=num_chunks * FRAMES_PER_CHUNK,
            per_chunk=FRAMES_PER_CHUNK, num_chunks=num_chunks, chunk_sec=geo.VAE_CLIP_FRAMES / geo.FPS,
            prompt=clip["caption"]["h3_prompt"],
        ),
    }]
    for chunk, indices in enumerate(chunk_frame_indices(num_chunks)):
        content.append({"type": "text", "text": f"\nSegment {chunk + 1}:"})
        content.extend({"type": "image", "image": _resize(frames[i])} for i in indices)
    return [{"role": "user", "content": content}]


def parse_response(text: str, num_chunks: int) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match is None:
        raise ValueError("no JSON object in response")
    data = json.loads(match.group(0))
    head = data["global"]
    segments = data["segments"]
    if len(segments) != num_chunks:
        raise ValueError(f"{len(segments)} segments, expected {num_chunks}")
    global_text = (
        f"integrated_multimodal_description: {head['integrated_multimodal_description'].strip()}\n\n"
        f"overall_soundscape: {head['overall_soundscape'].strip()}\n\n"
        f"non_diegetic_music: {str(head.get('non_diegetic_music') or 'N/A').strip()}"
    )
    events = []
    for chunk, segment in enumerate(segments):
        start, end = geo.chunk_time_range_sec(chunk)
        fields = {"event_description": segment["visual"].strip()}
        for key, name in (("camera", "camera_motion"), ("sound", "sound")):
            if str(segment.get(key) or "").strip():
                fields[name] = str(segment[key]).strip()
        events.append([start, end, fields])
    return {"global_text": global_text, "events": events}


def caption_records(clips: list[dict], args) -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    rank = int(os.environ.get("RANK", 0))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).to(device).eval()

    out_path = f"{args.out}/captions.rank{rank:04d}.jsonl"
    failed_path = f"{args.out}/_failed.rank{rank:04d}.tsv"
    started, finished, failures = time.time(), 0, 0
    for begin in range(0, len(clips), args.batch_size):
        batch, conversations = [], []
        for clip in clips[begin : begin + args.batch_size]:
            num_chunks = min(geo.max_chunks_for_frames(int(clip["max_duration_sec"] * geo.FPS + 1e-6)), args.max_chunks)
            try:
                conversations.append(build_conversation(clip, num_chunks))
                batch.append((clip, num_chunks))
            except Exception as error:
                failures += 1
                with open(failed_path, "a") as f:
                    f.write(f"{clip['id']}\tdecode {type(error).__name__}: {error}\n")
        if not batch:
            continue
        inputs = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(device)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        texts = processor.batch_decode(generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        lines = []
        for (clip, num_chunks), text in zip(batch, texts):
            try:
                parsed = parse_response(text, num_chunks)
            except Exception as error:
                failures += 1
                with open(failed_path, "a") as f:
                    f.write(f"{clip['id']}\tparse {type(error).__name__}: {error}\t{json.dumps(text)}\n")
                continue
            record = dict(clip)
            record["caption"] = {**clip["caption"], "global_text": parsed["global_text"]}
            record["events"] = parsed["events"]
            lines.append(json.dumps(record))
        if lines:
            with open(out_path, "a") as f:
                f.write("\n".join(lines) + "\n")
        finished += len(batch)
        rate = finished / (time.time() - started)
        print(f"[rank {rank}] {finished}/{len(clips)} clips, {failures} failed, {rate:.2f} clips/s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-chunks", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--merge", help="write the merged manifest of all ranks' records here and exit")
    args = parser.parse_args()

    if args.merge:
        seen, count = set(), 0
        with open(args.merge, "w") as out:
            for path in sorted(glob.glob(f"{args.out}/captions.rank*.jsonl")):
                for line in open(path):
                    record = json.loads(line)
                    if record["id"] not in seen:
                        seen.add(record["id"])
                        out.write(line if line.endswith("\n") else line + "\n")
                        count += 1
        print(f"wrote {count} clips to {args.merge}")
        return

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    os.makedirs(args.out, exist_ok=True)
    with open(args.manifest) as f:
        clips = [json.loads(line) for line in f]
    if args.limit:
        clips = clips[: args.limit]
    clips = clips[rank::world]
    done = set()
    for path in glob.glob(f"{args.out}/captions.rank*.jsonl"):
        done.update(json.loads(line)["id"] for line in open(path))
    clips = [c for c in clips if c["id"] not in done]
    print(f"[rank {rank}] {len(clips)} clips to caption ({len(done)} already done)", flush=True)
    try:
        caption_records(clips, args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
