"""Clip manifests (JSONL) for H3 encoding.

One line per clip:
    {"id", "source", "video_path", "start_sec", "max_duration_sec", "caption": {...}, "events": [[start, end, {...}]]}
`caption` holds the structured global fields; event times are seconds from the clip start.

    python -m h3_causal.preprocess.manifest loki --csv-glob '.../captions-*.csv' --local-root /data/loki --out loki.jsonl
    python -m h3_causal.preprocess.manifest game --num-clips 2000 --out game.jsonl
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys

from h3_causal.preprocess.captions import game_v7_events, loki_events

csv.field_size_limit(sys.maxsize)

GAME_EXPORT = "/mnt/weka/shrd/wm/yiyan/Game-WM-Export"
GAME_V7_STORE = f"{GAME_EXPORT}/outputs/AB-V1_768x448_wan22_raw/causal_v7_sekai2546h_minus96"


def _record(clip_id, source, video_path, start_sec, max_duration_sec, caption, events) -> dict:
    caption = {k: v for k, v in caption.items() if k != "events"}
    return {
        "id": clip_id,
        "source": source,
        "video_path": video_path,
        "start_sec": float(start_sec),
        "max_duration_sec": float(max_duration_sec),
        "caption": caption,
        "events": [[e.start, e.end, e.fields] for e in events],
    }


def loki_manifest(csv_glob: str, local_root: str, s3_prefix: str = "s3://") -> list[dict]:
    """Loki joint-training CSVs: path (s3 mp4), start_sec, end_sec, caption JSON with timed events."""
    records = []
    for path in sorted(glob.glob(csv_glob)):
        for row in csv.DictReader(open(path)):
            caption = json.loads(row["caption"])
            local = row["path"].replace(s3_prefix, local_root.rstrip("/") + "/", 1)
            clip_id = row["path"].rsplit("/", 1)[-1].rsplit(".", 1)[0] + f"_seg{row['segment_index']}"
            start, end = float(row["start_sec"]), float(row["end_sec"])
            records.append(_record(clip_id, "loki", local, start, end - start, caption, loki_events(caption)))
    return records


def game_manifest(num_clips: int, seed: int = 0, sources: tuple[str, ...] = ("game",)) -> list[dict]:
    """Raw-video game windows with v7 captions (HUD intact, 10 s, per-second events)."""
    rng = random.Random(seed)
    parts = sorted(glob.glob(f"{GAME_V7_STORE}/part-*.csv"))
    rng.shuffle(parts)
    rows = {}
    for part in parts:
        for row in csv.DictReader(open(part)):
            stem = row["path"].rsplit("/", 1)[-1].removesuffix("_index_0.safetensors")
            rows[stem] = row
        if len(rows) >= num_clips * 3:
            break

    manifests = f"{GAME_EXPORT}/manifests"
    wanted = {}
    with open(f"{manifests}/ab_v1_raw_windows.csv") as f:
        for row in csv.DictReader(f):
            if row["stem"] in rows and row["source"] in sources:
                wanted[int(row["global_index"])] = row["stem"]
    fps = {}
    with open(f"{manifests}/ab_v1_raw_parent_meta.tsv") as f:
        for line in f:
            mp4, rate, _ = line.rstrip("\n").split("\t")
            fps[mp4] = float(rate)

    records = []
    chosen = set(rng.sample(sorted(wanted), min(num_clips, len(wanted))))
    with open(f"{manifests}/ab_v1_raw_plans.jsonl") as f:
        for index, line in enumerate(f):
            if index not in chosen:
                continue
            plan = json.loads(line)
            if plan.get("kind") != "frame_runs" or len(plan["frame_runs"]) != 1:
                continue
            run = plan["frame_runs"][0]
            native_fps = fps.get(run["mp4"])
            if not native_fps:
                continue
            row = rows[wanted[index]]
            caption = json.loads(row["global_caption"])
            events = game_v7_events(json.loads(row["chunk_captions"]))
            # 161 frames at 16 fps: the window spans [first, first + 10 s].
            start = run["frames"][0] / native_fps
            records.append(_record(wanted[index], "game", run["mp4"], start, 10.0, caption, events))
            if len(records) == len(chosen):
                break
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    loki = sub.add_parser("loki")
    loki.add_argument("--csv-glob", required=True)
    loki.add_argument("--local-root", required=True)
    game = sub.add_parser("game")
    game.add_argument("--num-clips", type=int, default=2000)
    game.add_argument("--seed", type=int, default=0)
    for p in (loki, game):
        p.add_argument("--out", required=True)
    args = parser.parse_args()
    records = loki_manifest(args.csv_glob, args.local_root) if args.kind == "loki" else game_manifest(args.num_clips, args.seed)
    with open(args.out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"wrote {len(records)} clips to {args.out}")


if __name__ == "__main__":
    main()
