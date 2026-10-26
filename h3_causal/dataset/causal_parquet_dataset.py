"""Causal-H3 parquet samples -> diffusion-forcing batches (one sample per rank per step).

Rows come from `h3_causal.preprocess.encode` (one row per row group). The
layout, noise levels and noise are all built in the dataloader worker from a
seed derived from (base seed, epoch, sample index), so a resumed run replays
exactly the same corruption.
"""

from __future__ import annotations

import bisect
import glob
import os
import zlib
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import torch

from h3_causal.dataset.causal_packing import build_causal_plan
from h3_causal.train.diffusion_forcing import DFBatch, build_bsmntw_weights, build_df_batch


def decode_tensor(row: dict, prefix: str) -> torch.Tensor:
    return _tensor(row[f"{prefix}_bytes"], row[f"{prefix}_shape"], row[f"{prefix}_dtype"])


def _tensor(raw: bytes, shape, dtype: str) -> torch.Tensor:
    if dtype == "bfloat16":
        return torch.from_numpy(np.frombuffer(raw, dtype=np.int16).copy()).view(torch.bfloat16).view(shape)
    return torch.from_numpy(np.frombuffer(raw, dtype=np.dtype(dtype)).copy()).view(shape)


@dataclass
class ClipTensors:
    clip_id: str
    video_latents: torch.Tensor  # [24, 5C, H, W]
    keyframe_latents: torch.Tensor  # [24, 1, H, W]
    audio_latents: torch.Tensor  # [2, 32, Ta]
    global_text: torch.Tensor  # [Lg, 5120]
    global_text_tags: torch.Tensor  # [Lg]
    captions: list[torch.Tensor]  # unique chunk captions
    caption_index: list[int]  # per chunk
    num_chunks: int
    caption_texts: list[str]
    global_caption: str


def read_clip(row: dict) -> ClipTensors:
    captions = [
        _tensor(raw, shape, row["chunk_caption_dtype"])
        for raw, shape in zip(row["chunk_caption_bytes"], row["chunk_caption_shapes"])
    ]
    return ClipTensors(
        clip_id=row["id"],
        video_latents=decode_tensor(row, "vae_latent"),
        keyframe_latents=decode_tensor(row, "keyframe_latent"),
        audio_latents=decode_tensor(row, "audio_latent"),
        global_text=decode_tensor(row, "text_embedding"),
        global_text_tags=torch.tensor(row["text_token_tags"], dtype=torch.long),
        captions=captions,
        caption_index=list(row["chunk_caption_index"]),
        num_chunks=int(row["num_chunks"]),
        caption_texts=list(row["chunk_caption_texts"]),
        global_caption=row["caption"],
    )


def clip_text_rows(clip: ClipTensors, num_chunks: int, use_chunk_captions: bool) -> tuple[torch.Tensor, list[int]]:
    """Text rows in plan order (global, then each chunk's caption) and per-chunk caption lengths."""
    parts, lengths = [clip.global_text], []
    for chunk in range(num_chunks):
        index = clip.caption_index[chunk] if use_chunk_captions else -1
        if index < 0:
            lengths.append(0)
            continue
        parts.append(clip.captions[index])
        lengths.append(clip.captions[index].shape[0])
    return torch.cat(parts), lengths


class ParquetIndex:
    """(file, row group) addresses over every parquet file under the given directories."""

    def __init__(self, data_dirs: list[str]):
        self.files = sorted(f for d in data_dirs for f in glob.glob(os.path.join(d, "*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"no parquet files under {data_dirs}")
        self.offsets = [0]
        for path in self.files:
            self.offsets.append(self.offsets[-1] + pq.ParquetFile(path).metadata.num_row_groups)

    def __len__(self) -> int:
        return self.offsets[-1]

    def read(self, index: int) -> dict:
        file_index = bisect.bisect_right(self.offsets, index) - 1
        group = index - self.offsets[file_index]
        return pq.ParquetFile(self.files[file_index]).read_row_group(group).to_pylist()[0]


class CausalH3Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dirs: list[str],
        *,
        seed: int,
        num_train_timesteps: int = 1000,
        text_timestep: float = 1.0,
        max_chunks: int | None = None,
        use_chunk_captions: bool = True,
        holdout: int = 0,
    ):
        self.index = ParquetIndex(data_dirs)
        # The last `holdout` rows (by address) are reserved for evaluation.
        self.length = len(self.index) - holdout
        self.seed = seed
        self.num_train_timesteps = num_train_timesteps
        self.text_timestep = text_timestep
        self.max_chunks = max_chunks
        self.use_chunk_captions = use_chunk_captions
        self.bsmntw = build_bsmntw_weights(num_train_timesteps)
        self.epoch = 0

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, key: tuple[int, int]) -> DFBatch:
        epoch, index = key
        clip = read_clip(self.index.read(index))
        num_chunks = clip.num_chunks if self.max_chunks is None else min(clip.num_chunks, self.max_chunks)
        text_rows, caption_lengths = clip_text_rows(clip, num_chunks, self.use_chunk_captions)
        _, _, height, width = clip.video_latents.shape
        plan = build_causal_plan(
            global_text_tags=clip.global_text_tags,
            caption_lengths=caption_lengths,
            latent_height=height,
            latent_width=width,
            num_audio_latents=clip.audio_latents.shape[-1],
        )
        rng = torch.Generator().manual_seed(zlib.crc32(f"{self.seed}:{epoch}:{index}".encode()))
        return build_df_batch(
            plan,
            video_latents=clip.video_latents,
            keyframe_latents=clip.keyframe_latents,
            audio_latents=clip.audio_latents,
            text_rows=text_rows,
            rng=rng,
            num_train_timesteps=self.num_train_timesteps,
            bsmntw_weights=self.bsmntw,
            text_timestep=self.text_timestep,
        )


class ResumableSampler(torch.utils.data.Sampler):
    """Seeded per-epoch permutation, strided over data-parallel ranks; yields (epoch, index).

    Workers prefetch ahead of the trainer, so the resume point is the number of
    samples the trainer actually consumed (`set_consumed`), not this iterator's cursor.
    """

    def __init__(self, length: int, *, rank: int, world_size: int, seed: int):
        self.length, self.rank, self.world_size, self.seed = length, rank, world_size, seed
        self.per_rank = length // world_size
        if self.per_rank == 0:
            raise ValueError(f"{length} samples cannot feed {world_size} ranks")
        self.consumed = 0

    def _order(self, epoch: int) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(self.length, generator=generator).tolist()
        return order[: self.per_rank * self.world_size][self.rank :: self.world_size]

    def set_consumed(self, consumed: int) -> None:
        self.consumed = int(consumed)

    def __iter__(self):
        epoch, position = divmod(self.consumed, self.per_rank)
        while True:
            order = self._order(epoch)
            for index in order[position:]:
                yield epoch, index
            epoch, position = epoch + 1, 0


def collate_single(items: list[DFBatch]) -> DFBatch:
    if len(items) != 1:
        raise ValueError("causal H3 trains one packed sample per rank")
    return items[0]
