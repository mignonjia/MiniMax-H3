"""Diffusion-forcing fine-tune of MiniMax-H3 into a chunk-causal model (FSDP2 / HSDP).

    torchrun ... -m h3_causal.train.trainer --config h3_causal/configs/df_8n.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from h3_causal.dataset.causal_packing import build_block_mask
from h3_causal.dataset.causal_parquet_dataset import CausalH3Dataset, ResumableSampler, collate_single
from h3_causal.models.loading import export_weights, load_weights, read_config
from h3_causal.models.transformer import FP32_MODULES, H3Transformer
from h3_causal.train.config import TrainConfig, load_config
from h3_causal.train.diffusion_forcing import DFBatch, df_loss


def log(message: str) -> None:
    if dist.get_rank() == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def build_sharded_model(config: TrainConfig, mesh) -> H3Transformer:
    weights_dir = config.init_from or os.path.join(config.h3_root, "transformer")
    model_config = read_config(weights_dir, config.num_layers)
    with torch.device("meta"):
        model = H3Transformer(model_config)
    model.enable_gradient_checkpointing()
    # No input casting: the float64 RoPE positions and the timesteps must reach the model untouched;
    # every module aligns its inputs with its own (unsharded) weight dtype, like the release.
    bf16 = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    fp32 = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
    for name in FP32_MODULES:
        fully_shard(getattr(model, name), mesh=mesh, mp_policy=fp32)
    for block in model.token_refiner.refiner_blocks:
        fully_shard(block, mesh=mesh, mp_policy=bf16)
    for block in model.transformer_blocks:
        fully_shard(block, mesh=mesh, mp_policy=bf16)
    fully_shard(model, mesh=mesh, mp_policy=bf16)
    # Master weights are float32; FSDP casts to bf16 (or keeps fp32 for the heads) per forward.
    model.to_empty(device="cuda")
    model.float()
    load_weights(model, weights_dir, strict=config.num_layers is None)
    return model


def lr_at(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        ratio = config.warmup_start_ratio + (1 - config.warmup_start_ratio) * step / max(1, config.warmup_steps)
        return config.lr * ratio
    progress = min(1.0, (step - config.warmup_steps) / max(1, config.max_steps - config.warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.lr * (config.min_lr_ratio + (1 - config.min_lr_ratio) * cosine)


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        dist.init_process_group("nccl")
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        shard = min(config.hsdp_shard_size, self.world)
        if self.world % shard:
            raise ValueError(f"world size {self.world} is not a multiple of hsdp_shard_size {shard}")
        self.mesh = init_device_mesh("cuda", (self.world // shard, shard), mesh_dim_names=("replicate", "shard"))
        self.run_dir = os.path.join(config.output_dir, config.run_name or "run")
        if self.rank == 0:
            os.makedirs(self.run_dir, exist_ok=True)
            with open(os.path.join(self.run_dir, "config.json"), "w") as f:
                json.dump(config.to_dict(), f, indent=1)

        log(f"world {self.world}: HSDP {self.world // shard} replicas x {shard} shards")
        self.model = build_sharded_model(config, self.mesh)
        self.model_config = self.model.config
        params = [p for p in self.model.parameters() if p.requires_grad]
        log(f"parameters: {sum(p.numel() for p in params) / 1e9:.2f}B")
        self.optimizer = torch.optim.AdamW(
            params, lr=config.lr, betas=config.betas, weight_decay=config.weight_decay, eps=1e-8, foreach=True
        )
        self.ema = None
        if config.ema_decay > 0:
            self.ema = {n: p.detach().clone() for n, p in self.model.named_parameters()}

        self.dataset = CausalH3Dataset(
            config.data_dirs,
            seed=config.seed,
            num_train_timesteps=config.num_train_timesteps,
            text_timestep=config.text_timestep,
            max_chunks=config.max_chunks,
            use_chunk_captions=config.use_chunk_captions,
            holdout=config.holdout,
        )
        self.sampler = ResumableSampler(len(self.dataset), rank=self.rank, world_size=self.world, seed=config.seed)
        log(f"dataset: {len(self.dataset)} training clips ({self.sampler.per_rank} per rank per epoch)")
        self.step = 0
        self.consumed = 0
        self._resume()
        self.sampler.set_consumed(self.consumed)
        self.loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=collate_single,
            num_workers=config.num_workers,
            pin_memory=True,
            prefetch_factor=2 if config.num_workers else None,
            persistent_workers=config.num_workers > 0,
        )
        self.wandb = None
        if config.wandb_project and self.rank == 0:
            try:
                import wandb

                self.wandb = wandb.init(
                    project=config.wandb_project, name=config.run_name, config=config.to_dict(), resume="allow",
                    id=(config.run_name or "run").replace("/", "_"),
                )
            except Exception as error:  # logging must never stop training
                print(f"wandb disabled: {error}", flush=True)

    # ------------------------------------------------------------------ checkpoints

    def _checkpoint_dirs(self) -> list[str]:
        root = os.path.join(self.run_dir, "checkpoints")
        if not os.path.isdir(root):
            return []
        steps = sorted(int(d.split("-")[1]) for d in os.listdir(root) if d.startswith("step-") and
                       os.path.exists(os.path.join(root, d, ".complete")))
        return [os.path.join(root, f"step-{s:07d}") for s in steps]

    def _state(self) -> dict:
        model_state, optim_state = get_state_dict(self.model, self.optimizer)
        state = {"model": model_state, "optim": optim_state, "meta": {"step": self.step, "consumed": self.consumed}}
        if self.ema is not None:
            state["ema"] = self.ema
        return state

    def _resume(self) -> None:
        existing = self._checkpoint_dirs()
        if not existing:
            return
        path = existing[-1]
        state = self._state()
        dcp.load(state, checkpoint_id=path)
        set_state_dict(
            self.model, self.optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"],
            options=StateDictOptions(strict=True),
        )
        self.step, self.consumed = int(state["meta"]["step"]), int(state["meta"]["consumed"])
        log(f"resumed {path}: step {self.step}, {self.consumed} samples consumed per rank")

    def save(self) -> None:
        path = os.path.join(self.run_dir, "checkpoints", f"step-{self.step:07d}")
        dcp.save(self._state(), checkpoint_id=path)
        dist.barrier()
        if self.rank == 0:
            open(os.path.join(path, ".complete"), "w").close()
            for old in self._checkpoint_dirs()[: -self.config.keep_last]:
                shutil.rmtree(old, ignore_errors=True)
        log(f"saved {path}")

    def export(self) -> None:
        source = self.ema if self.ema is not None and self.step >= self.config.ema_start_step else None
        named = source.items() if source is not None else self.model.named_parameters()
        out = os.path.join(self.run_dir, "exports", f"step-{self.step:07d}-{'ema' if source else 'live'}")
        export_weights(named, self.model_config, out)
        dist.barrier()
        log(f"exported {out}")

    # ------------------------------------------------------------------ step

    def _forward_loss(self, batch: DFBatch) -> tuple[torch.Tensor, dict]:
        plan = batch.plan.to("cuda")
        mask = build_block_mask(plan, self.config.window_chunks, self.config.caption_window_chunks, "cuda")
        video_pred, audio_pred = self.model(
            hidden_states=batch.video_rows.cuda(non_blocking=True)[None],
            audio_hidden_states=batch.audio_rows.cuda(non_blocking=True)[None],
            encoder_hidden_states=batch.text_rows.cuda(non_blocking=True)[None],
            timestep=batch.timestep.cuda(),
            timestep_indices=batch.timestep_indices.cuda(),
            token_tags=plan.token_tags,
            position_ids=plan.position_ids,
            video_indices=plan.video_indices,
            audio_indices=plan.audio_indices,
            text_indices=plan.text_indices,
            text_segment_ids=plan.text_segment_ids,
            attention_mask=mask,
        )
        for name in ("video_target", "audio_target", "video_weight", "audio_weight"):
            setattr(batch, name, getattr(batch, name).cuda(non_blocking=True))
        loss, stats = df_loss(video_pred[0], audio_pred[0], batch, audio_loss_weight=self.config.audio_loss_weight)
        stats["tokens"] = torch.tensor(float(plan.sequence_length), device="cuda")
        stats["chunks"] = torch.tensor(float(plan.num_chunks), device="cuda")
        return loss, stats

    def _update_ema(self) -> None:
        if self.ema is None or self.step < self.config.ema_start_step:
            return
        decay = self.config.ema_decay if self.step > self.config.ema_start_step else 0.0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                target = self.ema[name]
                target_local = target.to_local() if isinstance(target, DTensor) else target
                param_local = param.to_local() if isinstance(param, DTensor) else param
                target_local.lerp_(param_local, 1.0 - decay)

    def train(self) -> None:
        config = self.config
        iterator = iter(self.loader)
        log(f"training from step {self.step} to {config.max_steps}")
        window_start = time.time()
        while self.step < config.max_steps:
            batch = next(iterator)
            lr = lr_at(self.step, config)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            loss, stats = self._forward_loss(batch)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip, foreach=True)
            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()
            # Every rank votes; one non-finite rank skips the update everywhere.
            finite = torch.isfinite(grad_norm) & torch.isfinite(loss.detach())
            finite = finite.float()
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if finite.item() > 0:
                self.optimizer.step()
            else:
                log(f"step {self.step}: non-finite loss/grad, update skipped")
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            self.consumed += 1
            self._update_ema()

            if self.step % config.log_every == 0:
                values = torch.stack([stats[k] for k in sorted(stats)] + [grad_norm.float()])
                dist.all_reduce(values, op=dist.ReduceOp.AVG)
                elapsed = (time.time() - window_start) / config.log_every
                window_start = time.time()
                record = {k: float(v) for k, v in zip(sorted(stats) + ["grad_norm"], values)}
                record |= {"lr": lr, "sec_per_step": elapsed, "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
                log(f"step {self.step} " + " ".join(f"{k}={v:.4g}" for k, v in record.items()))
                if self.wandb is not None:
                    self.wandb.log(record, step=self.step)
            if self.step % config.save_every == 0 or self.step == config.max_steps:
                self.save()
            if (config.export_every and self.step % config.export_every == 0) or self.step == config.max_steps:
                self.export()
        dist.barrier()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args, overrides = parser.parse_known_args()
    Trainer(load_config(args.config, overrides)).train()


if __name__ == "__main__":
    main()
