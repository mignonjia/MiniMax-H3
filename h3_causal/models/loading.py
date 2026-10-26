"""Build the H3 transformer and load checkpoint weights into (possibly FSDP-sharded) parameters."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, distribute_tensor

from h3_causal.models.transformer import H3Config, H3Transformer, RotaryPosEmbed


def read_config(weights_dir: str, num_layers: int | None = None) -> H3Config:
    with open(os.path.join(weights_dir, "config.json")) as f:
        config = H3Config.from_dict(json.load(f))
    if num_layers is not None:
        config.num_layers = num_layers
    return config


def weight_index(weights_dir: str) -> dict[str, str]:
    """Parameter name -> safetensors file, for sharded or single-file checkpoints."""
    for name in ("diffusion_pytorch_model.safetensors.index.json", "model.safetensors.index.json"):
        path = os.path.join(weights_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return {k: os.path.join(weights_dir, v) for k, v in json.load(f)["weight_map"].items()}
    for name in ("diffusion_pytorch_model.safetensors", "model.safetensors"):
        path = os.path.join(weights_dir, name)
        if os.path.exists(path):
            with safe_open(path, "pt") as f:
                return {k: path for k in f.keys()}
    raise FileNotFoundError(f"no safetensors checkpoint in {weights_dir}")


def reset_rope(model: torch.nn.Module) -> None:
    """Recompute non-persistent RoPE buffers (left uninitialised by `to_empty`)."""
    for module in model.modules():
        if isinstance(module, RotaryPosEmbed):
            fresh = RotaryPosEmbed(module.inv_freq.numel(), model.config.rope_theta)
            module.inv_freq = fresh.inv_freq.to(module.inv_freq.device)


@torch.no_grad()
def load_weights(model: H3Transformer, weights_dir: str, *, strict: bool = True) -> None:
    """Rank 0 reads each tensor once and scatters it into the local shards; plain tensors are filled directly."""
    index = weight_index(weights_dir)
    params = dict(model.named_parameters())
    missing = sorted(set(params) - set(index))
    if strict and missing:
        raise KeyError(f"checkpoint lacks {missing[:5]} (+{max(0, len(missing) - 5)})")
    rank = dist.get_rank() if dist.is_initialized() else 0
    handles: dict[str, object] = {}
    try:
        for name, param in params.items():
            if name not in index:
                continue
            local = param.to_local() if isinstance(param, DTensor) else param
            if isinstance(param, DTensor):
                full = torch.empty(param.shape, dtype=param.dtype, device=local.device)
                if rank == 0:
                    path = index[name]
                    if path not in handles:
                        handles[path] = safe_open(path, "pt", device="cpu")
                    full.copy_(handles[path].get_tensor(name))
                shard = distribute_tensor(full, param.device_mesh, param.placements, src_data_rank=0)
                local.copy_(shard.to_local())
                del full, shard
            else:
                path = index[name]
                if path not in handles:
                    handles[path] = safe_open(path, "pt", device="cpu")
                local.copy_(handles[path].get_tensor(name))
    finally:
        handles.clear()
    reset_rope(model)


@torch.no_grad()
def export_weights(named_tensors, config: H3Config, out_dir: str, *, dtype=torch.bfloat16, shard_bytes: int = 5 << 30):
    """Gather (possibly DTensor) tensors and write a diffusers-style sharded safetensors checkpoint on rank 0."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    shard, shard_size, weight_map, files = {}, 0, {}, []

    def flush():
        nonlocal shard, shard_size
        if shard:
            name = f"diffusion_pytorch_model-{len(files) + 1:05d}.safetensors"
            save_file(shard, os.path.join(out_dir, name))
            files.append(name)
            weight_map.update({k: name for k in shard})
        shard, shard_size = {}, 0

    for name, tensor in named_tensors:
        full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        if rank == 0:
            # Keep the release's float32 heads float32.
            keep_fp32 = name.split(".")[0] in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out")
            shard[name] = full.detach().to("cpu", torch.float32 if keep_fp32 else dtype).contiguous()
            shard_size += shard[name].numel() * shard[name].element_size()
            if shard_size >= shard_bytes:
                flush()
        del full
    if rank == 0:
        flush()
        with open(os.path.join(out_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
        config_dict = {k: (list(v) if isinstance(v, tuple) else v) for k, v in config.__dict__.items()}
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"_class_name": "MiniMaxH3Transformer3DModel", **config_dict}, f, indent=1)


def build_model(weights_dir: str, *, device, dtype=torch.bfloat16, num_layers: int | None = None) -> H3Transformer:
    """Unsharded model with weights (inference / eval on one GPU)."""
    config = read_config(weights_dir, num_layers)
    with torch.device("meta"):
        model = H3Transformer(config)
    model.to_empty(device=device)
    model.to(dtype)
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        getattr(model, name).float()
    load_weights(model, weights_dir, strict=num_layers is None)
    return model.eval()
