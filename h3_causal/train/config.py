"""Training configuration: a YAML file plus `key=value` CLI overrides."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from omegaconf import OmegaConf


@dataclass
class TrainConfig:
    # Data
    data_dirs: list[str] = field(default_factory=list)
    holdout: int = 64
    max_chunks: int | None = None
    use_chunk_captions: bool = True
    num_workers: int = 4
    # Causal recipe
    window_chunks: int = 8
    caption_window_chunks: int = 1
    text_timestep: float = 1.0
    num_train_timesteps: int = 1000
    audio_loss_weight: float = 1.0
    # Model
    h3_root: str = "/mnt/weka/shrd/wm/areeb/H3"
    init_from: str | None = None  # consolidated safetensors dir; default: h3_root/transformer
    num_layers: int | None = None  # truncate the stack (smoke tests only)
    # Optimisation
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 200
    warmup_start_ratio: float = 0.1
    max_steps: int = 3000
    min_lr_ratio: float = 0.3
    grad_clip: float = 1.0
    ema_decay: float = 0.0  # 0 disables EMA
    ema_start_step: int = 200
    # Parallelism
    hsdp_shard_size: int = 16  # ranks one replica is sharded over; replicas = world / shard
    # Checkpoints and logging
    output_dir: str = "outputs/h3_causal"
    save_every: int = 500
    keep_last: int = 3
    export_every: int = 0  # consolidated bf16 safetensors of EMA (or live) weights; 0 = only at the end
    log_every: int = 1
    wandb_project: str | None = None
    run_name: str | None = None
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | None, overrides: list[str]) -> TrainConfig:
    base = OmegaConf.structured(TrainConfig)
    merged = OmegaConf.merge(base, OmegaConf.load(path) if path else {}, OmegaConf.from_dotlist(overrides))
    values = OmegaConf.to_container(merged, resolve=True)
    config = TrainConfig(**{f.name: values[f.name] for f in fields(TrainConfig)})
    config.betas = tuple(config.betas)
    return config
