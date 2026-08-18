"""Typed configuration with YAML layering and dotted CLI overrides."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, TypeVar

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
T = TypeVar("T")


@dataclass
class SpurConfig:
    symmetry: int = 5
    num_tiles: int = 96
    num_ret_tiles: int = 120
    translation_canvas: float | None = None
    seed: int | None = None
    rotation_canvas: float = math.pi
    rotation_mask: float = math.pi / 4


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 8
    num_layers: int = 4
    num_global_tokens: int = 4
    class_embed_dim: int = 32
    time_embed_dim: int = 64
    dropout: float = 0.1


@dataclass
class FlowConfig:
    loss: str = "l2"
    lambda_inness: float = 1.0
    lsa_workers: int = 8


@dataclass
class TrainConfig:
    batch_size: int = 64
    num_epochs: int = 101
    steps_per_epoch: int = 1400
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    min_lr_factor: float = 0.1
    grad_clip: float = 1.0
    precision: str = "fp32"
    sample_steps: int = 100
    sample_batch_size: int = 1
    seed: int = 0
    device: str = "cuda"


@dataclass
class WandbConfig:
    enable: bool = True
    project: str = "penrose-aim"


@dataclass
class OutputConfig:
    directory: str = "outputs"
    resume: str | None = None


@dataclass
class Config:
    spur: SpurConfig
    model: ModelConfig
    flow: FlowConfig
    train: TrainConfig
    wandb: WandbConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SECTIONS: dict[str, type[Any]] = {
    "spur": SpurConfig,
    "model": ModelConfig,
    "flow": FlowConfig,
    "train": TrainConfig,
    "wandb": WandbConfig,
    "output": OutputConfig,
}

IMMUTABLE_ON_RESUME = (
    "spur.symmetry",
    "spur.num_tiles",
    "spur.num_ret_tiles",
    "model.d_model",
    "model.num_heads",
    "model.num_layers",
    "model.num_global_tokens",
    "model.class_embed_dim",
    "model.time_embed_dim",
    "flow.loss",
    "flow.lambda_inness",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return data


def _merge(target: dict[str, Any], source: dict[str, Any], prefix: str = "") -> None:
    for key, value in source.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in target:
            raise ValueError(f"Unknown configuration key: {dotted}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected a mapping for {dotted}")
            _merge(target[key], value, dotted)
        else:
            target[key] = value


def _override(target: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be dotted.key=value, got {expression!r}")
    dotted, raw = expression.split("=", 1)
    keys = dotted.split(".")
    cursor: dict[str, Any] = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            raise ValueError(f"Unknown configuration key: {dotted}")
        cursor = cursor[key]
    if keys[-1] not in cursor:
        raise ValueError(f"Unknown configuration key: {dotted}")
    cursor[keys[-1]] = yaml.safe_load(raw)


def _make_dataclass(cls: type[T], values: dict[str, Any]) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**values)


def _validate(config: Config) -> None:
    if config.spur.symmetry not in (5, 6):
        raise ValueError("spur.symmetry must be 5 or 6")
    if config.spur.num_tiles <= 0 or config.spur.num_ret_tiles <= 1:
        raise ValueError("tile counts must be positive and num_ret_tiles must exceed one")
    if config.model.d_model % config.model.num_heads:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if config.model.time_embed_dim < 4 or config.model.time_embed_dim % 2:
        raise ValueError("model.time_embed_dim must be an even integer >= 4")
    if config.flow.loss not in ("l1", "l2"):
        raise ValueError("flow.loss must be 'l1' or 'l2'")
    if config.train.precision != "fp32":
        raise ValueError("PenroseAim supports FP32 only")
    if min(
        config.train.batch_size,
        config.train.num_epochs,
        config.train.steps_per_epoch,
        config.train.sample_steps,
    ) <= 0:
        raise ValueError("training counts must be positive")


def config_from_dict(values: dict[str, Any]) -> Config:
    config = Config(
        **{
            name: _make_dataclass(section_type, copy.deepcopy(values[name]))
            for name, section_type in SECTIONS.items()
        }
    )
    _validate(config)
    return config


def load_config(argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Train PenroseAim")
    parser.add_argument("--config", type=Path, help="Optional experiment YAML")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="Resume model/run state with a fresh optimizer and scheduler",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted overrides applied last, e.g. train.batch_size=8",
    )
    args = parser.parse_args(argv)

    merged = _read_yaml(PROJECT_DIR / "default.yaml")
    if args.resume:
        import torch

        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        _merge(merged, checkpoint["config"])
    if args.config:
        _merge(merged, _read_yaml(args.config))
    for expression in args.overrides:
        _override(merged, expression)
    if args.resume:
        merged["output"]["resume"] = str(args.resume)

    return config_from_dict(merged), args


def effective_translation(config: SpurConfig) -> float:
    if config.translation_canvas is not None:
        return float(config.translation_canvas)
    return math.sqrt(config.num_tiles) if config.symmetry == 5 else 2.0


def make_names(config: Config, now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(timezone.utc)
    descriptor = (
        f"{config.model.d_model}x{config.model.num_layers}"
        f"_t{config.spur.num_ret_tiles}_{config.flow.loss}"
    )
    timestamp = f"{now:%m%d_%H%M}"
    identifier = f"aim_{timestamp}_{descriptor}"
    return identifier, identifier


def nested_value(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for key in dotted.split("."):
        value = value[key]
    return value


def validate_resume_config(current: Config, saved: dict[str, Any]) -> None:
    current_dict = current.to_dict()
    differences = [
        key
        for key in IMMUTABLE_ON_RESUME
        if nested_value(current_dict, key) != nested_value(saved, key)
    ]
    if differences:
        details = ", ".join(
            f"{key}: checkpoint={nested_value(saved, key)!r}, current={nested_value(current_dict, key)!r}"
            for key in differences
        )
        raise ValueError(f"Immutable resume configuration changed: {details}")
