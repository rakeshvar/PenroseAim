"""Train the single PenroseAim inness-aware OTFM model."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR

from config import Config, load_config, make_names, validate_resume_config
from losses import inness_weighted_velocity_loss
from model import DirectTransformer
from sampler import (
    build_spur,
    canonical_colors,
    lattice_loss,
    make_generator,
    prepare_flow_batch,
    reverse_sample,
)


def choose_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; override train.device=cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Config
) -> LambdaLR:
    epochs = config.train.num_epochs
    warmup = min(10, int(epochs * 0.05))
    floor = config.train.min_lr_factor

    def factor(position: int) -> float:
        if warmup and position <= warmup:
            return 0.01 + 0.99 * position / warmup
        if position <= epochs:
            progress = (position - warmup) / max(1, epochs - warmup)
            return floor + (1.0 - floor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return floor

    return LambdaLR(optimizer, lr_lambda=factor)


def capture_rng(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None,
        "generator": generator.get_state(),
    }


def restore_rng(states: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if states["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    generator.set_state(states["generator"])


def atomic_save(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def retain_newest_and_best(
    checkpoint_dir: Path, identifier: str, newest_epoch: int, best_epoch: int
) -> None:
    retain = {newest_epoch, best_epoch}
    for path in checkpoint_dir.glob(f"{identifier}_e*.pt"):
        try:
            epoch = int(path.stem.rsplit("_e", 1)[1])
        except (IndexError, ValueError):
            continue
        if epoch not in retain:
            path.unlink()


class WandbLogger:
    def __init__(
        self,
        config: Config,
        run_name: str,
        run_id: str | None,
        full_config: dict[str, Any],
        resume_existing: bool,
    ):
        self.run = None
        if not config.wandb.enable:
            return
        if not os.environ.get("WANDB_API_KEY"):
            print("WandB disabled: WANDB_API_KEY is not set")
            return
        try:
            import wandb
        except ModuleNotFoundError:
            print("WandB disabled: package is not installed")
            return
        kwargs: dict[str, Any] = {
            "project": config.wandb.project,
            "name": run_name,
            "config": full_config,
        }
        if run_id:
            kwargs.update(
                id=run_id,
                resume="must" if resume_existing else "never",
            )
        self.run = wandb.init(**kwargs)

    @property
    def run_id(self) -> str | None:
        return self.run.id if self.run is not None else None

    def log(self, metrics: dict[str, float], epoch: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=epoch)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def checkpoint_payload(
    *,
    epoch: int,
    average_loss: float,
    best_epoch: int,
    best_loss: float,
    model: DirectTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    config: Config,
    identifier: str,
    run_name: str,
    wandb_run_id: str | None,
    output_directory: Path,
    generator: torch.Generator,
    side: float,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "global_step": (epoch + 1) * config.train.steps_per_epoch,
        "average_loss": average_loss,
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "config": config.to_dict(),
        "side": side,
        "identifier": identifier,
        "wandb_run_name": run_name,
        "wandb_run_id": wandb_run_id,
        "output_directory": str(output_directory),
        "rng": capture_rng(generator),
    }


def train(config: Config, *, reset_optimizer: bool = False) -> Path | None:
    seed_everything(config.train.seed)
    device = choose_device(config.train.device)
    resume_path = Path(config.output.resume) if config.output.resume else None
    resume = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path
        else None
    )
    if resume:
        validate_resume_config(config, resume["config"])

    spur = build_spur(config, device)
    model = DirectTransformer(
        config.model, len(spur.class_names), spur.V1
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_scheduler(optimizer, config)
    generator = make_generator(device, config.train.seed)

    if resume:
        model.load_state_dict(resume["model"])
        if reset_optimizer:
            print("Optimizer and scheduler reset for weights-only continuation")
        else:
            optimizer.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
        restore_rng(resume["rng"], generator)
        start_epoch = resume["epoch"] + 1
        identifier = resume["identifier"]
        run_name = resume["wandb_run_name"]
        best_epoch = resume["best_epoch"]
        best_loss = resume["best_loss"]
        output_directory = Path(resume["output_directory"])
        wandb_run_id = resume.get("wandb_run_id")
    else:
        start_epoch = 0
        identifier, run_name = make_names(config)
        best_epoch = -1
        best_loss = math.inf
        output_directory = Path(config.output.directory) / identifier
        wandb_run_id = None

    checkpoint_dir = output_directory / "checkpoints"
    logger = WandbLogger(
        config,
        run_name,
        wandb_run_id or identifier,
        config.to_dict(),
        resume_existing=resume is not None,
    )
    print(f"Device: {device}")
    print(f"Identifier: {identifier}")
    print(f"Output: {output_directory}")

    last_checkpoint: Path | None = resume_path
    try:
        for epoch in range(start_epoch, config.train.num_epochs):
            model.train()
            loss_sum = 0.0
            geometry_sum = 0.0
            inness_sum = 0.0
            gradient_norm_sum = 0.0
            for _ in range(config.train.steps_per_epoch):
                prepared = prepare_flow_batch(spur, config, generator)
                prediction = model(
                    prepared.noisy_state,
                    prepared.colors,
                    prepared.time,
                    prepared.labels,
                )
                terms = inness_weighted_velocity_loss(
                    prediction,
                    prepared.target_velocity,
                    prepared.target_inness,
                    prepared.target_vertex_in,
                    config.flow,
                )
                optimizer.zero_grad(set_to_none=True)
                terms.total.backward()
                gradient_norm = clip_grad_norm_(
                    model.parameters(), config.train.grad_clip
                )
                optimizer.step()
                loss_sum += terms.total.detach().item()
                geometry_sum += terms.geometry.detach().item()
                inness_sum += terms.inness.detach().item()
                gradient_norm_sum += gradient_norm.detach().item()

            scheduler.step()
            average_loss = loss_sum / config.train.steps_per_epoch
            if average_loss < best_loss:
                best_loss = average_loss
                best_epoch = epoch

            colors = canonical_colors(
                config.spur.symmetry,
                config.train.sample_batch_size,
                config.spur.num_ret_tiles,
                device,
                generator,
            )
            labels = torch.zeros(
                config.train.sample_batch_size,
                device=device,
                dtype=torch.long,
            )
            sample = reverse_sample(
                model,
                spur,
                config,
                colors,
                labels,
                config.train.sample_steps,
                generator,
            )
            sample_lattice_loss = lattice_loss(
                config.spur.symmetry,
                sample[..., :2],
                spur.side,
            ).item()
            geometric_loss = geometry_sum / config.train.steps_per_epoch
            inness_loss = inness_sum / config.train.steps_per_epoch
            metrics = {
                "total_loss": average_loss,
                "geometric_loss": geometric_loss,
                "inness_loss": inness_loss,
                "lattice_loss": sample_lattice_loss,
                "grad_norm": gradient_norm_sum / config.train.steps_per_epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            logger.log(metrics, epoch)
            print(
                f"Epoch {epoch:03d} loss={average_loss:.6f} "
                f"geometry={geometric_loss:.6f} "
                f"inness={inness_loss:.6f} "
                f"lattice={sample_lattice_loss:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f} "
                f"lr={metrics['learning_rate']:.6g}"
            )

            checkpoint_id = f"{identifier}_e{epoch:03d}"
            path = checkpoint_dir / f"{checkpoint_id}.pt"
            atomic_save(
                checkpoint_payload(
                    epoch=epoch,
                    average_loss=average_loss,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    identifier=identifier,
                    run_name=run_name,
                    wandb_run_id=logger.run_id or wandb_run_id,
                    output_directory=output_directory,
                    generator=generator,
                    side=spur.side,
                ),
                path,
            )
            retain_newest_and_best(
                checkpoint_dir, identifier, epoch, best_epoch
            )
            last_checkpoint = path
    finally:
        logger.finish()
    return last_checkpoint


def main() -> None:
    config, args = load_config()
    train(config, reset_optimizer=args.reset_optimizer)


if __name__ == "__main__":
    main()
