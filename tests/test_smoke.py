"""End-to-end local smoke checks for PenroseAim."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from config import FlowConfig, config_from_dict, make_names  # noqa: E402
from losses import inness_weighted_velocity_loss  # noqa: E402
from model import DirectTransformer  # noqa: E402
from sampler import (  # noqa: E402
    build_spur,
    make_generator,
    prepare_flow_batch,
    reverse_sample,
)
from train import retain_newest_and_best  # noqa: E402


def smoke_values(symmetry: int, output: Path) -> dict:
    with (PROJECT_DIR / "default.yaml").open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    values = copy.deepcopy(values)
    values["spur"].update(
        symmetry=symmetry,
        num_tiles=24,
        num_ret_tiles=30,
        translation_canvas=2.0,
        seed=7,
    )
    values["model"].update(
        d_model=16,
        num_heads=4,
        num_layers=1,
        num_global_tokens=2,
        class_embed_dim=4,
        time_embed_dim=8,
        dropout=0.0,
    )
    values["flow"].update(loss="l2", lsa_workers=1)
    values["train"].update(
        batch_size=1,
        num_epochs=1,
        steps_per_epoch=1,
        sample_steps=2,
        sample_batch_size=1,
        device="cpu",
        seed=11,
    )
    values["wandb"]["enable"] = False
    values["output"]["directory"] = str(output)
    return values


def check_model_path(symmetry: int, output: Path) -> None:
    config = config_from_dict(smoke_values(symmetry, output))
    device = torch.device("cpu")
    spur = build_spur(config, device)
    spur._null_inness_probs = (0.4, 0.75)
    generator = make_generator(device, config.train.seed)
    model = DirectTransformer(
        config.model, len(spur.class_names), spur.V1
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    prepared = prepare_flow_batch(spur, config, generator)
    assert prepared.target_inness.min().item() >= 0.0
    assert prepared.target_inness.max().item() <= 1.0
    assert prepared.target_vertex_in.dtype == torch.bool
    assert prepared.target_vertex_in.shape[-1] == spur.V1
    assert torch.all(
        (prepared.noisy_state[..., 3:] == 0)
        | (prepared.noisy_state[..., 3:] == 1)
    )
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
    assert torch.isfinite(terms.total)
    optimizer.zero_grad(set_to_none=True)
    terms.total.backward()
    optimizer.step()

    trajectory = reverse_sample(
        model,
        spur,
        config,
        prepared.colors,
        prepared.labels,
        steps=2,
        generator=generator,
        return_trajectory=True,
    )
    assert trajectory.shape[0] == 2
    sampled = trajectory[-1]
    assert sampled.shape == prepared.noisy_state.shape
    assert torch.isfinite(sampled).all()
    assert torch.all(
        (sampled[..., 3:] == 0) | (sampled[..., 3:] == 1)
    )


def check_identifier_contract(root: Path) -> None:
    config = config_from_dict(smoke_values(6, root))
    identifier, run_name = make_names(
        config, datetime(2026, 8, 14, 0, 21, tzinfo=timezone.utc)
    )
    assert identifier == "aim_0814_0021_16x1_t30_l2"
    assert run_name == identifier


def check_soft_inness_weighting() -> None:
    prediction = torch.zeros(1, 2, 4)
    prediction[0, 0, :3] = 1.0
    target_velocity = torch.zeros(1, 2, 3)
    target_vertex_in = torch.zeros(1, 2, 1, dtype=torch.bool)
    config = FlowConfig(loss="l2", lambda_inness=1.0)

    edge_weighted = inness_weighted_velocity_loss(
        prediction,
        target_velocity,
        torch.tensor([[0.1, 1.0]]),
        target_vertex_in,
        config,
    )
    interior_weighted = inness_weighted_velocity_loss(
        prediction,
        target_velocity,
        torch.tensor([[1.0, 0.1]]),
        target_vertex_in,
        config,
    )
    assert interior_weighted.geometry > edge_weighted.geometry
    assert torch.equal(interior_weighted.inness, edge_weighted.inness)


def check_retention(root: Path) -> None:
    identifier = "aim-retention"
    folder = root / "checkpoints"
    folder.mkdir(parents=True)
    for epoch in range(3):
        (folder / f"{identifier}_e{epoch:03d}.pt").touch()
    retain_newest_and_best(folder, identifier, newest_epoch=2, best_epoch=0)
    names = {path.name for path in folder.glob("*.pt")}
    assert names == {
        f"{identifier}_e000.pt",
        f"{identifier}_e002.pt",
    }


def run_command(
    script: str, arguments: list[str], environment: dict[str, str]
) -> None:
    result = subprocess.run(
        [sys.executable, "-u", script, *arguments],
        cwd=PROJECT_DIR,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise AssertionError(
            f"Fresh-process command failed ({result.returncode}):\n{result.stdout}"
        )


def check_fresh_process_resume(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    values = smoke_values(6, root / "runs")
    config_path = root / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(values, sort_keys=False), encoding="utf-8"
    )
    environment = os.environ.copy()
    environment["PENROSE_SPUR_PATH"] = str(PROJECT_DIR.parent / "PenroseSpur")
    environment["PENROSE_INNESS_PROBS_PATH"] = str(root / "inness_probs.csv")
    run_command("train.py", ["--config", str(config_path)], environment)

    checkpoints = sorted((root / "runs").glob("*/checkpoints/*.pt"))
    assert len(checkpoints) == 1
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    assert first["epoch"] == 0
    assert first["global_step"] == 1
    assert first["side"] > 0

    run_command(
        "train.py",
        [
            "--resume",
            str(checkpoints[0]),
            "--reset-optimizer",
            "train.num_epochs=2",
        ],
        environment,
    )
    resumed = sorted((root / "runs").glob("*/checkpoints/*.pt"))
    newest = max(
        (
            torch.load(path, map_location="cpu", weights_only=False)
            for path in resumed
        ),
        key=lambda checkpoint: checkpoint["epoch"],
    )
    assert newest["epoch"] == 1
    assert newest["global_step"] == 2
    assert newest["identifier"] == first["identifier"]
    assert newest["output_directory"] == first["output_directory"]
    assert (root / "inness_probs.csv").exists()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="penrose-aim-smoke-") as temporary:
        root = Path(temporary)
        check_model_path(5, root / "direct-5")
        check_model_path(6, root / "direct-6")
        check_identifier_contract(root / "identifier")
        check_soft_inness_weighting()
        check_retention(root / "retention")
        check_fresh_process_resume(root / "resume")
    print("PenroseAim smoke test passed")


if __name__ == "__main__":
    main()
