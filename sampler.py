"""Hybrid geometry-flow/discrete-inness training and reverse sampling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch

from config import Config, config_from_dict, effective_translation
from model import DirectTransformer
from null_inness import get_rho_p, sample_vertex_in_null


PROJECT_DIR = Path(__file__).resolve().parent
SPUR_DIR = Path(
    os.environ.get("PENROSE_SPUR_PATH", PROJECT_DIR.parent / "PenroseSpur")
).resolve()
if not (SPUR_DIR / "sampler.py").exists():
    raise ImportError(
        f"PenroseSpur not found at {SPUR_DIR}; set PENROSE_SPUR_PATH"
    )
sys.path.insert(0, str(SPUR_DIR))
from show import VideoOptions, save_tiles_svg, save_trajectory_mp4  # noqa: E402


def _load_spur_file(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(
        f"penrose_spur_{name}", SPUR_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load PenroseSpur module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_spur_sampler = _load_spur_file("sampler")
_spur_match = _load_spur_file("match")
_spur_flow_geometry = _load_spur_file("flow_geometry")
_spur_utils = _load_spur_file("utils")
_spur_lattice_loss = _load_spur_file("lattice_loss")
_spur_convert = _load_spur_file("convert")
_spur_svg = _load_spur_file("svg")
SpurSampler = _spur_sampler.SpurSampler
ANGLE_SCALE = _spur_sampler.ANGLE_SCALE
lattice_loss = _spur_lattice_loss.lattice_loss
get_colors = _spur_utils.get_colors


@dataclass
class PreparedFlowBatch:
    noisy_state: torch.Tensor
    target_velocity: torch.Tensor
    target_inness: torch.Tensor
    target_vertex_in: torch.Tensor
    colors: torch.Tensor
    labels: torch.Tensor
    time: torch.Tensor


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def build_spur(config: Config, device: torch.device) -> Any:
    return SpurSampler(
        symmetry=config.spur.symmetry,
        num_tiles=config.spur.num_tiles,
        num_ret_tiles=config.spur.num_ret_tiles,
        translation_canvas=effective_translation(config.spur),
        seed=config.spur.seed,
        device=device,
        rotation_canvas=config.spur.rotation_canvas,
        rotation_mask=config.spur.rotation_mask,
    )


def _validated_inness(inness: torch.Tensor) -> torch.Tensor:
    tolerance = 1e-6
    if inness.min().item() < -tolerance or inness.max().item() > 1.0 + tolerance:
        raise ValueError(
            "PenroseSpur inness must already be normalized to [0,1], "
            f"got [{inness.min().item():.6g}, {inness.max().item():.6g}]"
        )
    return inness.clamp(0.0, 1.0)


def prepare_flow_batch(
    spur: Any,
    config: Config,
    generator: torch.Generator,
) -> PreparedFlowBatch:
    batch = spur.sample_batch(config.train.batch_size, generator=generator)
    data_geometry = batch["xya"].float()
    colors = batch["colors"].long()
    labels = batch["labels"].long()
    target_inness = _validated_inness(batch["inness"].float())
    target_vertex_in = batch["vertex_in"].bool()

    geometry_noise = spur.sample_noise(
        config.train.batch_size, generator=generator
    ).float()
    if not hasattr(spur, "_null_inness_probs"):
        spur._null_inness_probs = get_rho_p(spur)
    rho, p = spur._null_inness_probs
    vertex_in_0 = sample_vertex_in_null(
        tuple(target_vertex_in.shape),
        rho,
        p,
        device=spur.device,
        generator=generator,
    )
    match_result = _spur_match.match(
        data_geometry,
        geometry_noise,
        method="lsa",
        colors=colors,
        lsa_workers=config.flow.lsa_workers,
        return_details=True,
    )
    permutation = match_result.permutation
    if permutation is None:
        raise RuntimeError("LSA did not return a permutation")
    matched_vertex_in_0 = vertex_in_0.gather(
        1,
        permutation.unsqueeze(-1).expand(-1, -1, spur.V1),
    )
    unit_time = torch.rand(
        config.train.batch_size, device=spur.device, generator=generator
    )
    path_time = torch.sin(unit_time * math.pi / 2.0)
    noisy_geometry, target_velocity = _spur_flow_geometry.flow(
        match_result.matched_noise,
        data_geometry,
        path_time,
    )
    reveal = torch.rand(
        target_vertex_in.shape,
        device=spur.device,
        generator=generator,
    ) < path_time[:, None, None]
    vertex_in_t = torch.where(
        reveal,
        target_vertex_in,
        matched_vertex_in_0,
    )
    noisy_state = torch.cat(
        (noisy_geometry, vertex_in_t.to(data_geometry.dtype)),
        dim=-1,
    )
    return PreparedFlowBatch(
        noisy_state=noisy_state,
        target_velocity=target_velocity,
        target_inness=target_inness,
        target_vertex_in=target_vertex_in,
        colors=colors,
        labels=labels,
        time=path_time,
    )


def canonical_colors(
    symmetry: int,
    batch_size: int,
    num_tiles: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    base = get_colors(symmetry, num_tiles, device=device).long()
    return torch.stack(
        [base[torch.randperm(num_tiles, device=device, generator=generator)] for _ in range(batch_size)]
    )


@torch.no_grad()
def reverse_sample(
    model: DirectTransformer,
    spur: Any,
    config: Config,
    colors: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
    generator: torch.Generator,
    return_trajectory: bool = False,
) -> torch.Tensor:
    """Integrate geometry and ancestrally reveal discrete vertex probes."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    batch_size = labels.shape[0]
    if colors.shape != (
        batch_size,
        config.spur.num_ret_tiles,
    ):
        raise ValueError(
            f"Expected colors shape "
            f"{(batch_size, config.spur.num_ret_tiles)}, got {colors.shape}"
        )

    geometry = spur.sample_noise(
        batch_size, generator=generator
    ).float()
    if not hasattr(spur, "_null_inness_probs"):
        spur._null_inness_probs = get_rho_p(spur)
    rho, p = spur._null_inness_probs
    vertex_in = sample_vertex_in_null(
        (batch_size, config.spur.num_ret_tiles, spur.V1),
        rho,
        p,
        device=spur.device,
        generator=generator,
    ).to(geometry.dtype)
    state = torch.cat((geometry, vertex_in), dim=-1)
    reveal_times = torch.rand(
        vertex_in.shape,
        device=spur.device,
        generator=generator,
    )
    revealed = torch.zeros_like(vertex_in, dtype=torch.bool)

    unit_grid = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        device=spur.device,
        dtype=geometry.dtype,
    )
    path_grid = torch.sin(unit_grid * math.pi / 2.0)
    model.eval()
    trajectory = [] if return_trajectory else None
    for start, end in zip(path_grid[:-1], path_grid[1:]):
        time = start.expand(batch_size)
        prediction = model(state, colors, time, labels)
        state[..., :3] = _spur_flow_geometry.flow_step(
            state[..., :3],
            prediction[..., :3],
            end - start,
        )
        newly_revealed = (~revealed) & (reveal_times <= end)
        sampled_vertex_in = torch.rand(
            vertex_in.shape,
            device=spur.device,
            generator=generator,
        ) < prediction[..., 3:].sigmoid()
        state[..., 3:] = torch.where(
            newly_revealed,
            sampled_vertex_in.to(state.dtype),
            state[..., 3:],
        )
        revealed |= newly_revealed
        if trajectory is not None:
            trajectory.append(state.clone())
    return torch.stack(trajectory) if trajectory is not None else state


def decoded_sample(state: torch.Tensor, colors: torch.Tensor) -> dict[str, torch.Tensor]:
    theta = state[..., 2] / ANGLE_SCALE
    theta = torch.remainder(theta + math.pi, 2.0 * math.pi) - math.pi
    vertex_in = state[..., 3:] > 0.5
    return {
        "x": state[..., 0],
        "y": state[..., 1],
        "theta": theta,
        "s": vertex_in.float().mean(dim=-1),
        "vertex_in": vertex_in,
        "color": colors,
    }


def save_sample_svg(
    path: Path,
    state: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> Path:
    """Save the first generated sample using PenroseSpur's SVG renderer."""
    decoded = decoded_sample(state[:1], colors[:1])
    xya = torch.stack(
        (decoded["x"][0], decoded["y"][0], decoded["theta"][0]), dim=-1
    )
    return save_tiles_svg(
        path,
        xya,
        colors[0],
        symmetry=symmetry,
        side=side,
        show_arcs=symmetry == 5,
        opacities=decoded["s"][0],
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def class_label_for_name(class_names: list[str], value: str) -> int:
    try:
        label = int(value)
    except ValueError:
        normalized = [name.casefold() for name in class_names]
        try:
            return normalized.index(value.casefold())
        except ValueError as error:
            raise ValueError(
                f"Unknown class {value!r}; choices: {class_names}"
            ) from error
    if not 0 <= label < len(class_names):
        raise ValueError(f"Class label must be in [0,{len(class_names) - 1}]")
    return label


def frame_tiles(
    state: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
) -> tuple[np.ndarray, np.ndarray]:
    decoded = decoded_sample(state, colors)
    centers = torch.stack((decoded["x"][0], decoded["y"][0]), dim=-1)
    polygons = _spur_convert.vertices(
        symmetry,
        centers.detach().cpu().numpy(),
        decoded["theta"][0].detach().cpu().numpy(),
        colors[0].detach().cpu().numpy(),
        side,
    )
    return polygons, decoded["s"][0].detach().cpu().numpy()


def save_video(
    path: Path,
    trajectory: torch.Tensor,
    colors: torch.Tensor,
    symmetry: int,
    side: float,
    size: int,
    fps: int,
) -> Path:
    xya_frames = []
    opacity_frames = []
    for frame in trajectory:
        decoded = decoded_sample(frame, colors)
        xya_frames.append(
            torch.stack(
                (decoded["x"][0], decoded["y"][0], decoded["theta"][0]), dim=-1
            )
        )
        opacity_frames.append(decoded["s"][0])
    return save_trajectory_mp4(
        xya_frames,
        colors[0],
        path,
        symmetry=symmetry,
        side=side,
        opacities=opacity_frames,
        show_arcs=symmetry == 5,
        options=VideoOptions(fps=fps, display_height=size),
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample SVGs or animate a PenroseAim checkpoint"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "-a",
        choices=("all", "sample", "animate"),
        required=True,
        help="action: all classes, one-class SVG sampling, or animation",
    )
    parser.add_argument("-c", help="class name or numeric label", default="bat")
    parser.add_argument("-N", type=int, default=200, help="sampling steps")
    parser.add_argument(
        "-f",
        choices=("all", "ends"),
        default="ends",
        help="SVG frames for the sample action",
    )
    parser.add_argument("-o", type=Path, required=True, help="output directory")
    parser.add_argument("-S", type=int, default=720, help="video frame size")
    parser.add_argument("-r", type=int, default=20, help="video frame rate")
    parser.add_argument("-s", type=int, default=0, help="random seed")
    parser.add_argument("-d", default="auto", help="device")
    args = parser.parse_args()

    if args.N <= 0:
        parser.error("-N must be positive")
    if args.S <= 0:
        parser.error("-S must be positive")
    if args.r <= 0:
        parser.error("-r must be positive")
    if args.a != "all" and args.c is None:
        parser.error(f"-c is required for action {args.a!r}")

    device = _device(args.d)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    spur = build_spur(config, device)
    class_names = list(spur.class_names)
    model = DirectTransformer(
        config.model, len(class_names), spur.V1
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    side = float(checkpoint["side"])
    generator = make_generator(device, args.s)

    args.o.mkdir(parents=True, exist_ok=True)
    prefix = args.checkpoint.stem
    if args.a == "all":
        labels = torch.arange(len(class_names), device=device)
        colors = canonical_colors(
            config.spur.symmetry,
            len(class_names),
            config.spur.num_ret_tiles,
            device,
            generator,
        )
        state = reverse_sample(
            model, spur, config, colors, labels, args.N, generator
        )
        for label, class_name in enumerate(class_names):
            save_sample_svg(
                args.o / f"{prefix}_{label:02d}_{safe_name(class_name)}.svg",
                state[label : label + 1],
                colors[label : label + 1],
                config.spur.symmetry,
                side,
            )
        print(f"Saved {len(class_names)} final SVGs to {args.o}")
        return

    class_label = class_label_for_name(class_names, args.c)
    class_name = class_names[class_label]
    colors = canonical_colors(
        config.spur.symmetry,
        1,
        config.spur.num_ret_tiles,
        device,
        generator,
    )
    labels = torch.tensor([class_label], device=device)
    trajectory = reverse_sample(
        model,
        spur,
        config,
        colors,
        labels,
        args.N,
        generator,
        return_trajectory=True,
    )
    frame_indices = (
        range(len(trajectory))
        if args.a == "animate" or args.f == "all"
        else sorted({0, len(trajectory) - 1})
    )
    class_directory = args.o / f"{prefix}_{safe_name(class_name)}"
    for index in frame_indices:
        save_sample_svg(
            class_directory / f"frame_{index:04d}.svg",
            trajectory[index],
            colors,
            config.spur.symmetry,
            side,
        )
    print(f"Saved SVG frames to {class_directory}")

    if args.a == "animate":
        video_path = args.o / f"{prefix}_{safe_name(class_name)}.mp4"
        save_video(
            video_path,
            trajectory,
            colors,
            config.spur.symmetry,
            side,
            args.S,
            args.r,
        )
        print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
