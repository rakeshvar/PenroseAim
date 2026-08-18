"""Fit, cache, and sample the discrete null distribution for vertex inness."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
import torch


CSV_PATH = Path(
    os.environ.get(
        "PENROSE_INNESS_PROBS_PATH",
        Path(__file__).with_name("inness_probs.csv"),
    )
)
CSV_FIELDS = (
    "symmetry",
    "num_tiles",
    "num_ret_tiles",
    "rho",
    "p",
    "samples",
    "seeds",
)
DEFAULT_SEEDS = tuple(range(5))
POINT_BUDGET = 6_000_000


def _key(spur: Any) -> tuple[int, int, int]:
    return int(spur.symmetry), int(spur.num_tiles), int(spur.num_ret_tiles)


def _fit_one_inflated_binomial(
    histogram: np.ndarray,
) -> tuple[float, float]:
    """Return maximum-likelihood rho and p for a count histogram."""
    histogram = np.asarray(histogram, dtype=np.float64)
    m = histogram.size - 1
    if m <= 0 or histogram.sum() <= 0:
        raise ValueError("A non-empty count histogram with m >= 1 is required")

    counts = np.arange(m + 1, dtype=np.float64)
    mean_fraction = float((histogram * counts).sum() / (m * histogram.sum()))
    full_fraction = float(histogram[-1] / histogram.sum())
    initial_p = min(max(mean_fraction * 0.9, 1e-4), 1.0 - 1e-4)
    initial_rho = min(max(full_fraction * 0.8, 1e-4), 1.0 - 1e-4)

    log_choose = (
        gammaln(m + 1.0)
        - gammaln(counts + 1.0)
        - gammaln(m - counts + 1.0)
    )

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        rho, p = parameters
        log_binomial = (
            log_choose
            + counts * math.log(p)
            + (m - counts) * math.log1p(-p)
        )
        probabilities = (1.0 - rho) * np.exp(log_binomial)
        probabilities[-1] += rho
        return float(
            -(histogram * np.log(np.clip(probabilities, 1e-300, None))).sum()
        )

    result = minimize(
        negative_log_likelihood,
        np.array([initial_rho, initial_p]),
        method="L-BFGS-B",
        bounds=((1e-8, 1.0 - 1e-8), (1e-8, 1.0 - 1e-8)),
    )
    if not result.success:
        raise RuntimeError(f"Could not fit null vertex inness: {result.message}")
    rho, p = result.x
    return float(rho), float(p)


def find_rho_p(
    spur: Any,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> tuple[float, float, int, tuple[int, ...]]:
    """Fit rho and p from returned tiles for every mask under fixed seeds."""
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("At least one calibration seed is required")

    m = int(spur.V1)
    histogram = torch.zeros(m + 1, dtype=torch.long)
    mask_count = len(spur)
    batch_size = max(
        1,
        min(256, POINT_BUDGET // (int(spur.M) * m)),
    )

    with torch.no_grad():
        for seed in seeds:
            generator = torch.Generator(device=spur.device).manual_seed(seed)
            mask_indices = torch.arange(mask_count, device=spur.device)
            for start in range(0, mask_count, batch_size):
                batch = spur.sample_batch(
                    len(mask_indices[start : start + batch_size]),
                    mask_idx=mask_indices[start : start + batch_size],
                    generator=generator,
                )
                counts = batch["vertex_in"].sum(dim=-1).long().cpu().reshape(-1)
                histogram += torch.bincount(counts, minlength=m + 1)

    rho, p = _fit_one_inflated_binomial(histogram.numpy())
    return rho, p, mask_count * len(seeds), seeds


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def get_rho_p(
    spur: Any,
    csv_path: Path = CSV_PATH,
) -> tuple[float, float]:
    """Read a cached fit, or fit the active sampler and cache the result."""
    key = _key(spur)
    rows = _read_rows(csv_path)
    for row in rows:
        row_key = (
            int(row["symmetry"]),
            int(row["num_tiles"]),
            int(row["num_ret_tiles"]),
        )
        if row_key == key:
            return float(row["rho"]), float(row["p"])

    rho, p, samples, seeds = find_rho_p(spur)
    rows.append(
        {
            "symmetry": str(key[0]),
            "num_tiles": str(key[1]),
            "num_ret_tiles": str(key[2]),
            "rho": f"{rho:.12g}",
            "p": f"{p:.12g}",
            "samples": str(samples),
            "seeds": "|".join(map(str, seeds)),
        }
    )
    _write_rows_atomic(csv_path, rows)
    return rho, p


def sample_vertex_in_null(
    shape: tuple[int, int, int],
    rho: float,
    p: float,
    *,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample Boolean vertex states from a one-inflated Bernoulli model."""
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError(f"Expected positive (B,N,m) shape, got {shape}")
    if not 0.0 <= rho <= 1.0 or not 0.0 <= p <= 1.0:
        raise ValueError("rho and p must be probabilities")

    full = torch.rand(
        (*shape[:2], 1), device=device, generator=generator
    ) < rho
    probes = torch.rand(shape, device=device, generator=generator) < p
    return full | probes
