"""Focused checks for null vertex-inness fitting and caching."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from null_inness import (  # noqa: E402
    _fit_one_inflated_binomial,
    get_rho_p,
    sample_vertex_in_null,
)


class FakeSpur:
    symmetry = 6
    num_tiles = 96
    num_ret_tiles = 120


def check_fit() -> None:
    m = 7
    expected_rho = 0.35
    expected_p = 0.68
    probabilities = np.array(
        [
            (1.0 - expected_rho)
            * math.comb(m, count)
            * expected_p**count
            * (1.0 - expected_p) ** (m - count)
            for count in range(m + 1)
        ]
    )
    probabilities[-1] += expected_rho
    histogram = np.maximum(1, np.rint(probabilities * 1_000_000)).astype(int)
    rho, p = _fit_one_inflated_binomial(histogram)
    assert abs(rho - expected_rho) < 1e-3
    assert abs(p - expected_p) < 1e-3


def check_cache() -> None:
    with tempfile.TemporaryDirectory(prefix="null-inness-") as temporary:
        path = Path(temporary) / "fits.csv"
        fitted = (0.4, 0.7, 7000, (0, 1, 2, 3, 4))
        with patch("null_inness.find_rho_p", return_value=fitted) as fit:
            assert get_rho_p(FakeSpur(), path) == fitted[:2]
            fit.assert_called_once()
        with patch(
            "null_inness.find_rho_p",
            side_effect=AssertionError("cache miss"),
        ):
            assert get_rho_p(FakeSpur(), path) == fitted[:2]


def check_sampling() -> None:
    shape = (32, 20, 7)
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)
    first = sample_vertex_in_null(
        shape,
        0.4,
        0.7,
        device=torch.device("cpu"),
        generator=first_generator,
    )
    second = sample_vertex_in_null(
        shape,
        0.4,
        0.7,
        device=torch.device("cpu"),
        generator=second_generator,
    )
    assert first.dtype == torch.bool
    assert first.shape == shape
    assert torch.equal(first, second)

    all_full = sample_vertex_in_null(
        shape,
        1.0,
        0.0,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )
    assert all_full.all()


def main() -> None:
    check_fit()
    check_cache()
    check_sampling()
    print("Null vertex-inness checks passed")


if __name__ == "__main__":
    main()
