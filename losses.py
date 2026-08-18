"""The single inness-aware OTFM training loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from config import FlowConfig


@dataclass
class LossTerms:
    total: torch.Tensor
    geometry: torch.Tensor
    inness: torch.Tensor


def inness_weighted_velocity_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_inness: torch.Tensor,
    target_vertex_in: torch.Tensor,
    config: FlowConfig,
) -> LossTerms:
    """Weight geometry velocity by soft inness and supervise clean probes."""
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError(f"Expected target shape (B,N,3), got {target.shape}")
    if prediction.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"Prediction and target batch/tile shapes differ: "
            f"{prediction.shape} and {target.shape}"
        )
    if target_vertex_in.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"Expected vertex_in prefix {target.shape[:2]}, "
            f"got {target_vertex_in.shape}"
        )
    if prediction.shape[-1] != 3 + target_vertex_in.shape[-1]:
        raise ValueError(
            f"Expected prediction final dimension "
            f"{3 + target_vertex_in.shape[-1]}, got {prediction.shape[-1]}"
        )
    if target_inness.shape != target.shape[:2]:
        raise ValueError(f"Expected inness shape {target.shape[:2]}, got {target_inness.shape}")

    difference = prediction[..., :3] - target
    error = difference.abs() if config.loss == "l1" else difference.square()
    weights = target_inness.to(error.dtype)
    denominator = (3.0 * weights.sum()).clamp_min(torch.finfo(error.dtype).eps)
    xya_error = (error * weights.unsqueeze(-1)).sum() / denominator
    inness_error = F.binary_cross_entropy_with_logits(
        prediction[..., 3:],
        target_vertex_in.to(prediction.dtype),
    )
    total = xya_error + config.lambda_inness * inness_error
    return LossTerms(total=total, geometry=xya_error, inness=inness_error)
