# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next low-latency GEMM hook for AMD ROCm."""

import torch
from torch import nn


def enable_qwen38next_low_latency_gemm(
    module: nn.Module,
    dtype: torch.dtype,
) -> None:
    """Keep the standard vLLM linear methods on AMD ROCm."""

    del module, dtype
