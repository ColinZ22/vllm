# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen3.8-Flash-Next model components."""

from .hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
