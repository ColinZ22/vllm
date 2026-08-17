# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Common Qwen3.8-Flash-Next model components."""

from .hyperconnection import (
    HYPERCONNECTION_CLASS_DICT,
    GatedResidualSimple,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

__all__ = [
    "GatedResidualSimple",
    "GroupedGemmaRMSNorm",
    "HYPERCONNECTION_CLASS_DICT",
    "HyperConnectionBase",
    "HyperConnectionConfig",
]
