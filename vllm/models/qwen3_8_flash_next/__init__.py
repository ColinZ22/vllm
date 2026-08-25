# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next model package."""

from typing import TYPE_CHECKING, Any

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

if TYPE_CHECKING:
    from .nvidia.model import (
        Qwen3_8FlashNextForCausalLM,
        Qwen3_8FlashNextForConditionalGeneration,
    )
    from .nvidia.mtp import Qwen3_8FlashNextMTP


def __getattr__(name: str) -> Any:
    if name in {
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
        "Qwen3_8FlashNextMTP",
    }:
        from vllm.platforms import current_platform

        if current_platform.is_xpu() or current_platform.is_tpu():
            raise NotImplementedError(
                "Qwen3.8-Flash-Next currently supports CUDA and ROCm only"
            )
        if current_platform.is_rocm():
            from .amd.model import (
                Qwen3_8FlashNextForCausalLM,
                Qwen3_8FlashNextForConditionalGeneration,
            )
            from .amd.mtp import Qwen3_8FlashNextMTP
        else:
            from .nvidia.model import (
                Qwen3_8FlashNextForCausalLM,
                Qwen3_8FlashNextForConditionalGeneration,
            )
            from .nvidia.mtp import Qwen3_8FlashNextMTP

        return {
            "Qwen3_8FlashNextForCausalLM": Qwen3_8FlashNextForCausalLM,
            "Qwen3_8FlashNextForConditionalGeneration": (
                Qwen3_8FlashNextForConditionalGeneration
            ),
            "Qwen3_8FlashNextMTP": Qwen3_8FlashNextMTP,
        }[name]
    raise AttributeError(name)


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextMTP",
]
