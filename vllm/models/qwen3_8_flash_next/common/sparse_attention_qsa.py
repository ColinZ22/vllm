# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference sparse attention for Qwen3.8-Flash-Next QSA-selected token indices."""

from __future__ import annotations

import torch
from torch import nn

from .qsa_cache import logical_to_physical_qsa_slots
from .qsa_reference import qsa_sparse_attention_reference


class QSAProductionKernelUnavailable(NotImplementedError):
    """Raised when a production CUDA QSA kernel was not selected."""


def qsa_logical_to_physical_slots(
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Map fixed-width request-relative QSA indices to main-KV cache slots."""

    if logical_indices.ndim != 2:
        raise ValueError("QSA logical indices must be [query_tokens, topk]")
    if token_to_req.ndim != 1 or token_to_req.numel() != logical_indices.shape[0]:
        raise ValueError("QSA token-to-request mapping must match query rows")
    requests = token_to_req.to(device=block_table.device, dtype=torch.long)
    if requests.numel() and (
        torch.any(requests < 0) or torch.any(requests >= sequence_lengths.numel())
    ):
        raise ValueError("QSA token-to-request mapping is out of range")
    row_lengths = sequence_lengths.to(block_table.device).index_select(0, requests)
    logical = logical_indices.to(device=block_table.device, dtype=torch.long)
    valid = (logical >= 0) & (logical < row_lengths.unsqueeze(1))
    slots = logical_to_physical_qsa_slots(
        block_table,
        requests.unsqueeze(1),
        logical.clamp_min(0),
        block_size,
    )
    return torch.where(valid, slots, torch.full_like(slots, -1)).to(torch.int32)


def qsa_sparse_attention_from_logical_indices(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_size: int,
    softmax_scale: float | None = None,
    *,
    allow_cuda_reference: bool = False,
) -> torch.Tensor:
    """Run explicit sparse GQA over flat main-cache K/V tensors.

    CUDA callers must opt into the slow reference path. This helper is for
    correctness tests; the model owner uses its explicit FlashAttention gather
    adapter and never presents this loop as a production kernel.
    """

    if q.is_cuda and not allow_cuda_reference:
        raise QSAProductionKernelUnavailable(
            "Qwen3.8-Flash-Next QSA has no fused CUDA sparse-attention kernel; pass "
            "allow_cuda_reference=True only for validation"
        )
    physical_slots = qsa_logical_to_physical_slots(
        logical_indices,
        block_table,
        token_to_req,
        sequence_lengths,
        block_size,
    )
    return qsa_sparse_attention_reference(
        q,
        key_cache,
        value_cache,
        physical_slots,
        softmax_scale,
    )


class QSAReferenceSparseAttention(nn.Module):
    """Module wrapper around the explicit QSA sparse-attention reference."""

    def __init__(
        self,
        block_size: int,
        softmax_scale: float | None = None,
        *,
        allow_cuda_reference: bool = False,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("QSA main-cache block size must be positive")
        self.block_size = block_size
        self.softmax_scale = softmax_scale
        self.allow_cuda_reference = allow_cuda_reference

    def forward(
        self,
        q: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        logical_indices: torch.Tensor,
        block_table: torch.Tensor,
        token_to_req: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return qsa_sparse_attention_from_logical_indices(
            q,
            key_cache,
            value_cache,
            logical_indices,
            block_table,
            token_to_req,
            sequence_lengths,
            self.block_size,
            self.softmax_scale,
            allow_cuda_reference=self.allow_cuda_reference,
        )


__all__ = [
    "QSAProductionKernelUnavailable",
    "QSAReferenceSparseAttention",
    "qsa_logical_to_physical_slots",
    "qsa_sparse_attention_from_logical_indices",
]
