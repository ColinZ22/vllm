# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-agnostic reference operations for Qwen3.8-Flash-Next QSA."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

QSA_PREFILL_LOGITS_BUDGET_BYTES = 128 * 1024 * 1024

_QSA_CONFIG_FIELDS = (
    "indexer_n_heads",
    "indexer_kv_heads",
    "indexer_head_dim",
    "indexer_budget",
    "indexer_compress_ratio",
)
_SUPPORTED_BLOCK_TOPK = frozenset((512, 2048))


def validate_qsa_config(config: Any) -> None:
    """Validate the QSA fields consumed by the reference and fused kernels."""

    missing = [
        field for field in _QSA_CONFIG_FIELDS if getattr(config, field, None) is None
    ]
    if missing:
        raise ValueError(f"QSA config is missing required fields: {missing}")

    values = {field: int(getattr(config, field)) for field in _QSA_CONFIG_FIELDS}
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"QSA config values must be positive: {values}")
    if values["indexer_kv_heads"] != 1:
        raise ValueError("QSA MQA requires indexer_kv_heads=1")

    budget = values["indexer_budget"]
    compress_ratio = values["indexer_compress_ratio"]
    if budget % compress_ratio:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    block_topk = budget // compress_ratio
    if block_topk not in _SUPPORTED_BLOCK_TOPK:
        raise ValueError(
            "QSA requires indexer_budget / indexer_compress_ratio to be "
            f"512 or 2048, got {block_topk}"
        )


def average_pool_qsa_keys(key_groups: torch.Tensor) -> torch.Tensor:
    """Average complete compression groups in FP32 and restore input dtype."""

    if key_groups.ndim != 4:
        raise ValueError(
            "QSA key groups must be [groups, ratio, kv_heads, head_dim], "
            f"got {tuple(key_groups.shape)}"
        )
    return key_groups.float().mean(dim=1).to(key_groups.dtype)


def build_qsa_row_ranges(
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor,
    query_sequence_ids: torch.Tensor,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequence-isolated ranges into a packed compressed-key tensor."""

    if compress_ratio <= 0:
        raise ValueError("compress_ratio must be positive")
    if sequence_lengths.ndim != 1:
        raise ValueError("sequence_lengths must be one-dimensional")
    if query_positions.ndim != 1 or query_sequence_ids.ndim != 1:
        raise ValueError("query positions and sequence ids must be one-dimensional")
    if query_positions.numel() != query_sequence_ids.numel():
        raise ValueError("query positions and sequence ids must have the same length")
    if sequence_lengths.numel() and torch.any(sequence_lengths < 0):
        raise ValueError("sequence lengths must be non-negative")

    device = sequence_lengths.device
    sequence_lengths = sequence_lengths.to(dtype=torch.int32)
    compressed_lengths = torch.div(
        sequence_lengths, compress_ratio, rounding_mode="floor"
    )
    compressed_cu_seqlens = F.pad(compressed_lengths.cumsum(0), (1, 0)).to(torch.int32)
    sequence_ids = query_sequence_ids.to(device=device, dtype=torch.long)
    if sequence_ids.numel() and (
        torch.any(sequence_ids < 0)
        or torch.any(sequence_ids >= sequence_lengths.numel())
    ):
        raise ValueError("query sequence ids are out of range")

    row_starts = compressed_cu_seqlens.index_select(0, sequence_ids)
    visible_blocks = torch.div(
        query_positions.to(device=device, dtype=torch.int32) + 1,
        compress_ratio,
        rounding_mode="floor",
    ).clamp_min_(0)
    max_blocks = compressed_lengths.index_select(0, sequence_ids)
    row_ends = row_starts + torch.minimum(visible_blocks, max_blocks)
    return row_starts, row_ends, compressed_cu_seqlens


def _validate_mqa_inputs(q: torch.Tensor, k: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError(
            f"QSA requires q [tokens, heads, head_dim], got {tuple(q.shape)}"
        )
    if k.ndim != 3 or k.shape[1] != 1 or k.shape[2] <= 0:
        raise ValueError(
            f"QSA MQA requires k [tokens, 1, head_dim], got {tuple(k.shape)}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if q.device != k.device:
        raise ValueError("QSA query and key tensors must be on the same device")


def qsa_weight_free_mqa_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Compute ``sum_heads(relu(q @ k)) / scale`` in FP32."""

    _validate_mqa_inputs(q, k)
    scale = math.sqrt(q.shape[-1]) if score_scale is None else score_scale
    if scale <= 0:
        raise ValueError("score_scale must be positive")
    scores = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    return torch.relu(scores).sum(dim=-1) / scale


def _validate_row_ranges(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        raise ValueError("QSA row starts and ends must be one-dimensional")
    if row_starts.numel() != rows or row_ends.numel() != rows:
        raise ValueError("QSA row ranges must have one entry per query")
    starts = row_starts.to(dtype=torch.long)
    ends = row_ends.to(dtype=torch.long)
    if rows and (
        torch.any(starts < 0) or torch.any(starts > ends) or torch.any(ends > columns)
    ):
        raise ValueError("QSA row ranges must satisfy 0 <= start <= end <= keys")
    return starts, ends


def qsa_mqa_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Score packed variable-length prefill rows without crossing sequences."""

    logits = qsa_weight_free_mqa_scores(q, k, score_scale)
    starts, ends = _validate_row_ranges(row_starts, row_ends, q.shape[0], k.shape[0])
    columns = torch.arange(k.shape[0], device=q.device).unsqueeze(0)
    valid = (columns >= starts.to(q.device).unsqueeze(1)) & (
        columns < ends.to(q.device).unsqueeze(1)
    )
    return logits.masked_fill(~valid, -torch.inf)


def qsa_mqa_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    context_lens: torch.Tensor,
    max_model_len: int,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Score a variable-length paged MQA cache in FP32."""

    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA decode query must be [batch, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA decode cache must be [pages, page_size, 1, head_dim]")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("QSA query and key head dimensions must match")
    if page_table.ndim != 2 or page_table.shape[0] != q.shape[0]:
        raise ValueError("QSA page table must have one row per query")
    if context_lens.ndim != 1 or context_lens.numel() != q.shape[0]:
        raise ValueError("QSA context lengths must have one entry per query")
    if max_model_len < 0:
        raise ValueError("max_model_len must be non-negative")
    if q.device != k_cache.device or q.device != page_table.device:
        raise ValueError("QSA decode tensors must be on the same device")

    scale = math.sqrt(q.shape[-1]) if score_scale is None else score_scale
    if scale <= 0:
        raise ValueError("score_scale must be positive")
    batch = q.shape[0]
    page_size = k_cache.shape[1]
    total = page_table.shape[1] * page_size
    pages = page_table.long().clamp_min(0)
    if pages.numel() and torch.any(pages >= k_cache.shape[0]):
        raise ValueError("QSA page table contains an out-of-range physical page")
    gathered = k_cache[pages.reshape(-1), :, 0].reshape(batch, total, q.shape[-1])
    scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    scores = torch.relu(scores).sum(dim=-1) / scale
    positions = torch.arange(total, device=q.device).unsqueeze(0)
    scores.masked_fill_(
        positions >= context_lens.to(device=q.device).unsqueeze(1), -torch.inf
    )

    logits = torch.full(
        (batch, max_model_len), -torch.inf, dtype=torch.float32, device=q.device
    )
    copy_len = min(total, max_model_len)
    if copy_len:
        logits[:, :copy_len] = scores[:, :copy_len]
    return logits


def qsa_relative_topk(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select top compressed blocks and return row-relative indices."""

    if logits.ndim != 2:
        raise ValueError("QSA logits must be a two-dimensional tensor")
    if topk <= 0:
        raise ValueError("topk must be positive")
    starts, ends = _validate_row_ranges(
        row_starts, row_ends, logits.shape[0], logits.shape[1]
    )
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(starts[row])
        length = int(ends[row] - starts[row])
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    """Expand compressed blocks and append the visible incomplete tail."""

    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    block_topk = (token_topk + compress_ratio - 1) // compress_ratio
    final_topk = token_topk + compress_ratio - 1
    if block_indices.ndim != 2 or block_indices.shape[1] != block_topk:
        raise ValueError(
            f"expected block indices [rows, {block_topk}], "
            f"got {tuple(block_indices.shape)}"
        )
    rows = block_indices.shape[0]
    if query_positions.numel() != rows or sequence_lengths.numel() != rows:
        raise ValueError("query positions and sequence lengths must match top-k rows")

    device = block_indices.device
    blocks = block_indices.long()
    offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]

    query_positions = query_positions.to(device=device, dtype=torch.long)
    sequence_lengths = sequence_lengths.to(device=device, dtype=torch.long)
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    visible_tokens = query_positions + 1
    tail_start = (
        torch.div(visible_tokens, compress_ratio, rounding_mode="floor")
        * compress_ratio
    )
    tail_count = visible_tokens - tail_start
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(final_topk, device=device).unsqueeze(0).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + final_topk)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def qsa_sparse_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_slots: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Compute sparse grouped-query attention over selected physical slots."""

    if q.ndim != 3 or k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("q, k_cache and v_cache must be rank-3 tensors")
    if token_slots.ndim != 2 or token_slots.shape[0] != q.shape[0]:
        raise ValueError("token_slots must have one row per query token")
    if k_cache.shape != v_cache.shape:
        raise ValueError("QSA key and value caches must have the same shape")
    if q.shape[-1] != k_cache.shape[-1]:
        raise ValueError("QSA query and cache head dimensions must match")
    if not k_cache.shape[1] or q.shape[1] % k_cache.shape[1]:
        raise ValueError("QSA query heads must be divisible by KV heads")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("QSA attention tensors must be on the same device")

    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if scale <= 0:
        raise ValueError("softmax_scale must be positive")
    if q.shape[0] == 0:
        return torch.empty_like(q)

    outputs = []
    repeats = q.shape[1] // k_cache.shape[1]
    for row in range(q.shape[0]):
        slots = token_slots[row, token_slots[row] >= 0].long()
        if slots.numel() == 0:
            outputs.append(torch.zeros_like(q[row]))
            continue
        if torch.any(slots >= k_cache.shape[0]):
            raise ValueError("QSA token slots contain an out-of-range cache slot")
        keys = k_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        values = v_cache.index_select(0, slots).repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hk,khd->hd", probabilities, values.float()).to(q.dtype)
        )
    return torch.stack(outputs)


def qsa_prefill_row_chunk_size(
    rows: int,
    keys: int,
    heads: int,
    logits_budget_bytes: int = QSA_PREFILL_LOGITS_BUDGET_BYTES,
) -> int:
    """Choose a row tile that bounds the padded FP32 logits workspace."""

    if rows < 0 or keys < 0 or heads <= 0:
        raise ValueError(
            "rows and keys must be non-negative and heads must be positive"
        )
    if logits_budget_bytes <= 0:
        raise ValueError("logits_budget_bytes must be positive")
    if rows == 0 or keys == 0:
        return max(rows, 1)

    block_q = max(1, 128 // heads)
    bytes_per_row = keys * torch.float32.itemsize
    max_padded_rows = max(block_q, logits_budget_bytes // bytes_per_row)
    max_padded_rows = max(block_q, max_padded_rows // block_q * block_q)
    return min(rows, max_padded_rows)


def select_qsa_prefill_tokens_reference(
    q: torch.Tensor,
    compressed_keys: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    logits_budget_bytes: int = QSA_PREFILL_LOGITS_BUDGET_BYTES,
) -> torch.Tensor:
    """Select prefill tokens while bounding the temporary FP32 logits matrix."""

    if token_topk <= 0 or compress_ratio <= 0:
        raise ValueError("token_topk and compress_ratio must be positive")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    _validate_mqa_inputs(q, compressed_keys)
    starts, ends = _validate_row_ranges(
        row_starts, row_ends, q.shape[0], compressed_keys.shape[0]
    )
    if query_positions.numel() != q.shape[0]:
        raise ValueError("query positions must have one entry per query")
    if sequence_lengths.numel() != q.shape[0]:
        raise ValueError("sequence lengths must have one entry per query")

    rows = q.shape[0]
    output = torch.empty(
        (rows, token_topk + compress_ratio - 1),
        dtype=torch.int32,
        device=q.device,
    )
    if rows == 0:
        return output

    block_topk = token_topk // compress_ratio
    row_chunk_size = qsa_prefill_row_chunk_size(
        rows,
        compressed_keys.shape[0],
        q.shape[1],
        logits_budget_bytes,
    )
    for row_start in range(0, rows, row_chunk_size):
        row_end = min(row_start + row_chunk_size, rows)
        row_slice = slice(row_start, row_end)
        if compressed_keys.shape[0] == 0:
            block_indices = torch.full(
                (row_end - row_start, block_topk),
                -1,
                dtype=torch.int32,
                device=q.device,
            )
        else:
            logits = qsa_mqa_prefill_reference(
                q[row_slice],
                compressed_keys,
                starts[row_slice],
                ends[row_slice],
            )
            block_indices = qsa_relative_topk(
                logits, starts[row_slice], ends[row_slice], block_topk
            )
        output[row_slice].copy_(
            expand_qsa_block_indices(
                block_indices,
                query_positions[row_slice],
                sequence_lengths[row_slice],
                compress_ratio,
                token_topk,
            )
        )
    return output


__all__ = [
    "QSA_PREFILL_LOGITS_BUDGET_BYTES",
    "average_pool_qsa_keys",
    "build_qsa_row_ranges",
    "expand_qsa_block_indices",
    "qsa_mqa_decode_reference",
    "qsa_mqa_prefill_reference",
    "qsa_prefill_row_chunk_size",
    "qsa_relative_topk",
    "qsa_sparse_attention_reference",
    "qsa_weight_free_mqa_scores",
    "select_qsa_prefill_tokens_reference",
    "validate_qsa_config",
]
