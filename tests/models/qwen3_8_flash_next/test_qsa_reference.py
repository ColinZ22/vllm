# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen3_8_flash_next.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen3_8_flash_next.nvidia import (
    model as _qwen3_8_flash_next_model,  # noqa: F401
)
from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as qsa_ops
from vllm.models.qwen3_8_flash_next.nvidia.qsa import qsa_token_to_request
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

requires_qsa_kernels = pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA kernels require CUDA and Triton",
)


def _qsa_mqa_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
) -> torch.Tensor:
    pages = page_table.index_select(0, token_to_req.long()).long()
    keys = k_cache[pages, :, 0, :].flatten(1, 2)
    scores = torch.einsum("rhd,rnd->rnh", q.float(), keys.float())
    logits = torch.relu(scores).sum(dim=-1) / math.sqrt(q.shape[-1])
    positions = torch.arange(keys.shape[1], device=q.device).unsqueeze(0)
    return logits.masked_fill(positions >= visible_lengths.unsqueeze(1), -torch.inf)


def _qsa_relative_topk_reference(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    for row in range(logits.shape[0]):
        start = int(row_starts[row].item())
        length = int((row_ends[row] - row_starts[row]).item())
        width = min(length, topk)
        if width:
            output[row, :width] = torch.topk(
                logits[row, start : start + length], width
            ).indices.to(torch.int32)
    return output


def _expand_qsa_indices_reference(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
) -> torch.Tensor:
    rows = block_indices.shape[0]
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    offsets = torch.arange(compress_ratio, device=block_indices.device)
    blocks = block_indices.long()
    expanded = blocks.unsqueeze(-1) * compress_ratio + offsets
    expanded = torch.where(
        blocks.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1)
    ).reshape(rows, block_topk * compress_ratio)
    expanded = expanded[:, :token_topk]
    expanded = torch.where(
        (expanded >= 0) & (expanded < sequence_lengths.unsqueeze(1)),
        expanded,
        torch.full_like(expanded, -1),
    )

    tail_offsets = torch.arange(compress_ratio - 1, device=block_indices.device)
    visible_tokens = query_positions + 1
    tail_start = visible_tokens // compress_ratio * compress_ratio
    tail = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
    tail_count = (visible_tokens - tail_start).unsqueeze(1)
    tail_valid = (tail_offsets.unsqueeze(0) < tail_count) & (
        tail < sequence_lengths.unsqueeze(1)
    )
    tail = torch.where(tail_valid, tail, torch.full_like(tail, -1))

    result = torch.cat((expanded, tail), dim=1)
    order = torch.arange(output_width, device=result.device).expand(rows, -1)
    sort_key = torch.where(result >= 0, order, order + output_width)
    return result.gather(1, torch.argsort(sort_key, dim=1, stable=True)).to(torch.int32)


def _qsa_select_paged_tokens_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
) -> torch.Tensor:
    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.minimum(
        (query_positions + 1) // compress_ratio,
        row_sequence_lengths // compress_ratio,
    ).to(torch.int32)
    logits = _qsa_mqa_paged_reference(
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_blocks,
    )
    starts = torch.zeros_like(visible_blocks)
    blocks = _qsa_relative_topk_reference(
        logits,
        starts,
        visible_blocks,
        token_topk // compress_ratio,
    )
    return _expand_qsa_indices_reference(
        blocks,
        query_positions,
        row_sequence_lengths,
        compress_ratio,
        token_topk,
    )


def _qsa_sparse_paged_attention_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    output = torch.zeros_like(q)
    repeats = q.shape[1] // k_cache.shape[2]
    page_size = k_cache.shape[1]
    for row in range(q.shape[0]):
        logical = logical_indices[row]
        logical = logical[logical >= 0].long()
        if not logical.numel():
            continue
        request = token_to_req[row].long()
        pages = block_table[request, logical // page_size].long()
        offsets = logical % page_size
        keys = k_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        values = v_cache[pages, offsets].repeat_interleave(repeats, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float())
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        output[row] = torch.einsum("hk,khd->hd", probabilities, values.float()).to(
            q.dtype
        )
    return output


def test_qsa_request_mapping_marks_cudagraph_padding_inert() -> None:
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)

    token_to_req = qsa_token_to_request(query_start_loc, num_tokens=16)

    assert token_to_req.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4


def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 1
    builder.storage_block_size = 64
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32)
    builder.arange_buffer = torch.arange(16, dtype=torch.int64)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)
    token_to_req = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [0] * 4)
    common = SimpleNamespace(
        num_actual_tokens=16,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32),
        slot_mapping=torch.tensor(list(range(12)) + [-1] * 4),
        block_table_tensor=torch.empty((4, 0), dtype=torch.int32),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.logical_positions.tolist() == [
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        64,
        65,
        66,
        67,
        -1,
        -1,
        -1,
        -1,
    ]
    assert metadata.slot_mapping.tolist() == list(range(12)) + [-1] * 4


def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.storage_block_size = 16
    builder.token_to_req_buffer = torch.empty(8, dtype=torch.int32)
    builder.arange_buffer = torch.arange(8, dtype=torch.int64)
    builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64)
    builder.logical_positions_buffer = torch.empty(8, dtype=torch.int64)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
    token_to_req = torch.zeros(8, dtype=torch.int32)
    common = SimpleNamespace(
        num_actual_tokens=8,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.slot_mapping.tolist() == [-1] * 8


@requires_qsa_kernels
def test_qsa_mqa_paged_matches_test_reference() -> None:
    torch.manual_seed(1)
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(5, 4, 1, 16, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1], [4, 2]], device="cuda", dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    visible_lengths = torch.tensor([5, 2, 7], device="cuda", dtype=torch.int32)

    actual = qsa_ops.qsa_mqa_paged(q, cache, page_table, token_to_req, visible_lengths)
    expected = _qsa_mqa_paged_reference(
        q, cache, page_table, token_to_req, visible_lengths
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@requires_qsa_kernels
def test_qsa_relative_topk_is_request_relative() -> None:
    logits = torch.full((2, 9), -torch.inf, device="cuda")
    logits[0, 3:7] = torch.tensor([1, 9, 2, 8], device="cuda")
    logits[1, 1:4] = torch.tensor([6, 7, 5], device="cuda")
    starts = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    ends = torch.tensor([7, 4], device="cuda", dtype=torch.int32)

    actual = qsa_ops.qsa_relative_topk_cuda(logits, starts, ends, topk=4)
    expected = _qsa_relative_topk_reference(logits, starts, ends, topk=4)

    torch.testing.assert_close(actual, expected)


@requires_qsa_kernels
def test_qsa_block_expansion_matches_test_reference() -> None:
    blocks = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([5, 10], device="cuda")
    sequence_lengths = torch.tensor([6, 11], device="cuda")

    actual = qsa_ops.expand_qsa_block_indices_cuda(
        blocks,
        query_positions,
        sequence_lengths,
        compress_ratio=4,
        token_topk=8,
    )
    expected = _expand_qsa_indices_reference(
        blocks,
        query_positions,
        sequence_lengths,
        compress_ratio=4,
        token_topk=8,
    )

    torch.testing.assert_close(actual, expected)


@requires_qsa_kernels
def test_qsa_sparse_paged_attention_matches_test_reference() -> None:
    torch.manual_seed(2)
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(4, 4, 2, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    logical_indices = torch.tensor(
        [[0, 2, 4, 6], [1, 3, 5, -1], [-1, -1, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    scale = q.shape[-1] ** -0.5

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        scale,
    )
    expected = _qsa_sparse_paged_attention_reference(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        scale,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@requires_qsa_kernels
def test_qsa_selection_chunks_workspace_and_matches_test_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, keys, heads, head_dim = 65, 64, 4, 16
    token_topk, compress_ratio = 8, 4
    torch.manual_seed(3)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(4, 16, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1, 0, 2]], device="cuda", dtype=torch.int32)
    token_to_req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    query_positions = torch.full((rows,), 255, device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([256], device="cuda", dtype=torch.int32)
    monkeypatch.setattr(qsa_ops, "_LOGITS_WORKSPACE_BYTES", 32 * keys * 4)
    original_score = qsa_ops.qsa_mqa_paged
    scored_row_counts = []

    def record_score(query: torch.Tensor, *args, **kwargs):
        scored_row_counts.append(query.shape[0])
        return original_score(query, *args, **kwargs)

    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", record_score)

    actual = qsa_ops.qsa_select_paged_tokens(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )
    expected = _qsa_select_paged_tokens_reference(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
    )

    torch.testing.assert_close(actual.sort().values, expected.sort().values)
    assert scored_row_counts == [32, 32, 1]


@requires_qsa_kernels
def test_qsa_selection_handles_no_complete_compressed_blocks() -> None:
    q = torch.zeros(2, 4, 8, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(1, 16, 1, 8, device="cuda", dtype=torch.bfloat16)
    page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    token_to_req = torch.zeros(2, device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([3], device="cuda", dtype=torch.int32)

    selected = qsa_ops.qsa_select_paged_tokens(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        token_topk=8,
        compress_ratio=4,
    )

    assert selected[0, :2].tolist() == [0, 1]
    assert selected[1, :3].tolist() == [0, 1, 2]
    assert torch.all(selected[0, 2:] == -1)
    assert torch.all(selected[1, 3:] == -1)


@requires_qsa_kernels
def test_qsa_cache_store_and_compression_match_test_reference() -> None:
    rows = torch.arange(64, device="cuda", dtype=torch.float32)
    rows = rows.reshape(8, 1, 8).to(torch.bfloat16)
    raw_cache = torch.zeros(2, 4, 1, 8, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([4, 5, 6, 7, 0, 1, 2, 3], device="cuda")
    qsa_ops.qsa_store_cache_rows(raw_cache, slots, rows)

    block_table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    token_to_req = torch.zeros(2, device="cuda", dtype=torch.int32)
    logical_positions = torch.tensor([3, 7], device="cuda", dtype=torch.int32)
    compressed_slots = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    pooled, first_positions = qsa_ops.qsa_compress_groups_with_ratio(
        raw_cache,
        block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        compress_ratio=4,
    )

    expected = rows.reshape(2, 4, 1, 8).float().mean(dim=1).to(torch.bfloat16)
    expected_positions = torch.tensor([[0, 0, 0], [4, 4, 4]], device="cuda")
    torch.testing.assert_close(pooled, expected)
    torch.testing.assert_close(first_positions, expected_positions)
