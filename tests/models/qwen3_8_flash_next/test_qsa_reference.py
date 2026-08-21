# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen3_8_flash_next.common import qsa_cache
from vllm.models.qwen3_8_flash_next.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen3_8_flash_next.nvidia import (
    model as _qwen3_8_flash_next_model,  # noqa: F401
)
from vllm.models.qwen3_8_flash_next.nvidia.ops import qsa as qsa_ops
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


@requires_qsa_kernels
def test_qsa_side_metadata_marks_cudagraph_padding_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 1
    builder.storage_block_size = 64
    builder.token_to_req_buffer = torch.empty(16, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(16, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(16, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32, device=device)
    token_to_req = torch.tensor([0] * 4 + [1] * 4 + [2] * 4 + [0] * 4, device=device)
    common = SimpleNamespace(
        num_actual_tokens=16,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(list(range(12)) + [-1] * 4, device=device),
        block_table_tensor=torch.empty((4, 0), dtype=torch.int32, device=device),
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


@requires_qsa_kernels
def test_qsa_compressed_metadata_keeps_dummy_slots_inert() -> None:
    device = torch.device("cuda")
    builder = QSAMetadataBuilder.__new__(QSAMetadataBuilder)
    builder.compress_ratio = 4
    builder.storage_block_size = 16
    builder.token_to_req_buffer = torch.empty(8, dtype=torch.int32, device=device)
    builder.slot_mapping_buffer = torch.empty(8, dtype=torch.int64, device=device)
    builder.logical_positions_buffer = torch.empty(8, dtype=torch.int64, device=device)
    query_start_loc = torch.tensor([0, 8], dtype=torch.int32, device=device)
    token_to_req = torch.zeros(8, dtype=torch.int32, device=device)
    common = SimpleNamespace(
        num_actual_tokens=8,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([8], dtype=torch.int32, device=device),
        slot_mapping=torch.full((8,), -1, dtype=torch.int64, device=device),
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32, device=device),
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    metadata = builder.build(0, common)

    assert metadata.slot_mapping.tolist() == [-1] * 8


@requires_qsa_kernels
@pytest.mark.parametrize("compress_ratio", [1, 4])
def test_qsa_triton_metadata_matches_pytorch(
    compress_ratio: int,
) -> None:
    device = torch.device("cuda")
    num_tokens = 8
    query_start_loc = torch.tensor([0, 3, 3, 5], dtype=torch.int32, device=device)
    token_to_req = torch.tensor(
        [0, 0, 0, 2, 2, 0, 0, 0], dtype=torch.int32, device=device
    )
    block_table_storage = torch.tensor(
        [
            [4, -1, 8, -1, 12, -1],
            [1, -1, 2, -1, 3, -1],
            [7, -1, 9, -1, 11, -1],
        ],
        dtype=torch.int32,
        device=device,
    )
    common = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([10, 0, 20], dtype=torch.int32, device=device),
        slot_mapping=torch.tensor(
            [0, 1, -1, 3, 4, -1, -1, -1], dtype=torch.int64, device=device
        ),
        block_table_tensor=block_table_storage[:, ::2],
        token_to_req_indices=lambda buffer: buffer.copy_(token_to_req),
    )

    def make_buffers() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.empty(num_tokens, dtype=torch.int32, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
            torch.empty(num_tokens, dtype=torch.int64, device=device),
        )

    actual_buffers = make_buffers()
    actual = qsa_cache.build_qsa_metadata_triton(
        common,
        *actual_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
    )
    actual = tuple(tensor.clone() for tensor in actual)

    expected_buffers = make_buffers()
    expected = qsa_cache._build_qsa_metadata_torch(
        common,
        *expected_buffers,
        storage_block_size=2,
        compress_ratio=compress_ratio,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


@requires_qsa_kernels
def test_qsa_mqa_paged_matches_test_reference() -> None:
    torch.manual_seed(1)
    q = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(5, 4, 1, 16, device="cuda", dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1], [4, 2]], device="cuda", dtype=torch.int32)
    token_to_req = torch.tensor([0, 1, 0], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([4, 1, 6], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([8, 2], device="cuda", dtype=torch.int32)
    visible_lengths = torch.tensor([5, 2, 7], device="cuda", dtype=torch.int32)

    actual, actual_visible_blocks = qsa_ops.qsa_mqa_paged(
        q,
        cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio=1,
    )
    expected = _qsa_mqa_paged_reference(
        q, cache, page_table, token_to_req, visible_lengths
    )

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_visible_blocks, visible_lengths)


@requires_qsa_kernels
def test_qsa_block_expansion_matches_test_reference() -> None:
    blocks = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    query_positions = torch.tensor([5, 10], device="cuda")
    sequence_lengths = torch.tensor([6, 11], device="cuda")
    token_to_req = torch.tensor([0, 1], device="cuda", dtype=torch.int32)

    actual = qsa_ops.expand_qsa_block_indices_cuda(
        blocks,
        query_positions,
        sequence_lengths,
        token_to_req,
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
@pytest.mark.parametrize(
    ("num_rows", "num_query_heads", "num_kv_heads", "page_size"),
    [
        # Kernel-visible pages with --block-size 256 and hybrid-cache alignment.
        pytest.param(1, 24, 2, 1792, id="tp1_split64"),
        pytest.param(16, 12, 1, 1792, id="tp2_split32"),
        pytest.param(32, 6, 1, 1024, id="tp4_split8"),
        pytest.param(257, 6, 1, 1024, id="tp4_split4"),
        pytest.param(513, 6, 1, 1024, id="tp4_split1"),
    ],
)
def test_qsa_sparse_paged_attention_matches_test_reference(
    num_rows: int,
    num_query_heads: int,
    num_kv_heads: int,
    page_size: int,
) -> None:
    torch.manual_seed(2)
    head_dim = 256
    num_requests = 2
    num_selected_pages = 64
    # Keep the newest page outside the synthetic top-k as causal headroom.
    num_pages_per_request = num_selected_pages + 1
    num_cache_blocks = num_requests * num_pages_per_request
    indexer_budget = 2048
    indexer_compress_ratio = 4
    selection_width = indexer_budget + indexer_compress_ratio - 1
    q = torch.randn(
        num_rows, num_query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        num_cache_blocks,
        page_size,
        num_kv_heads,
        2 * head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache, v_cache = kv_cache.split(head_dim, dim=-1)
    block_table = (
        torch.randperm(num_cache_blocks, device="cuda")
        .reshape(num_requests, num_pages_per_request)
        .to(torch.int32)
    )
    rows_per_request = math.ceil(num_rows / num_requests)
    row_indices = torch.arange(num_rows, device="cuda", dtype=torch.int32)
    token_to_req = row_indices // rows_per_request
    request_row_counts = torch.tensor(
        [rows_per_request, num_rows - rows_per_request],
        device="cuda",
        dtype=torch.int32,
    )

    context_length = num_pages_per_request * page_size - 1
    block_topk = indexer_budget // indexer_compress_ratio
    compressed_blocks_per_page = page_size // indexer_compress_ratio
    selection = torch.arange(block_topk, device="cuda")
    selected_pages = selection % num_selected_pages
    selected_offsets = selection // num_selected_pages
    row_shifts = 2 * row_indices.unsqueeze(1)
    # Eight blocks per page; adjacent rows overlap by six of those eight.
    selected_offsets = (selected_offsets + row_shifts) % compressed_blocks_per_page
    block_indices = (selected_pages * compressed_blocks_per_page + selected_offsets).to(
        torch.int32
    )
    rows_within_request = row_indices % rows_per_request
    query_positions = (
        context_length - request_row_counts[token_to_req.long()] + rows_within_request
    ).to(torch.int64)
    sequence_lengths = torch.full(
        (num_requests,), context_length, device="cuda", dtype=torch.int32
    )
    logical_indices = qsa_ops.expand_qsa_block_indices_cuda(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        indexer_compress_ratio,
        indexer_budget,
    )
    assert logical_indices.shape == (num_rows, selection_width)
    scale = q.shape[-1] ** -0.5

    actual = qsa_ops.qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
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
    workspace_init,
) -> None:
    rows, keys, heads, head_dim = 65, 640, 4, 16
    token_topk, compress_ratio = 2048, 4
    torch.manual_seed(3)
    q = torch.randn(rows, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(40, 16, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    page_table = torch.randperm(40, device="cuda", dtype=torch.int32).unsqueeze(0)
    token_to_req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    query_positions = torch.full((rows,), 2559, device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([2560], device="cuda", dtype=torch.int32)
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
def test_qsa_selection_handles_no_complete_compressed_blocks(workspace_init) -> None:
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
        token_topk=2048,
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
