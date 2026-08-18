# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for the Qwen3.8-Flash-Next weight-free QSA path."""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton

_LOGITS_WORKSPACE_BYTES = 128 * 1024 * 1024


@triton.jit
def _qsa_mqa_paged_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_req_ptr,
    visible_lengths_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_table_req,
    stride_table_page,
    stride_logits_row,
    num_rows,
    num_columns,
    num_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    visible = tl.load(visible_lengths_ptr + row)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (row < num_rows)
        & (columns < num_columns)
        & (columns < visible)
        & (request >= 0)
        & (request < num_requests)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_req
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_pages)
    # physical_page * block stride can overflow int32 for large caches.
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)
    score = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + row * stride_logits_row + columns,
        tl.where(valid, score, -float("inf")),
        mask=(row < num_rows) & (columns < num_columns),
    )


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    sequence_length = tl.load(sequence_lengths_ptr + row)
    complete_blocks = tl.minimum((query_position + 1) // COMPRESS_RATIO, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_row,
    stride_indices_column,
    stride_table_req,
    stride_table_page,
    stride_output_row,
    stride_output_head,
    stride_output_dim,
    num_rows,
    num_cache_blocks,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_req_ptr + row)
    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    for start in tl.range(0, TOPK, BLOCK_N):
        columns = start + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
        page_offset = tl.maximum(logical_token, 0) % PAGE_SIZE
        valid = (
            (row < num_rows)
            & (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_requests - 1) * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(query, keys)
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    output = tl.where(
        normalizer[:, None] > 0,
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    tl.store(
        output_ptr
        + row * stride_output_row
        + (first_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(row < num_rows)
        & (head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_cache_ptr,
    rope_cache_ptr,
    raw_table_ptr,
    rope_table_ptr,
    token_to_req_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_block,
    stride_raw_token,
    stride_raw_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_raw_table_req,
    stride_raw_table_page,
    stride_rope_table_req,
    stride_rope_table_page,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_raw_blocks,
    num_rope_blocks,
    num_raw_requests,
    num_rope_requests,
    RAW_PAGE_SIZE: tl.constexpr,
    RAW_TABLE_WIDTH: tl.constexpr,
    ROPE_PAGE_SIZE: tl.constexpr,
    ROPE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_row = (
        (row < num_rows)
        & (request >= 0)
        & (request < num_raw_requests)
        & (request < num_rope_requests)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    if valid_row:
        for group_offset in tl.range(0, COMPRESS_RATIO):
            position = end_position - (COMPRESS_RATIO - 1 - group_offset)
            logical_page = position // RAW_PAGE_SIZE
            page_offset = position % RAW_PAGE_SIZE
            valid = logical_page < RAW_TABLE_WIDTH
            physical_page = tl.load(
                raw_table_ptr
                + request * stride_raw_table_req
                + tl.minimum(logical_page, RAW_TABLE_WIDTH - 1) * stride_raw_table_page,
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_raw_blocks)
            # physical_page * block stride can overflow int32 for large caches.
            values = tl.load(
                raw_cache_ptr
                + tl.maximum(physical_page, 0).to(tl.int64) * stride_raw_block
                + page_offset * stride_raw_token
                + dims * stride_raw_dim,
                mask=valid & (dims < HEAD_DIM),
                other=0.0,
            ).to(tl.float32)
            accumulator += values

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        rope_logical_page = first_position // ROPE_PAGE_SIZE
        rope_page_offset = first_position % ROPE_PAGE_SIZE
        valid_rope = valid_row & (rope_logical_page < ROPE_TABLE_WIDTH)
        rope_physical_page = tl.load(
            rope_table_ptr
            + tl.minimum(tl.maximum(request, 0), num_rope_requests - 1)
            * stride_rope_table_req
            + tl.minimum(rope_logical_page, ROPE_TABLE_WIDTH - 1)
            * stride_rope_table_page,
            mask=valid_rope,
            other=-1,
        )
        valid_rope &= (rope_physical_page >= 0) & (rope_physical_page < num_rope_blocks)
        rope_values = tl.load(
            rope_cache_ptr
            + tl.maximum(rope_physical_page, 0).to(tl.int64) * stride_rope_block
            + rope_page_offset * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_rope & (position_dims < 3),
            other=0,
        )
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            rope_values,
            mask=(row < num_rows) & (position_dims < 3),
        )
    else:
        first_position = tl.where(valid_row, first_position, 0)
        tl.store(
            first_positions_ptr
            + row * stride_positions_row
            + position_dims * stride_positions_dim,
            first_position,
            mask=(row < num_rows) & (position_dims < 3),
        )


def _validate_mqa(q: torch.Tensor) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_lengths: torch.Tensor,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> torch.Tensor:
    """Compute QSA scores directly from a paged compressed-key cache."""

    _validate_mqa(q)
    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA scoring requires CUDA and Triton")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if visible_lengths.shape != (q.shape[0],):
        raise ValueError("QSA visible lengths must match query rows")
    score_divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if score_divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0:
        raise ValueError("QSA score width must be non-negative")
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    if not q.shape[0] or not columns:
        return logits
    block_n = 32
    _qsa_mqa_paged_kernel[(q.shape[0], triton.cdiv(columns, block_n))](
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_lengths,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(3),
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        k_cache.shape[0],
        page_table.shape[0],
        float(score_divisor),
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        num_warps=4,
    )
    return logits


def qsa_relative_topk_cuda(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select ragged top-k indices relative to each row's start on-device."""

    if logits.ndim != 2 or topk <= 0:
        raise ValueError("QSA logits must be rank two and topk must be positive")
    if row_starts.shape != (logits.shape[0],) or row_ends.shape != (logits.shape[0],):
        raise ValueError("QSA row ranges must match logits rows")
    output = torch.full(
        (logits.shape[0], topk), -1, dtype=torch.int32, device=logits.device
    )
    candidate_count = min(topk, logits.shape[1])
    if not logits.shape[0] or not candidate_count:
        return output
    candidates = torch.topk(logits, candidate_count, dim=1).indices
    starts = row_starts.to(device=logits.device, dtype=torch.long)
    lengths = (row_ends - row_starts).to(device=logits.device, dtype=torch.long)
    ranks = torch.arange(candidate_count, device=logits.device).unsqueeze(0)
    relative = candidates - starts.unsqueeze(1)
    valid = (
        (ranks < lengths.unsqueeze(1))
        & (relative >= 0)
        & (relative < lengths.unsqueeze(1))
    )
    output[:, :candidate_count] = torch.where(
        valid, relative.to(torch.int32), torch.full_like(relative, -1).to(torch.int32)
    )
    return output


def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and compact the incomplete causal tail."""

    if not block_indices.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA expansion requires Triton")
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if sequence_lengths.shape != query_positions.shape:
        raise ValueError("QSA sequence lengths must match query positions")
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices,
        query_positions,
        sequence_lengths,
        out,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        block_indices.shape[0],
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score, select, and expand QSA indices without host synchronization."""

    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    columns = page_table.shape[1] * k_cache.shape[1]
    row_sequence_lengths = sequence_lengths.index_select(0, token_to_req.long())
    visible_blocks = torch.div(
        query_positions + 1, compress_ratio, rounding_mode="floor"
    ).to(torch.int32)
    max_blocks = torch.div(
        row_sequence_lengths, compress_ratio, rounding_mode="floor"
    ).to(torch.int32)
    visible_blocks = torch.minimum(visible_blocks, max_blocks)
    block_topk = token_topk // compress_ratio
    rows_per_chunk = max(1, _LOGITS_WORKSPACE_BYTES // max(columns * 4, 1))
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        row_slice = slice(row_start, row_end)
        logits = qsa_mqa_paged(
            q[row_slice],
            k_cache,
            page_table,
            token_to_req[row_slice],
            visible_blocks[row_slice],
        )
        starts = torch.zeros_like(visible_blocks[row_slice])
        blocks = qsa_relative_topk_cuda(
            logits, starts, visible_blocks[row_slice], block_topk
        )
        expand_qsa_block_indices_cuda(
            blocks,
            query_positions[row_slice],
            row_sequence_lengths[row_slice],
            compress_ratio,
            token_topk,
            out[row_slice],
        )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches."""

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires CUDA and Triton")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    if scale <= 0:
        raise ValueError("QSA softmax scale must be positive")
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(group_size))
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], k_cache.shape[2])](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=16,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_cache: torch.Tensor,
    raw_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool raw-key groups and load their packed or derived positions."""

    if not raw_cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if raw_cache.ndim != 4 or raw_cache.shape[2] != 1:
        raise ValueError("QSA raw cache has an invalid shape")
    if raw_block_table.ndim != 2:
        raise ValueError("QSA raw compression block table must be rank two")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != raw_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (not all(raw_cache.shape) or not all(raw_block_table.shape)):
        raise ValueError("QSA raw cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_cache.shape[3]), dtype=raw_cache.dtype, device=raw_cache.device
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_cache.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = raw_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_cache,
        rope_cache,
        raw_block_table,
        raw_block_table,
        token_to_req,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_cache.stride(0),
        raw_cache.stride(1),
        raw_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        raw_block_table.stride(0),
        raw_block_table.stride(1),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        raw_cache.shape[0],
        rope_cache.shape[0],
        raw_block_table.shape[0],
        raw_block_table.shape[0],
        RAW_PAGE_SIZE=raw_cache.shape[1],
        RAW_TABLE_WIDTH=raw_block_table.shape[1],
        ROPE_PAGE_SIZE=rope_cache.shape[1],
        ROPE_TABLE_WIDTH=raw_block_table.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_cache.shape[3],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_cache.shape[3]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "expand_qsa_block_indices_cuda",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_relative_topk_cuda",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
