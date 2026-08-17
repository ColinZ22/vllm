# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen3_8_flash_next.common import qsa_reference as qsa
from vllm.models.qwen3_8_flash_next.common.qsa_cache import QSAMetadataBuilder
from vllm.models.qwen3_8_flash_next.nvidia import (
    model as _qwen3_8_flash_next_model,  # noqa: F401
)
from vllm.models.qwen3_8_flash_next.nvidia.qsa import (
    qsa_query_positions,
    qsa_token_to_request,
)


def _valid_config(**overrides):
    values = {
        "indexer_n_heads": 8,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 64,
        "indexer_budget": 1024,
        "indexer_compress_ratio": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qsa_config_validation_accepts_kernel_shapes():
    qsa.validate_qsa_config(_valid_config())
    qsa.validate_qsa_config(
        _valid_config(indexer_budget=8192, indexer_compress_ratio=4)
    )


def test_qsa_request_mapping_marks_cudagraph_padding_inert():
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)

    token_to_req = qsa_token_to_request(query_start_loc, num_tokens=16)
    logical_positions = qsa_query_positions(
        query_start_loc,
        token_to_req,
        sequence_lengths=torch.tensor([68, 68, 68, 0], dtype=torch.int32),
    )

    assert token_to_req.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert logical_positions.tolist() == [
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


def test_qsa_side_metadata_marks_cudagraph_padding_inert():
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


def test_qsa_compressed_metadata_keeps_dummy_slots_inert():
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


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (SimpleNamespace(), "missing required fields"),
        (_valid_config(indexer_n_heads=0), "must be positive"),
        (_valid_config(indexer_kv_heads=2), "indexer_kv_heads=1"),
        (
            _valid_config(indexer_budget=1025, indexer_compress_ratio=2),
            "must be divisible",
        ),
        (
            _valid_config(indexer_budget=1024, indexer_compress_ratio=4),
            "512 or 2048",
        ),
    ],
)
def test_qsa_config_validation_rejects_unsupported_values(config, match):
    with pytest.raises(ValueError, match=match):
        qsa.validate_qsa_config(config)


def test_qsa_compression_average_matches_fp32_training_reference():
    keys = torch.arange(2 * 4 * 2 * 8, dtype=torch.float32).reshape(2, 4, 2, 8)
    expected = keys.mean(dim=1).to(torch.bfloat16)

    actual = qsa.average_pool_qsa_keys(keys.to(torch.bfloat16))

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_row_ranges_isolate_ragged_sequences():
    starts, ends, compressed_cu = qsa.build_qsa_row_ranges(
        sequence_lengths=torch.tensor([10, 7], dtype=torch.int32),
        query_positions=torch.tensor([8, 9, 4, 6], dtype=torch.int32),
        query_sequence_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        compress_ratio=4,
    )

    assert compressed_cu.tolist() == [0, 2, 3]
    assert starts.tolist() == [0, 0, 2, 2]
    assert ends.tolist() == [2, 2, 3, 3]


def test_weight_free_mqa_uses_relu_head_sum_and_row_masks():
    torch.manual_seed(1)
    q = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    k = torch.randn(5, 1, 8, dtype=torch.bfloat16)
    starts = torch.tensor([0, 0, 2, 4], dtype=torch.int32)
    ends = torch.tensor([2, 1, 4, 4], dtype=torch.int32)

    actual = qsa.qsa_mqa_prefill_reference(q, k, starts, ends)
    expected = torch.einsum("mhd,nd->mnh", q.float(), k[:, 0].float())
    expected = torch.relu(expected).sum(dim=-1) / (8**0.5)
    columns = torch.arange(5).unsqueeze(0)
    expected.masked_fill_(
        (columns < starts.unsqueeze(1)) | (columns >= ends.unsqueeze(1)),
        -torch.inf,
    )

    torch.testing.assert_close(actual, expected)
    assert torch.isneginf(actual[0, 2:]).all()
    assert torch.isneginf(actual[2, :2]).all()
    assert torch.isneginf(actual[3]).all()


def test_qsa_decode_mqa_reads_each_paged_sequence():
    torch.manual_seed(2)
    q = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    cache = torch.randn(5, 3, 1, 8, dtype=torch.bfloat16)
    page_table = torch.tensor([[3, 1], [4, 2]], dtype=torch.int32)
    context_lens = torch.tensor([5, 2], dtype=torch.int32)

    actual = qsa.qsa_mqa_decode_reference(
        q, cache, page_table, context_lens, max_model_len=8
    )
    gathered = cache[page_table.long(), :, 0].reshape(2, 6, 8)
    expected_scores = torch.einsum("bhd,bnd->bnh", q.float(), gathered.float())
    expected_scores = torch.relu(expected_scores).sum(dim=-1) / (8**0.5)
    positions = torch.arange(6).unsqueeze(0)
    expected_scores.masked_fill_(positions >= context_lens.unsqueeze(1), -torch.inf)
    expected = torch.full((2, 8), -torch.inf)
    expected[:, :6] = expected_scores

    torch.testing.assert_close(actual, expected)


def test_qsa_topk_is_relative_and_cannot_cross_row_ranges():
    logits = torch.full((2, 9), -10.0)
    logits[0, 0] = 1000
    logits[0, 3] = 1
    logits[0, 4] = 9
    logits[0, 5] = 2
    logits[0, 6] = 8
    logits[1, 8] = 1000
    logits[1, 1] = 6
    logits[1, 2] = 7
    logits[1, 3] = 5
    starts = torch.tensor([3, 1], dtype=torch.int32)
    ends = torch.tensor([7, 4], dtype=torch.int32)

    indices = qsa.qsa_relative_topk(logits, starts, ends, topk=4)

    assert indices[0].tolist() == [1, 3, 2, 0]
    assert indices[1, :3].tolist() == [1, 0, 2]
    assert indices[1, 3].item() == -1


def test_qsa_block_expansion_adds_only_the_incomplete_tail():
    blocks = torch.tensor([[0, -1], [1, 0]], dtype=torch.int32)

    result = qsa.expand_qsa_block_indices(
        blocks,
        query_positions=torch.tensor([5, 10]),
        sequence_lengths=torch.tensor([6, 11]),
        compress_ratio=4,
        token_topk=8,
    )

    assert result.shape == (2, 11)
    assert result[0, :6].tolist() == [0, 1, 2, 3, 4, 5]
    assert sorted(result[1].tolist()) == list(range(11))
    assert torch.all(result[0, 6:] == -1)


def test_qsa_sparse_attention_matches_explicit_gqa_and_handles_empty_rows():
    torch.manual_seed(3)
    q = torch.randn(3, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(7, 2, 16, dtype=torch.bfloat16)
    v = torch.randn(7, 2, 16, dtype=torch.bfloat16)
    slots = torch.tensor(
        [[0, 2, 4, 6], [1, 3, 5, -1], [-1, -1, -1, -1]],
        dtype=torch.int32,
    )

    actual = qsa.qsa_sparse_attention_reference(q, k, v, slots)
    expected_rows = []
    for row, selected in enumerate((slots[0], slots[1, :3])):
        keys = k[selected.long()].repeat_interleave(2, dim=1)
        values = v[selected.long()].repeat_interleave(2, dim=1)
        scores = torch.einsum("hd,khd->hk", q[row].float(), keys.float()) / 4
        expected_rows.append(
            torch.einsum("hk,khd->hd", scores.softmax(-1), values.float()).to(
                torch.bfloat16
            )
        )
    expected_rows.append(torch.zeros_like(q[2]))

    torch.testing.assert_close(actual, torch.stack(expected_rows), rtol=2e-2, atol=2e-2)


def test_qsa_prefill_selection_microchunks_128_mib_workspace(monkeypatch):
    rows, keys, heads, head_dim = 65, 64, 4, 8
    token_topk, compress_ratio = 8, 4
    budget = 32 * keys * torch.float32.itemsize
    torch.manual_seed(4)
    query = torch.randn(rows, heads, head_dim, dtype=torch.bfloat16)
    compressed_keys = torch.randn(keys, 1, head_dim, dtype=torch.bfloat16)
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full((rows,), keys, dtype=torch.int32)
    positions = torch.full((rows,), keys * compress_ratio - 1, dtype=torch.long)
    sequence_lengths = torch.full((rows,), keys * compress_ratio, dtype=torch.int32)

    original_score = qsa.qsa_mqa_prefill_reference
    scored_row_counts = []

    def record_score(*args, **kwargs):
        scored_row_counts.append(args[0].shape[0])
        return original_score(*args, **kwargs)

    monkeypatch.setattr(qsa, "qsa_mqa_prefill_reference", record_score)
    actual = qsa.select_qsa_prefill_tokens_reference(
        query,
        compressed_keys,
        starts,
        ends,
        positions,
        sequence_lengths,
        token_topk,
        compress_ratio,
        logits_budget_bytes=budget,
    )

    logits = original_score(query, compressed_keys, starts, ends)
    blocks = qsa.qsa_relative_topk(
        logits, starts, ends, topk=token_topk // compress_ratio
    )
    expected = qsa.expand_qsa_block_indices(
        blocks, positions, sequence_lengths, compress_ratio, token_topk
    )

    assert qsa.QSA_PREFILL_LOGITS_BUDGET_BYTES == 128 * 1024 * 1024
    assert (
        qsa.qsa_prefill_row_chunk_size(rows, keys, heads, logits_budget_bytes=budget)
        == 32
    )
    assert scored_row_counts == [32, 32, 1]
    torch.testing.assert_close(actual, expected)


def test_qsa_prefill_selection_handles_no_compressed_blocks():
    query = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    compressed_keys = torch.empty(0, 1, 8, dtype=torch.bfloat16)

    selected = qsa.select_qsa_prefill_tokens_reference(
        query,
        compressed_keys,
        row_starts=torch.zeros(2, dtype=torch.int32),
        row_ends=torch.zeros(2, dtype=torch.int32),
        query_positions=torch.tensor([1, 2]),
        sequence_lengths=torch.tensor([2, 3]),
        token_topk=8,
        compress_ratio=4,
    )

    assert selected[0, :2].tolist() == [0, 1]
    assert selected[1, :3].tolist() == [0, 1, 2]
    assert torch.all(selected[0, 2:] == -1)
    assert torch.all(selected[1, 3:] == -1)
