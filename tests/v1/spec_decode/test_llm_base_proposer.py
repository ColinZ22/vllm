# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SpecDecodeBaseProposer.initialize_attn_backend.

Block tables are stored at kernel-block granularity, so the proposer's
``block_size`` (used for slot-mapping math) must be the kernel block size,
not the KV cache manager's block size — the two differ when manager blocks
are split for the attention kernel. The value must also be deterministic:
``_draft_attn_layer_names`` is a set, whose iteration order varies across
processes, so anything derived from iteration order must not leak into
``block_size``.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
from vllm.v1.spec_decode.eagle import EagleProposer

SCHEDULER_BLOCK_SIZE = 256
KERNEL_BLOCK_SIZE = 64


class _FakeAttentionGroup:
    def __init__(self, backend, layer_names, kv_cache_spec, kv_cache_group_id):
        self.backend = backend
        self.layer_names = list(layer_names)
        self.kv_cache_spec = kv_cache_spec
        self.kv_cache_group_id = kv_cache_group_id
        self.kernel_block_size = None
        self.builder = SimpleNamespace(kv_cache_spec=self.kv_cache_spec)

    def create_metadata_builders(self, vllm_config, device, kernel_block_size=None):
        self.kernel_block_size = kernel_block_size

    def get_metadata_builder(self):
        return self.builder


def _make_proposer(
    monkeypatch: pytest.MonkeyPatch, layer_names: set[str]
) -> EagleProposer:
    fake_layers = {}
    for name in layer_names:
        backend = SimpleNamespace(full_cls_name=lambda: "FakeBackend")
        fake_layers[name] = SimpleNamespace(
            get_attn_backend=lambda backend=backend: backend
        )
    monkeypatch.setattr(
        llm_base_proposer, "get_layers_from_vllm_config", lambda *a, **k: fake_layers
    )
    monkeypatch.setattr(llm_base_proposer, "AttentionGroup", _FakeAttentionGroup)

    proposer = EagleProposer.__new__(EagleProposer)
    proposer.vllm_config = None
    proposer.device = None
    proposer._draft_attn_layer_names = set(layer_names)
    proposer.kv_cache_gid = -1
    proposer.draft_attn_groups = []
    proposer.block_size = -1
    return proposer


def _make_kv_cache_config(layer_names: set[str]) -> SimpleNamespace:
    spec = SimpleNamespace(block_size=SCHEDULER_BLOCK_SIZE)
    group = SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=spec)
    return SimpleNamespace(kv_cache_groups=[group])


def test_block_size_uses_kernel_block_size(monkeypatch: pytest.MonkeyPatch):
    """The proposer's slot-mapping math runs against the kernel-granularity
    block table, so block_size must come from kernel_block_sizes."""
    layer_names = {"draft.0.self_attn.attn"}
    proposer = _make_proposer(monkeypatch, layer_names)

    proposer.initialize_attn_backend(
        _make_kv_cache_config(layer_names),
        kernel_block_sizes=[KERNEL_BLOCK_SIZE],
    )

    assert proposer.block_size == KERNEL_BLOCK_SIZE
    assert proposer.block_size != SCHEDULER_BLOCK_SIZE
    # The metadata builder keeps receiving the kernel block size as well.
    assert proposer.draft_attn_groups[0].kernel_block_size == KERNEL_BLOCK_SIZE


def test_block_size_falls_back_to_kv_cache_spec(monkeypatch: pytest.MonkeyPatch):
    layer_names = {"draft.0.self_attn.attn"}
    proposer = _make_proposer(monkeypatch, layer_names)

    proposer.initialize_attn_backend(
        _make_kv_cache_config(layer_names), kernel_block_sizes=None
    )

    assert proposer.block_size == SCHEDULER_BLOCK_SIZE


def test_draft_layer_iteration_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    """_draft_attn_layer_names is a set; the attention groups built from it
    must not depend on its (process-random) iteration order."""
    layer_names = {"draft.c.attn", "draft.a.attn", "draft.b.attn"}
    expected_order = sorted(layer_names)

    for insertion_order in (expected_order, expected_order[::-1]):
        proposer = _make_proposer(monkeypatch, set(insertion_order))
        proposer.initialize_attn_backend(
            _make_kv_cache_config(set(insertion_order)),
            kernel_block_sizes=[KERNEL_BLOCK_SIZE],
        )
        assert len(proposer.draft_attn_groups) == 1
        assert proposer.draft_attn_groups[0].layer_names == expected_order
        assert proposer.block_size == KERNEL_BLOCK_SIZE


def _make_qwen3_8_flash_next_multigroup_proposer(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[EagleProposer, SimpleNamespace]:
    main_layer = "draft.mtp.layers.0.self_attn.attn"
    raw_layer = "draft.mtp.layers.0.self_attn.indexer.raw_key_cache"
    compressed_layer = "draft.mtp.layers.0.self_attn.indexer.compressed_key_cache"
    layer_names = {main_layer, raw_layer, compressed_layer}
    proposer = _make_proposer(monkeypatch, layer_names)
    proposer._uses_per_group_attn_metadata = True
    proposer._draft_uses_multiple_kv_groups = False
    proposer._per_group_block_tables = {}
    proposer._per_group_slot_mappings = {}
    proposer._per_group_slot_mapping_buffers = {}
    proposer._per_group_kernel_block_sizes = {}
    proposer.max_positions = 16
    proposer.max_model_len = 1024
    proposer.arange = torch.arange(17, dtype=torch.int32)
    proposer._slot_mapping_buffer = torch.empty(16, dtype=torch.int64)
    proposer.device = torch.device("cpu")

    main_spec = FullAttentionSpec(
        block_size=SCHEDULER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        head_size_v=128,
        dtype=torch.bfloat16,
    )
    raw_spec = FullAttentionSpec(
        block_size=SCHEDULER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=140,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed_spec = MLAAttentionSpec(
        block_size=SCHEDULER_BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        compress_ratio=64,
    )
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=[raw_layer],
                kv_cache_spec=raw_spec,
            ),
            SimpleNamespace(
                layer_names=[compressed_layer],
                kv_cache_spec=compressed_spec,
            ),
            SimpleNamespace(
                layer_names=[main_layer],
                kv_cache_spec=main_spec,
            ),
        ]
    )
    return proposer, config


def test_qwen3_8_flash_next_mtp_initializes_each_draft_kv_cache_group(
    monkeypatch: pytest.MonkeyPatch,
):
    proposer, config = _make_qwen3_8_flash_next_multigroup_proposer(monkeypatch)

    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * 3,
    )

    assert proposer.kv_cache_gid == 2
    assert proposer.block_size == KERNEL_BLOCK_SIZE
    assert proposer._draft_uses_multiple_kv_groups
    assert {group.kv_cache_group_id for group in proposer.draft_attn_groups} == {
        0,
        1,
        2,
    }
    assert {
        layer_name
        for group in proposer.draft_attn_groups
        for layer_name in group.layer_names
    } == proposer._draft_attn_layer_names


def test_qwen3_8_flash_next_mtp_routes_block_tables_and_slots_by_group(
    monkeypatch: pytest.MonkeyPatch,
):
    proposer, config = _make_qwen3_8_flash_next_multigroup_proposer(monkeypatch)
    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * 3,
    )

    def build_for_drafting(*, common_attn_metadata, draft_index):
        return SimpleNamespace(
            common=common_attn_metadata,
            draft_index=draft_index,
        )

    for group in proposer.draft_attn_groups:
        group.builder.build_for_drafting = build_for_drafting

    tables = {
        gid: torch.tensor([[10 * (gid + 1), 10 * (gid + 1) + 1]], dtype=torch.int32)
        for gid in range(3)
    }
    slots_by_gid = {
        gid: torch.tensor(
            [640 * (gid + 1), 640 * (gid + 1) + 1],
            dtype=torch.int64,
        )
        for gid in range(3)
    }
    for gid in range(3):
        proposer.set_per_group_attn_metadata(gid, tables[gid], slots_by_gid[gid])
    primary_table = tables[proposer.kv_cache_gid]
    primary_slots = slots_by_gid[proposer.kv_cache_gid]
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table_tensor=primary_table,
        slot_mapping=primary_slots,
    )

    _, per_layer = proposer.build_per_group_and_layer_attn_metadata(
        common,
        draft_index=3,
    )

    for group in proposer.draft_attn_groups:
        for layer_name in group.layer_names:
            metadata = per_layer[layer_name]
            expected_table = tables[group.kv_cache_group_id]
            expected_slots = slots_by_gid[group.kv_cache_group_id]
            assert torch.equal(metadata.common.block_table_tensor, expected_table)
            assert torch.equal(metadata.common.slot_mapping, expected_slots)
            assert metadata.draft_index == 3

    slot_mapping_by_layer = proposer._get_slot_mapping(4, primary_slots)
    for group in proposer.draft_attn_groups:
        expected_prefix = slots_by_gid[group.kv_cache_group_id]
        for layer_name in group.layer_names:
            layer_slots = slot_mapping_by_layer[layer_name]
            assert torch.equal(layer_slots[:2], expected_prefix)
            assert torch.equal(
                layer_slots[2:],
                torch.full(
                    (2,),
                    llm_base_proposer.PADDING_SLOT_ID,
                    dtype=torch.int64,
                ),
            )


def test_qwen3_8_flash_next_mtp_dummy_run_routes_slot_mappings_by_group(
    monkeypatch: pytest.MonkeyPatch,
):
    proposer, config = _make_qwen3_8_flash_next_multigroup_proposer(monkeypatch)
    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * 3,
    )
    proposer.parallel_drafting = True
    proposer.num_speculative_tokens = 1
    proposer._determine_batch_execution_and_padding = lambda *args, **kwargs: (
        llm_base_proposer.CUDAGraphMode.NONE,
        2,
        None,
    )
    proposer.vllm_config = None
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.input_ids = torch.zeros(2, dtype=torch.int64)
    proposer.positions = torch.arange(2, dtype=torch.int64)
    proposer.model = lambda **kwargs: None
    monkeypatch.setattr(
        llm_base_proposer,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    slots_by_gid = {
        gid: torch.tensor([10 * gid, 10 * gid + 1], dtype=torch.int64)
        for gid in range(3)
    }
    slots_by_layer = {
        layer_name: slots_by_gid[group.kv_cache_group_id]
        for group in proposer.draft_attn_groups
        for layer_name in group.layer_names
    }

    proposer.dummy_run(
        2,
        use_cudagraphs=False,
        slot_mappings=slots_by_layer,
    )

    for group in proposer.draft_attn_groups:
        expected = slots_by_gid[group.kv_cache_group_id]
        for layer_name in group.layer_names:
            actual = proposer._slot_mapping_buffer_for_group(group.kv_cache_group_id)[
                :2
            ]
            assert torch.equal(actual, expected), layer_name


def test_qwen3_8_flash_next_mtp_group_slot_mapping_marks_cudagraph_padding(
    monkeypatch: pytest.MonkeyPatch,
):
    proposer, config = _make_qwen3_8_flash_next_multigroup_proposer(monkeypatch)
    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * 3,
    )
    query_start_loc = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32)
    common = SimpleNamespace(
        num_reqs=4,
        num_actual_tokens=16,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.tensor([68, 68, 68, 0], dtype=torch.int32),
    )
    block_table = torch.tensor(
        [[10, 11], [20, 21], [30, 31], [-1, -1]],
        dtype=torch.int32,
    )

    slots = proposer._build_group_slot_mapping(common, block_table, gid=0)

    expected = torch.tensor(
        [
            704,
            705,
            706,
            707,
            1344,
            1345,
            1346,
            1347,
            1984,
            1985,
            1986,
            1987,
            -1,
            -1,
            -1,
            -1,
        ],
        dtype=torch.int64,
    )
    assert torch.equal(slots, expected)


def test_qwen3_8_flash_next_mtp_routes_shared_metadata_to_main_group(
    monkeypatch: pytest.MonkeyPatch,
):
    proposer, config = _make_qwen3_8_flash_next_multigroup_proposer(monkeypatch)
    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * 3,
    )
    main_gid = proposer.kv_cache_gid
    main_table = torch.tensor([[30, 31]], dtype=torch.int32)
    main_slots = torch.tensor([1920, 1921], dtype=torch.int64)
    proposer.set_per_group_attn_metadata(main_gid, main_table, main_slots)
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2,
        block_table_tensor=torch.tensor([[10, 11]], dtype=torch.int32),
        slot_mapping=torch.tensor([640, 641], dtype=torch.int64),
    )

    proposer._set_primary_common_attn_metadata(common)

    assert torch.equal(common.block_table_tensor, main_table)
    assert torch.equal(common.slot_mapping, main_slots)


def test_qwen3_8_flash_next_mtp_advances_slot_mappings_for_every_group(
    monkeypatch: pytest.MonkeyPatch,
):
    proposer, config = _make_qwen3_8_flash_next_multigroup_proposer(monkeypatch)
    proposer.initialize_attn_backend(
        config,
        kernel_block_sizes=[KERNEL_BLOCK_SIZE] * 3,
    )
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0
    proposer.draft_uses_xdrope_dim = 0
    proposer.positions = torch.empty(16, dtype=torch.int64)

    def fake_step_update(
        *,
        positions_1d,
        block_table_tensor,
        seq_lens,
        block_size,
        max_model_len,
        out_clamped_positions,
        out_slot_mapping,
        input_batch_size,
    ):
        del max_model_len
        next_positions = positions_1d + 1
        out_clamped_positions.copy_(next_positions)
        logical_blocks = next_positions // block_size
        physical_blocks = block_table_tensor.gather(
            1, logical_blocks.long().unsqueeze(1)
        ).squeeze(1)
        out_slot_mapping[: positions_1d.shape[0]].copy_(
            physical_blocks * block_size + next_positions.remainder(block_size)
        )
        if input_batch_size > positions_1d.shape[0]:
            out_slot_mapping[positions_1d.shape[0] :].fill_(
                llm_base_proposer.PADDING_SLOT_ID
            )
        seq_lens.add_(1)

    monkeypatch.setattr(
        llm_base_proposer,
        "eagle_step_update_slot_mapping_and_metadata",
        fake_step_update,
    )

    tables = {
        gid: torch.tensor([[10 * (gid + 1), 10 * (gid + 1) + 1]], dtype=torch.int32)
        for gid in range(3)
    }
    for gid in range(3):
        proposer.set_per_group_attn_metadata(
            gid,
            tables[gid],
            torch.tensor([0], dtype=torch.int64),
        )
    primary_gid = proposer.kv_cache_gid
    common = SimpleNamespace(
        block_table_tensor=tables[primary_gid],
        seq_lens=torch.tensor([64], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        max_seq_len=64,
        _seq_lens_cpu=None,
        _num_computed_tokens_cpu=None,
        seq_lens_cpu_upper_bound=None,
    )

    next_positions = proposer._update_positions_dependent_metadata(
        torch.tensor([63], dtype=torch.int64),
        common,
        batch_size=1,
        input_batch_size=1,
        block_size=KERNEL_BLOCK_SIZE,
    )

    assert torch.equal(next_positions, torch.tensor([64], dtype=torch.int64))
    for gid, table in tables.items():
        expected_slot = table[0, 1].long() * KERNEL_BLOCK_SIZE
        assert torch.equal(
            proposer._per_group_slot_mappings[gid],
            expected_slot.unsqueeze(0),
        )
