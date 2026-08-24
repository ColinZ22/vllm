# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.config.compilation import CompilationConfig
from vllm.config.speculative import SpeculativeConfig
from vllm.models.qwen3_8_flash_next.config import (
    Qwen3_8FlashNextConfig,
    Qwen3_8FlashNextTextConfig,
)
from vllm.models.qwen3_8_flash_next.nvidia.model_state import Qwen3_8FlashNextModelState
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
)
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridAttnMetadata,
    MambaHybridModelState,
)


def _text_config(**kwargs) -> Qwen3_8FlashNextTextConfig:
    values = {
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "num_experts": 0,
        "hc_count": 2,
        "hc_lowrank": 4,
        "ple_layer_ids": [1],
        "mtp_num_hidden_layers": 1,
        "mtp": {"hybrid": True},
    }
    values.update(kwargs)
    return Qwen3_8FlashNextTextConfig(**values)


def test_qwen3_8_flash_next_framework_defaults_enable_architecture_features() -> None:
    config = _text_config()

    assert config.hc_count == 2
    assert config.output_gate_type == "sigmoid"
    assert config.spec_hidden_size == 32
    assert config.spec_decode_returns_tuple


def test_qwen3_8_flash_next_mtp_returns_sample_and_multi_streams() -> None:
    from vllm.models.qwen3_8_flash_next.nvidia.mtp import (
        Qwen3_8FlashNextMultiTokenPredictor,
    )

    model = object.__new__(Qwen3_8FlashNextMultiTokenPredictor)
    torch.nn.Module.__init__(model)
    model.hc_count = 2
    model.hidden_size = 4
    model.num_mtp_layers = 1
    model.layers = [
        lambda **kwargs: (
            kwargs["hidden_states"],
            kwargs["hidden_states"],
            torch.zeros(kwargs["hidden_states"].shape[0], 2),
        ),
    ]
    model.hyper_connection_mixer = SimpleNamespace(
        combine_and_mix=lambda hidden_states, block_output, injection: (
            hidden_states,
            hidden_states.unflatten(-1, (2, 4)).mean(-2),
            None,
        ),
    )
    multi_hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8)
    pp_group = SimpleNamespace(is_first_rank=False, is_last_rank=True)

    with patch(
        "vllm.models.qwen3_8_flash_next.nvidia.mtp.get_pp_group",
        return_value=pp_group,
    ):
        sample_hidden, returned_multi_hidden = model.forward(
            input_ids=None,
            positions=torch.arange(2),
            intermediate_tensors={"hidden_states": multi_hidden},
        )

    torch.testing.assert_close(
        sample_hidden,
        multi_hidden.unflatten(-1, (2, 4)).mean(dim=-2),
    )
    assert returned_multi_hidden is multi_hidden


def test_qwen3_8_flash_next_qsa_enables_per_group_draft_metadata() -> None:
    config = _text_config(
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=2048,
        indexer_compress_ratio=4,
    )

    assert config.uses_per_group_attn_metadata


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"indexer_n_heads": 2}, "missing required fields"),
        (
            {
                "indexer_n_heads": 0,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 8,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
            },
            "must be positive",
        ),
        (
            {
                "indexer_n_heads": 2,
                "indexer_kv_heads": 2,
                "indexer_head_dim": 8,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
            },
            "indexer_kv_heads=1",
        ),
        (
            {
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 8,
                "indexer_budget": 1025,
                "indexer_compress_ratio": 2,
            },
            "must be divisible",
        ),
        (
            {
                "indexer_n_heads": 2,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 8,
                "indexer_budget": 1024,
                "indexer_compress_ratio": 4,
            },
            "512 or 2048",
        ),
    ],
)
def test_qwen3_8_flash_next_rejects_invalid_qsa_config(
    overrides: dict[str, int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _text_config(**overrides)


def test_qwen3_8_flash_next_qsa_is_split_from_piecewise_graphs() -> None:
    assert (
        "vllm::qwen3_8_flash_next_qsa_with_output" in CompilationConfig._attention_ops
    )


def test_qwen3_8_flash_next_mtp_override_preserves_text_backbone_layout() -> None:
    config = Qwen3_8FlashNextConfig(
        architectures=["Qwen3_8FlashNextForConditionalGeneration"],
        text_config=_text_config(),
    )

    draft_config = SpeculativeConfig.hf_config_override(config)

    assert draft_config.model_type == "qwen3_8_flash_next_mtp"
    assert draft_config.architectures == ["Qwen3_8FlashNextMTP"]
    assert draft_config.n_predict == 1
    assert draft_config.text_config.num_hidden_layers == 2
    assert draft_config.text_config.layer_types == [
        "linear_attention",
        "full_attention",
    ]


def test_qwen3_8_flash_next_registers_v2_model_state() -> None:
    from vllm.models.qwen3_8_flash_next.nvidia.model import (
        Qwen3_8FlashNextForCausalLM,
        Qwen3_8FlashNextForConditionalGeneration,
    )

    assert (
        Qwen3_8FlashNextForCausalLM.get_model_state_cls() is Qwen3_8FlashNextModelState
    )
    assert (
        Qwen3_8FlashNextForConditionalGeneration.get_model_state_cls()
        is Qwen3_8FlashNextModelState
    )


def test_qwen3_8_flash_next_uses_local_moe_metadata() -> None:
    from vllm.model_executor.models.qwen3_next import QwenNextMixtureOfExperts
    from vllm.models.qwen3_8_flash_next.nvidia.model import (
        Qwen3_8FlashNextDecoderLayer,
        Qwen3_8FlashNextMixtureOfExperts,
        Qwen3_8FlashNextSparseMoeBlock,
    )

    assert not issubclass(Qwen3_8FlashNextMixtureOfExperts, QwenNextMixtureOfExperts)

    layer = Qwen3_8FlashNextDecoderLayer.__new__(Qwen3_8FlashNextDecoderLayer)
    torch.nn.Module.__init__(layer)
    moe = Qwen3_8FlashNextSparseMoeBlock.__new__(Qwen3_8FlashNextSparseMoeBlock)
    torch.nn.Module.__init__(moe)
    moe.experts = torch.nn.Identity()
    moe.n_logical_experts = 8
    moe.n_physical_experts = 10
    moe.n_local_physical_experts = 5
    moe.n_routed_experts = 8
    moe.n_shared_experts = 1
    moe.n_redundant_experts = 2
    layer.mlp = moe

    metadata = Qwen3_8FlashNextMixtureOfExperts()
    metadata.set_moe_parameters([layer])

    assert metadata.moe_layers == [moe.experts]
    assert metadata.num_moe_layers == 1
    assert metadata.num_logical_experts == 8
    assert metadata.num_physical_experts == 10
    assert metadata.num_shared_experts == 1


def test_qwen3_8_flash_next_model_state_prepares_ngram_context() -> None:
    model_state = object.__new__(Qwen3_8FlashNextModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_context_len = 3
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ngram_context_offsets = torch.arange(-3, 0, dtype=torch.int64)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    input_batch = SimpleNamespace(
        num_reqs=2,
        num_reqs_after_padding=3,
        idx_mapping=torch.tensor([1, 0]),
        query_start_loc=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    req_states = SimpleNamespace(
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([3, 1])),
        all_token_ids=SimpleNamespace(
            gpu=torch.tensor([[1, 2, 3, 4], [20, 21, 22, 23]], dtype=torch.int32)
        ),
    )

    with patch.object(MambaHybridModelState, "prepare_inputs", return_value={}):
        model_inputs = model_state.prepare_inputs(input_batch, req_states)

    torch.testing.assert_close(
        model_inputs["query_start_loc"],
        torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        model_inputs["ngram_context"],
        torch.tensor([[99, 99, 20], [1, 2, 3], [99, 99, 99]], dtype=torch.int32),
    )


def test_qwen3_8_flash_next_model_state_prepares_stable_dummy_ngram_inputs() -> None:
    model_state = object.__new__(Qwen3_8FlashNextModelState)
    model_state.uses_ngram_embedding = True
    model_state.ngram_eos_token_id = 99
    model_state.ngram_context = torch.empty((4, 3), dtype=torch.int32)
    model_state.ple_query_start_loc = torch.empty(5, dtype=torch.int32)

    with patch.object(MambaHybridModelState, "prepare_dummy_inputs", return_value={}):
        first = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)
        query_start_loc_ptr = first["query_start_loc"].data_ptr()
        ngram_context_ptr = first["ngram_context"].data_ptr()
        second = model_state.prepare_dummy_inputs(num_reqs=3, num_tokens=8)

    torch.testing.assert_close(
        second["query_start_loc"], torch.tensor([0, 2, 5, 8], dtype=torch.int32)
    )
    torch.testing.assert_close(
        second["ngram_context"], torch.full((3, 3), 99, dtype=torch.int32)
    )
    assert second["query_start_loc"].data_ptr() == query_start_loc_ptr
    assert second["ngram_context"].data_ptr() == ngram_context_ptr


def test_qwen3_8_flash_next_model_state_skips_ngram_state_without_ple() -> None:
    def init_base_state(
        state,
        vllm_config,
        model,
        encoder_cache,
        device,
    ) -> None:
        state.model_config = vllm_config.model_config
        state.max_num_reqs = 4
        state.device = device

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_layer_ids=[]),
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    with patch.object(MambaHybridModelState, "__init__", init_base_state):
        model_state = Qwen3_8FlashNextModelState(
            vllm_config,
            torch.nn.Identity(),
            None,
            torch.device("cpu"),
        )

    assert not model_state.uses_ngram_embedding
    assert model_state.ngram_context_len == 0
    assert model_state.ngram_eos_token_id == 0
    assert not hasattr(model_state, "ngram_context")
    base_inputs = {"input_ids": torch.tensor([1])}
    with (
        patch.object(
            MambaHybridModelState,
            "prepare_inputs",
            return_value=base_inputs,
        ),
        patch.object(
            MambaHybridModelState,
            "prepare_dummy_inputs",
            return_value=base_inputs,
        ),
    ):
        assert (
            model_state.prepare_inputs(SimpleNamespace(), SimpleNamespace())
            is base_inputs
        )
        assert model_state.prepare_dummy_inputs(1, 1) is base_inputs


def test_qwen3_8_flash_next_model_state_rejects_pp_with_ple() -> None:
    def init_base_state(
        state,
        vllm_config,
        model,
        encoder_cache,
        device,
    ) -> None:
        state.model_config = vllm_config.model_config
        state.max_num_reqs = 4
        state.device = device

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(ple_layer_ids=[1]),
        ),
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
    )
    with (
        patch.object(MambaHybridModelState, "__init__", init_base_state),
        pytest.raises(RuntimeError, match="pipeline_parallel_size=1"),
    ):
        Qwen3_8FlashNextModelState(
            vllm_config,
            torch.nn.Identity(),
            None,
            torch.device("cpu"),
        )


def test_qwen3_8_flash_next_ple_builder_receives_spec_decode_metadata() -> None:
    num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32)
    num_decode_draft_tokens_cpu = torch.tensor([-1, 2], dtype=torch.int32)
    metadata = MambaHybridAttnMetadata(
        is_prefilling=torch.tensor([False, False]),
        num_accepted_tokens=num_accepted_tokens,
        num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
    )
    builder = PleShortConvAttentionMetadataBuilder.__new__(
        PleShortConvAttentionMetadataBuilder
    )

    kwargs = metadata.get_extra_attn_kwargs(builder, num_reqs=2)

    torch.testing.assert_close(kwargs["num_accepted_tokens"], num_accepted_tokens)
    torch.testing.assert_close(
        kwargs["num_decode_draft_tokens_cpu"], num_decode_draft_tokens_cpu
    )
