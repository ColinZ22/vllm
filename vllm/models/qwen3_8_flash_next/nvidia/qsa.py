# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton production kernels and a CPU oracle."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.sparse_attention_qsa import (
    QSAProductionKernelUnavailable,
    QSAReferenceSparseAttention,
)
from . import model
from .indexer_qsa import QSAIndexer

_QSA_GATHER_BUDGET_BYTES = 128 * 1024 * 1024


def qsa_token_to_request(
    query_start_loc: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Expand packed query boundaries into one request id per query token."""

    if query_start_loc.ndim != 1 or query_start_loc.numel() == 0:
        raise ValueError("query_start_loc must contain packed row boundaries")
    query_lens = torch.diff(query_start_loc).long()
    requests = torch.arange(
        query_lens.numel(), dtype=torch.int32, device=query_start_loc.device
    )
    if num_tokens == 0:
        return requests[:0]
    padded_query_lens = query_lens.clone()
    padded_query_lens[-1] += num_tokens - query_start_loc[-1]
    return torch.repeat_interleave(
        requests,
        padded_query_lens,
        output_size=num_tokens,
    )


def qsa_query_positions(
    query_start_loc: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """Return request-relative logical position for every packed query row."""

    rows = token_to_req.numel()
    requests = token_to_req.to(device=query_start_loc.device, dtype=torch.long)
    query_lens = torch.diff(query_start_loc).long()
    row_ids = torch.arange(rows, dtype=torch.long, device=query_start_loc.device)
    within_query = row_ids - query_start_loc.index_select(0, requests).long()
    logical_positions = (
        sequence_lengths.to(query_start_loc.device).index_select(0, requests).long()
        - query_lens.index_select(0, requests)
        + within_query
    )
    return torch.where(row_ids < query_start_loc[-1], logical_positions, -1)


def gather_qsa_selected_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather request-relative QSA selections from a paged main K/V cache.

    Invalid entries must form a contiguous ``-1`` suffix. They are zero-filled
    and excluded through the returned per-row valid counts.
    """

    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA K/V caches must be [blocks, slots, heads, dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("QSA key and value caches must have identical shapes")
    if logical_indices.ndim != 2:
        raise ValueError("QSA logical indices must be [tokens, width]")
    rows, width = logical_indices.shape
    if token_to_req.shape != (rows,):
        raise ValueError("QSA token-to-request mapping must match query rows")
    if block_table.ndim != 2:
        raise ValueError("QSA block table must be two-dimensional")
    if sequence_lengths.ndim != 1:
        raise ValueError("QSA sequence lengths must be one-dimensional")
    if block_table.shape[0] != sequence_lengths.numel():
        raise ValueError("QSA block table and sequence lengths must agree")
    if query_positions is not None and query_positions.shape != (rows,):
        raise ValueError("QSA query positions must match query rows")

    device = key_cache.device
    logical = logical_indices.to(device=device, dtype=torch.long)
    requests = token_to_req.to(device=device, dtype=torch.long)
    seq_lens = sequence_lengths.to(device=device, dtype=torch.long)
    table = block_table.to(device=device)
    if requests.numel() and bool(
        ((requests < 0) | (requests >= seq_lens.numel())).any().item()
    ):
        raise ValueError("QSA request mapping is out of range")

    valid = logical >= 0
    valid_counts = valid.sum(dim=1, dtype=torch.int32)
    columns = torch.arange(width, device=device).unsqueeze(0)
    if bool((valid != (columns < valid_counts.unsqueeze(1))).any().item()):
        raise ValueError("QSA valid indices must precede the -1 padding suffix")

    row_lengths = seq_lens.index_select(0, requests)
    if bool((valid & (logical >= row_lengths.unsqueeze(1))).any().item()):
        raise ValueError("QSA selected an index outside its request sequence")
    if query_positions is not None:
        row_positions = query_positions.to(device=device, dtype=torch.long)
        if bool((valid & (logical > row_positions.unsqueeze(1))).any().item()):
            raise ValueError("QSA selected a token after its query position")

    selected_shape = (rows, width, key_cache.shape[2], key_cache.shape[3])
    if not bool(valid.any().item()):
        return (
            key_cache.new_zeros(selected_shape),
            value_cache.new_zeros(selected_shape),
            valid_counts,
        )
    if block_table.shape[1] == 0 or key_cache.shape[0] == 0:
        raise ValueError("QSA cannot gather from an empty paged cache")

    block_size = key_cache.shape[1]
    safe_logical = logical.clamp_min(0)
    logical_blocks = torch.div(safe_logical, block_size, rounding_mode="floor")
    if bool((valid & (logical_blocks >= table.shape[1])).any().item()):
        raise ValueError("QSA selected an index outside its block table")
    safe_blocks = logical_blocks.clamp_max(table.shape[1] - 1)
    physical_blocks = table[requests.unsqueeze(1), safe_blocks].long()
    if bool((valid & (physical_blocks < 0)).any().item()):
        raise ValueError("QSA selected an unallocated cache block")
    if bool((valid & (physical_blocks >= key_cache.shape[0])).any().item()):
        raise ValueError("QSA selected a physical block outside its cache")

    safe_physical = physical_blocks.clamp(0, key_cache.shape[0] - 1)
    offsets = safe_logical.remainder(block_size)
    selected_k = key_cache[safe_physical, offsets]
    selected_v = value_cache[safe_physical, offsets]
    mask = valid.unsqueeze(-1).unsqueeze(-1)
    selected_k = torch.where(mask, selected_k, torch.zeros_like(selected_k))
    selected_v = torch.where(mask, selected_v, torch.zeros_like(selected_v))
    return selected_k, selected_v, valid_counts


class Qwen3_8FlashNextQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen3_8FlashNextQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen3_8FlashNextQSAFlashAttentionImpl]:
        return Qwen3_8FlashNextQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen3_8FlashNextQSAMetadataBuilder]:
        return Qwen3_8FlashNextQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen3_8FlashNextQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA, with a gathered FlashAttention CPU oracle path."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = qsa_token_to_request(
            attn_metadata.query_start_loc,
            num_tokens,
        )
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        if key_cache.dtype != torch.bfloat16 or query.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires BF16 Q/K/V")

        if query.is_cuda:
            from .ops.qsa import qsa_sparse_paged_attention

            qsa_sparse_paged_attention(
                query[:num_tokens],
                key_cache,
                value_cache,
                logical_indices,
                attn_metadata.block_table,
                token_to_req,
                self.scale,
                output[:num_tokens],
            )
            return output

        query_positions = qsa_query_positions(
            attn_metadata.query_start_loc,
            token_to_req,
            attn_metadata.seq_lens,
        )

        width = logical_indices.shape[1]
        if width <= 0:
            raise RuntimeError("QSA selection width must be positive")
        bytes_per_row = (
            width * self.num_kv_heads * self.head_size * key_cache.element_size() * 2
        )
        rows_per_chunk = max(1, _QSA_GATHER_BUDGET_BYTES // bytes_per_row)
        fa_version = self.vllm_flash_attn_version
        if fa_version is None:
            raise RuntimeError("QSA could not resolve a FlashAttention version")
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        for row_start in range(0, num_tokens, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, num_tokens)
            selected_k, selected_v, valid_counts = gather_qsa_selected_kv(
                key_cache,
                value_cache,
                logical_indices[row_start:row_end],
                attn_metadata.block_table,
                token_to_req[row_start:row_end],
                attn_metadata.seq_lens,
                query_positions[row_start:row_end],
            )
            if bool((valid_counts <= 0).any().item()):
                raise RuntimeError("QSA produced an empty selection for a query")
            rows = row_end - row_start
            cu_q = torch.arange(rows + 1, dtype=torch.int32, device=query.device)
            cu_k = cu_q * width
            flash_attn_varlen_func(
                q=query[row_start:row_end],
                k=selected_k.flatten(0, 1),
                v=selected_v.flatten(0, 1),
                out=output[row_start:row_end],
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                seqused_k=valid_counts,
                max_seqlen_q=1,
                max_seqlen_k=width,
                softmax_scale=self.scale,
                causal=False,
                fa_version=fa_version,
            )
        return output


class Qwen3_8FlashNextQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Any,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = False,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support KV quantization"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires causal decoder attention"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support dual-chunk RoPE"
            )
        # Qwen3.8-Flash-Next full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=bool(getattr(config, "qkv_bias", False)),
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and text_only
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen3_8FlashNextQSAFlashAttentionBackend
        self.impl = Qwen3_8FlashNextQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen3_8FlashNextQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen3_8_flash_next_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen3_8_flash_next_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_qsa_with_output_fake,
)


def require_qsa_cuda_kernel() -> None:
    """Validate that the production QSA Triton path can run."""

    if not current_platform.is_cuda() or not HAS_TRITON:
        raise QSAProductionKernelUnavailable(
            "Qwen3.8-Flash-Next QSA production kernels require CUDA and Triton"
        )


__all__ = [
    "QSAIndexer",
    "QSAProductionKernelUnavailable",
    "QSAReferenceSparseAttention",
    "Qwen3_8FlashNextQSAAttention",
    "Qwen3_8FlashNextQSAFlashAttentionBackend",
    "Qwen3_8FlashNextQSAFlashAttentionImpl",
    "gather_qsa_selected_kv",
    "qsa_query_positions",
    "qsa_token_to_request",
    "qwen3_8_flash_next_qsa_with_output",
    "require_qsa_cuda_kernel",
]
