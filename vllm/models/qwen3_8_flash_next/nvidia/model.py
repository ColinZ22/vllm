# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.8-Flash-Next model."""

from collections.abc import Iterable
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateCopyFuncsByType,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    MultiModalEmbeddings,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextDecoderLayer,
    Qwen3NextMLP,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.configs.qwen3_8_flash_next import (
    Qwen3_8FlashNextTextConfig,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from ..common.hyperconnection import (
    HYPERCONNECTION_CLASS_DICT,
    GatedResidualSimple,
    HyperConnectionConfig,
)
from ..config import Qwen3_8FlashNextConfig
from .ple_layer import Qwen3_8FlashNextNGramEmbedding, Qwen3_8FlashNextPLELayer
from .qsa import Qwen3_8FlashNextQSAAttention


def without_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Return ``None`` for weights excluded from Qwen3.8-Flash-Next ModelOpt-FP4."""

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config


def _remap_qsa_cache_scale_name(
    name: str,
    qsa_layer_ids: frozenset[int],
) -> str:
    """Map serialized main-cache scales onto the merged QSA owner.

    Regular attention keeps cache scales below its ``attn`` child. QSA owns
    that cache directly, so only QSA layers need the final path component
    moved to the owner's persistent ``_k_scale``/``_v_scale`` buffers.
    """

    scale_suffixes = {
        "k_proj.k_scale": "_k_scale",
        "k_proj.output_scale": "_k_scale",
        "attn.k_scale": "_k_scale",
        "attn._k_scale": "_k_scale",
        "k_scale": "_k_scale",
        "_k_scale": "_k_scale",
        "v_proj.v_scale": "_v_scale",
        "v_proj.output_scale": "_v_scale",
        "attn.v_scale": "_v_scale",
        "attn._v_scale": "_v_scale",
        "v_scale": "_v_scale",
        "_v_scale": "_v_scale",
    }
    for layer_id in qsa_layer_ids:
        marker = f"layers.{layer_id}.self_attn."
        marker_start = name.find(marker)
        if marker_start < 0 or (marker_start > 0 and name[marker_start - 1] != "."):
            continue
        suffix = name[marker_start + len(marker) :]
        mapped_suffix = scale_suffixes.get(suffix)
        if mapped_suffix is not None:
            return f"{name[: marker_start + len(marker)]}{mapped_suffix}"
    return name


_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]


# ---------------------------------------------------------------------------
# Qwen3_8FlashNextRMSNorm: custom RMSNorm with optional gated_layernorm support
# ---------------------------------------------------------------------------
class Qwen3_8FlashNextRMSNorm(nn.Module):
    """Qwen3.8-Flash-Next RMSNorm with its checkpoint-compatible optional gates."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        pre_affine: bool = False,
        gated_layernorm: bool = False,
        gated_layernorm_lowrank: int = 16,
        use_gemma_rms_norm: bool = True,
        group_size: int | None = None,
    ) -> None:
        super().__init__()
        if group_size is not None and dim % group_size:
            raise ValueError(
                f"dim ({dim}) must be divisible by group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.use_gemma_rms_norm = use_gemma_rms_norm
        self.weight = nn.Parameter(torch.zeros(dim))
        self.pre_affine = pre_affine
        self.pre_weight = nn.Parameter(torch.ones(dim)) if pre_affine else None
        self.gated_layernorm = gated_layernorm
        if gated_layernorm:
            self.gated_layernorm_downproj = nn.Linear(
                dim, gated_layernorm_lowrank, bias=False
            )
            self.gated_layernorm_upproj = nn.Linear(
                gated_layernorm_lowrank, dim, bias=False
            )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is None:
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_g = x.reshape(*x.shape[:-1], x.shape[-1] // self.group_size, self.group_size)
        x_g = x_g * torch.rsqrt(x_g.pow(2).mean(-1, keepdim=True) + self.eps)
        return x_g.flatten(-2)

    def _gate(self, x: torch.Tensor) -> torch.Tensor:
        gate_score = F.silu(self.gated_layernorm_downproj(x))
        gate_score = torch.sigmoid(self.gated_layernorm_upproj(gate_score))
        return x * gate_score

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        if residual is not None:
            x = x.float() + residual.float()
            residual = x
        if self.pre_weight is not None:
            x = x * self.pre_weight.float()
        output = self._norm(x.float())
        if self.use_gemma_rms_norm:
            output = output * (1.0 + self.weight.float())
        else:
            output = output * self.weight.float()
        output = output.to(orig_dtype)
        if self.gated_layernorm:
            output = output.to(torch.bfloat16)
            output = self._gate(output)
        return output if residual is None else (output, residual)

    def extra_repr(self) -> str:
        return (
            f"{tuple(self.weight.shape)}, eps={self.eps}, group_size={self.group_size}"
        )


class Qwen3_8FlashNextSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """Qwen3Next MoE with Qwen3.8-Flash-Next HC validation."""

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config
        use_hc = bool(getattr(config, "use_hc", False))
        if use_hc and vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # The current FusedMoEFactory owns its final tensor-parallel
        # reduction. Do not reduce the result a second time in the HC caller.
        self.requires_tp_all_reduce = False


class Qwen3_8FlashNextDecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
        force_disable_ple: bool = False,
        force_plain_hc_role: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        # use_hc: HyperConnection initialization
        self.use_hc = bool(getattr(config, "use_hc", False))
        if self.use_hc and vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next HC does not support sequence-parallel MoE"
            )
        # PLE declaration (guarded by use_ple + ple_layer_ids).
        # When use_ple=False the decoder layer is bit-wise identical to the
        # pre-PLE version: self.ple stays None and the forward path skips
        # the PLE branch entirely.
        self.ple: Qwen3_8FlashNextPLELayer | None = None
        use_ple = getattr(config, "use_ple", False)
        ple_layer_ids = getattr(config, "ple_layer_ids", None) or []
        if (
            use_ple
            and ple_layer_ids
            and (self.layer_idx + 1) in ple_layer_ids
            and not force_disable_ple
        ):
            ple_layer_ids_sorted = sorted(set(ple_layer_ids))
            ple_dense_layer_id_map = {
                abs_id: idx for idx, abs_id in enumerate(ple_layer_ids_sorted)
            }
            ple_dense_layer_id = ple_dense_layer_id_map[self.layer_idx + 1]
            self.ple = Qwen3_8FlashNextPLELayer(
                config,
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                ple_dense_layer_id=ple_dense_layer_id,
                prefix=f"{prefix}.ple",
            )

        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=not self.use_hc,
            )
        elif layer_type == "full_attention":
            use_qsa = getattr(config, "indexer_n_heads", None) is not None
            if not use_qsa:
                self.self_attn = Qwen3NextAttention(
                    config,
                    model_config=model_config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    reduce_results=not self.use_hc,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                self.self_attn = Qwen3_8FlashNextQSAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    layer_id=self.layer_idx,
                    quant_config=quant_config,
                    reduce_results=not self.use_hc,
                    prefix=f"{prefix}.self_attn",
                )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        num_experts = getattr(config, "num_experts", 0) or 0
        absolute_layer_id = self.layer_idx + 1
        is_moe_layer = self.layer_idx not in mlp_only_layers and (
            num_experts > 0 and absolute_layer_id % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = Qwen3_8FlashNextSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=not self.use_hc,
                prefix=f"{prefix}.mlp",
            )

        # gated_layernorm + use_hc: conditionally initialize HyperConnection
        # modules instead of separate pre-attention layer norms.
        if self.use_hc:
            hc_method = getattr(config, "hc_method", "")
            hc_class = HYPERCONNECTION_CLASS_DICT[hc_method]

            # Identify MTP layers (layer_idx >= num_hidden_layers) purely so
            # the role prefix picks up the extra MTP HC stream; layout is
            # always HS-major (HC outer, HS inner).
            is_mtp_layer = self.layer_idx >= config.num_hidden_layers
            # MTP layers force a plain HC role (no "mtp_" prefix) so the HC
            # stream count stays identical to the main model (hc_count, no +1).
            if force_plain_hc_role:
                is_mtp_layer = False
            hc_config = HyperConnectionConfig(
                hc_count=getattr(config, "hc_count", 4),
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                init_method_std=0.02,
                mtp_hc=getattr(config, "mtp_hc", False),
                hc_lowrank=getattr(config, "hc_lowrank", 128),
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=getattr(config, "hc_per_branch_norm", False),
            )
            role_prefix = "mtp_" if is_mtp_layer else ""
            self.attn_hyper_connection = hc_class(
                hc_config,
                layer_idx=self.layer_idx,
                role=f"{role_prefix}attn",
            )
            self.mlp_hyper_connection = hc_class(
                hc_config,
                layer_idx=self.layer_idx,
                role=f"{role_prefix}mlp",
            )
        else:
            norm_kwargs = {
                "eps": config.rms_norm_eps,
                "pre_affine": getattr(config, "pre_affine", False),
                "gated_layernorm": getattr(config, "gated_layernorm", False),
                "gated_layernorm_lowrank": getattr(
                    config, "gated_layernorm_lowrank", 16
                ),
                "use_gemma_rms_norm": getattr(config, "use_gemma_rms_norm", True),
            }
            self.input_layernorm = Qwen3_8FlashNextRMSNorm(
                config.hidden_size, **norm_kwargs
            )
            self.post_attention_layernorm = Qwen3_8FlashNextRMSNorm(
                config.hidden_size, **norm_kwargs
            )

        self.layer_scale = bool(getattr(config, "layer_scale", False))
        if self.layer_scale:
            self.attn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size, dtype=model_config.dtype)
            )
            self.ffn_layer_scale = nn.Parameter(
                torch.zeros(1, 1, config.hidden_size, dtype=model_config.dtype)
            )

    def _apply_layer_scale(
        self, hidden_states: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        if hidden_states.ndim == 2:
            return hidden_states * (scale.to(hidden_states.dtype)[0] + 1)
        return hidden_states * (scale.to(hidden_states.dtype) + 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del kwargs
        # When use_hc=True, PLE is applied INSIDE the HC pipeline after the
        # previous layer's combined output and before this layer's attention
        # mix. use_ple=False layers are entirely unaffected.
        if self.use_hc:
            hc_method = getattr(self.config, "hc_method", "")
            if "gated_residual" not in hc_method:
                raise ValueError(
                    "Only support forward Qwen3.8-Flash-Next decoder layer with "
                    "hc_method set to gate_residual_simple, but got " + hc_method
                )
            if residual is not None:
                raise ValueError("HC layers do not use a separate residual tensor")
            if self.ple is not None:
                if input_ids is None:
                    raise ValueError("PLE requires input_ids")
                hidden_states = hidden_states + self.ple(
                    hidden_states,
                    input_ids,
                    query_start_loc,
                    ngram_context,
                )

            mixed, hc_residual = self.attn_hyper_connection.mix(hidden_states)
            if self.layer_type == "linear_attention":
                self_attention_output = self.linear_attn(hidden_states=mixed)
            elif self.layer_type == "full_attention":
                self_attention_output = self.self_attn(
                    hidden_states=mixed,
                    positions=positions,
                )
            else:
                raise ValueError("Invalid layer_type")
            hidden_states = self_attention_output
            if get_tensor_model_parallel_world_size() > 1:
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            if self.layer_scale:
                hidden_states = self._apply_layer_scale(
                    hidden_states, self.attn_layer_scale
                )
            hidden_states = self.attn_hyper_connection.combine(
                hidden_states, hc_residual
            )

            mixed, hc_residual = self.mlp_hyper_connection.mix(hidden_states)
            hidden_states = self.mlp(mixed)
            if get_tensor_model_parallel_world_size() > 1 and getattr(
                self.mlp, "requires_tp_all_reduce", True
            ):
                hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            if self.layer_scale:
                hidden_states = self._apply_layer_scale(
                    hidden_states, self.ffn_layer_scale
                )
            hidden_states = self.mlp_hyper_connection.combine(
                hidden_states, hc_residual
            )
            return hidden_states, None

        # Non-hc path.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            self_attention_output = self.linear_attn(hidden_states=hidden_states)
        elif self.layer_type == "full_attention":
            self_attention_output = self.self_attn(
                hidden_states=hidden_states,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if self.layer_scale:
            hidden_states = self._apply_layer_scale(
                hidden_states, self.attn_layer_scale
            )
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale:
            hidden_states = self._apply_layer_scale(hidden_states, self.ffn_layer_scale)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "query_start_loc": 0,
        "ngram_context": 0,
        "deepstack_input_embeds": 0,
    }
)
class Qwen3_8FlashNextModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.vocab_size = config.vocab_size
        self._qsa_layer_ids = frozenset(
            layer_idx
            for layer_idx, layer_type in enumerate(config.layer_types)
            if layer_type == "full_attention"
            and getattr(config, "indexer_n_heads", None) is not None
        )
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str) -> Qwen3_8FlashNextDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return Qwen3_8FlashNextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[layer_idx],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.use_hc = bool(config.use_hc)
        intermediate_size = (
            config.hidden_size * config.hc_count if self.use_hc else config.hidden_size
        )
        intermediate_keys = (
            ["hidden_states"] if self.use_hc else ["hidden_states", "residual"]
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            intermediate_keys, intermediate_size
        )

        self.hc_final_method = config.hc_final_method
        self.hc_has_final_norm = self.use_hc and (
            self.hc_final_method != "hyperconnection_average"
        )
        self.hyper_connection_mixer: GatedResidualSimple | None
        if get_pp_group().is_last_rank and self.hc_has_final_norm:
            if self.hc_final_method != "gated_residual_simple":
                raise ValueError(
                    f"Unsupported hc_final_method {self.hc_final_method!r}"
                )
            hc_config = HyperConnectionConfig(
                hc_count=getattr(config, "hc_count", 4),
                hidden_size=config.hidden_size,
                params_dtype=torch.bfloat16,
                init_method_std=0.02,
                mtp_hc=getattr(config, "mtp_hc", False),
                hc_lowrank=getattr(config, "hc_lowrank", 128),
                rms_norm_eps=config.rms_norm_eps,
                hc_per_branch_norm=getattr(config, "hc_per_branch_norm", False),
            )
            self.hyper_connection_mixer = GatedResidualSimple(
                hc_config, use_combine=False, role="final"
            )
            self.norm = PPMissingLayer()
        elif get_pp_group().is_last_rank:
            self.hyper_connection_mixer = None
            self.norm = Qwen3_8FlashNextRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                pre_affine=getattr(config, "pre_affine", False),
                gated_layernorm=getattr(config, "gated_layernorm", False),
                gated_layernorm_lowrank=getattr(config, "gated_layernorm_lowrank", 16),
                use_gemma_rms_norm=getattr(config, "use_gemma_rms_norm", True),
            )
        else:
            self.hyper_connection_mixer = None
            self.norm = PPMissingLayer()

        spec_config = vllm_config.speculative_config
        # MTP HC multi-stream outputs: when speculative method=="mtp" and the
        # model uses HC with hc_count>1, retain the pre-final-mixer multi-stream
        # residual [T, hc_count*H] so the MTP drafter can feed a real
        # multi-stream backbone hidden on its first step (scheme A). Derived
        # purely from config (NOT node identity) so P/D nodes stay consistent.
        needs_mtp_hidden = (
            self.use_hc
            and bool(getattr(config, "mtp_hc", False))
            and int(getattr(config, "hc_count", 1)) > 1
            and spec_config is not None
            and getattr(spec_config, "method", None) == "mtp"
            and get_pp_group().is_last_rank
        )
        if needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.hc_count * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
            )
        else:
            self._mtp_hidden_buffer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        ngram_context: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids or inputs_embeds is required")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
            if self.use_hc:
                # use_hc: expand hidden_states to [hc_count * hidden_size]
                hidden_states = hidden_states.repeat(1, self.config.hc_count)
        else:
            if intermediate_tensors is None:
                raise ValueError("pipeline stage requires intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None if self.use_hc else intermediate_tensors["residual"]

        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                deepstack_embed = deepstack_input_embeds[
                    f"deepstack_input_embeds_{layer_idx}"
                ]
                if self.use_hc:
                    deepstack_embed = (
                        deepstack_embed.unsqueeze(-2)
                        .expand(
                            *deepstack_embed.shape[:-1],
                            self.config.hc_count,
                            self.config.hidden_size,
                        )
                        .flatten(-2)
                    )
                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            output = {"hidden_states": hidden_states}
            if not self.use_hc:
                output["residual"] = residual
            return IntermediateTensors(output)

        if self.use_hc:
            if self._mtp_hidden_buffer is not None:
                # Capture the pre-final-mixer multi-stream residual
                # [T, hc_count*H] for the MTP drafter (zero extra compute:
                # this tensor is needed by the final mixer regardless).
                num_tokens = hidden_states.shape[0]
                self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states)
            if self.hyper_connection_mixer is not None:
                hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
            else:
                hidden_states = hidden_states.unflatten(
                    -1, (self.config.hc_count, self.config.hidden_size)
                ).mean(dim=-2)
                hidden_states = self.norm(hidden_states)
        else:
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = (
            (
                _remap_qsa_cache_scale_name(name, self._qsa_layer_ids),
                weight,
            )
            for name, weight in weights
        )
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        # Non-persistent PLE state rebuilt in __init__; skip any ckpt
        # column for them.
        skip_substrs = ["hashstats_"]
        if self.hc_has_final_norm:
            skip_substrs.append("hyper_connection_mixer.block_inject_weight")
        loader = AutoWeightsLoader(
            self,
            skip_substrs=skip_substrs,
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        loaded = loader.load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )
        for module_name, module in self.named_modules():
            if not isinstance(module, Qwen3_8FlashNextNGramEmbedding):
                continue
            token_lookup_name = f"{module_name}.token_lookup"
            module.finalize_token_lookup(token_lookup_name in loaded)
        return loaded


class Qwen3_8FlashNextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."}
    )
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Qwen3_8FlashNextTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = vllm_config.scheduler_config
        if vllm_config.cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.8-Flash-Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.model = Qwen3_8FlashNextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        # Set MoE hyperparameters
        if (getattr(config, "num_experts", 0) or 0) > 0:
            QwenNextMixtureOfExperts.set_moe_parameters(self)

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Forward kwargs unchanged so the runner's _maybe_add_ngram_kwargs
        # path (query_start_loc / ngram_context) reaches Qwen3_8FlashNextModel.
        # When use_ple=False the runner doesn't inject them, kwargs is {},
        # and behavior is bit-wise identical to the pre-PLE version.
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            **kwargs,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @classmethod
    def get_ple_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_ple_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        conv_kernel_size = getattr(hf_config, "ple_conv_kernel_size", 4)
        ngram_size = getattr(hf_config, "ngram_size", 1)
        ple_embedding_backend = getattr(hf_config, "ple_embedding_backend", "unigram")
        short_conv_dilation = ngram_size if ple_embedding_backend == "ngram" else 1
        conv_state_len = (conv_kernel_size - 1) * short_conv_dilation
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        use_hc = bool(getattr(hf_config, "use_hc", False))
        hc_count = int(getattr(hf_config, "hc_count", 4)) if use_hc else 1
        hc_hidden_size = hf_config.hidden_size * hc_count
        conv_channels = hc_hidden_size if use_hc else hf_config.hidden_size
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=conv_channels,
            conv_kernel=conv_state_len + 1,
            num_spec=num_spec,
        )

    @classmethod
    def get_gdn_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_gdn_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return cls.get_gdn_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return cls.get_gdn_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        copy_funcs_by_type = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (
                MambaStateCopyFuncCalculator.short_conv_state_copy_func()
            ),
        }
        missing_types = mamba_types - copy_funcs_by_type.keys()
        assert not missing_types, f"missing state copy funcs for {missing_types}"
        return {
            mamba_type: copy_funcs_by_type[mamba_type] for mamba_type in mamba_types
        }

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        """Return all MambaSpecs for this model (GDN layers + PLE layer).

        The PLE layer uses a separate short_conv MambaSpec whose page_size_bytes
        may exceed the GDN spec; callers should take the maximum.
        """
        config = vllm_config.model_config.hf_text_config
        specs = [
            MambaSpec(
                shapes=cls.get_gdn_mamba_state_shape_from_config(vllm_config),
                dtypes=cls.get_gdn_mamba_state_dtype_from_config(vllm_config),
                block_size=-1,
            )
        ]
        if config.use_ple and config.ple_layer_ids:
            specs.append(
                MambaSpec(
                    shapes=cls.get_ple_mamba_state_shape_from_config(vllm_config),
                    dtypes=cls.get_ple_mamba_state_dtype_from_config(vllm_config),
                    block_size=-1,
                )
            )
        return tuple(specs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.model._mtp_hidden_buffer

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        del mm_features
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen3_8FlashNextMixtureOfExperts(MixtureOfExperts):
    """Expose Qwen3.8-Flash-Next routed experts through vLLM's EPLB protocol."""

    language_model: Qwen3_8FlashNextForCausalLM

    def _set_moe_parameters(self) -> None:
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        language_model = getattr(self, "model", None)
        if language_model is None:
            language_model = self.language_model.model
        for layer in language_model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)

        self.num_moe_layers = len(self.moe_layers)
        if example_moe is None:
            self.num_expert_groups = 1
            self.num_shared_experts = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_redundant_experts = 0
            return

        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for moe in self.moe_mlp_layers:
            moe.n_physical_experts = num_physical_experts
            moe.n_local_physical_experts = num_local_physical_experts
            moe.n_redundant_experts = self.num_redundant_experts
            moe.experts.update_expert_map()


class Qwen3_8FlashNextProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self) -> Qwen3_8FlashNextConfig:
        return self.ctx.get_hf_config(Qwen3_8FlashNextConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_8FlashNextProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3_8FlashNextForConditionalGeneration(
    Qwen3_5ForConditionalGeneration,
    HasInnerState,
    Qwen3_8FlashNextMixtureOfExperts,
):
    """Qwen3-VL vision tower backed by the Qwen3.8-Flash-Next language model."""

    requires_raw_input_tokens = True

    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping

    @staticmethod
    def get_model_state_cls():
        from .model_state import Qwen3_8FlashNextModelState

        return Qwen3_8FlashNextModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config: Qwen3_8FlashNextConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError(
                "Qwen3_8FlashNextForConditionalGeneration requires multimodal_config"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config.text_config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (
            not self.language_model_only
            and hasattr(config.vision_config, "deepstack_visual_indexes")
            and not isinstance(self.visual, StageMissingLayer)
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    config.text_config.hidden_size,
                )
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3_8FlashNextForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        if not get_pp_group().is_first_rank and self.use_deepstack:
            assert self.language_model.model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )
        self._set_moe_parameters()

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError(
                "Qwen3.8-Flash-Next language_model_only does not accept "
                "multimodal embeddings"
            )

        is_multimodal = _require_is_multimodal(is_multimodal)
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = (
                self._compute_deepstack_embeds(
                    inputs_embeds=inputs_embeds,
                    multimodal_embeddings=multimodal_embeddings,
                    is_multimodal=is_multimodal,
                )
            )
        else:
            deepstack_input_embeds = None

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        return self.language_model.get_mtp_target_hidden_states()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            query_start_loc=kwargs.get("query_start_loc"),
            ngram_context=kwargs.get("ngram_context"),
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["visual."] if self.language_model_only else None,
            skip_substrs=["mtp."],
            ignore_unexpected_suffixes=_QWEN38_FLASH_NEXT_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(
        cls,
        mamba_types: set[MambaAttentionBackendEnum],
    ) -> MambaStateCopyFuncsByType:
        return Qwen3_8FlashNextForCausalLM.get_mamba_state_copy_funcs(mamba_types)

    @classmethod
    def get_mamba_specs_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[MambaSpec, ...]:
        return Qwen3_8FlashNextForCausalLM.get_mamba_specs_from_config(vllm_config)


__all__ = [
    "Qwen3_8FlashNextDecoderLayer",
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextModel",
    "Qwen3_8FlashNextRMSNorm",
    "Qwen3_8FlashNextSparseMoeBlock",
]
