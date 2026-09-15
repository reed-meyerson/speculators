import logging
import warnings
from copy import deepcopy
from functools import cache, partial

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

logger = logging.getLogger(__name__)

# Verifier families whose final norm follows the Gemma convention
# `x_norm * (1 + w)` instead of the plain RMSNorm `x_norm * w`. Matched
# exactly against the verifier's effective `model_type` (after text_config
# extraction) — gemma3n and gemma4 dropped this convention, so prefix
# matching would over-include. Qwen3.5 is in here because transformers'
# Qwen3_5RMSNorm computes `output * (1.0 + weight)` and vLLM's is an alias
# of GemmaRMSNorm.
GEMMA_STYLE_FINAL_NORM_MODEL_TYPES = frozenset(
    (
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "qwen3_5",
        "qwen3_5_text",
        "recurrent_gemma",
    )
)

# Verifier families that scale the token embedding before the first decoder
# layer, and by how much. Their layer-0 auxiliary hidden state is the *scaled*
# embedding -- vLLM captures it after `embed_input_ids`, which folds the factor
# in -- so pretraining reproduces layer 0 by applying the same factor to the
# draft's frozen copy of `embed_tokens`.
#
# Enumerated, not prefix-matched, for the same reason as the final-norm set
# above: the convention is not stable within a family. Unlisted architectures
# are assumed to feed layer 0 the unscaled embedding, which holds for Llama,
# Mistral, Qwen, DeepSeek, GLM, Phi, gpt-oss, MiniMax, OLMo, Nemotron and Kimi.
_SQRT_HIDDEN_SIZE_EMBEDDING_MODEL_TYPES = frozenset(
    (
        "diffusion_gemma",
        "diffusion_gemma_text",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "gemma3n",
        "gemma3n_text",
        "gemma4",
        "gemma4_text",
        "gemma4_unified",
        "gemma4_unified_text",
        "recurrent_gemma",
        "t5gemma",
        "t5gemma2",
        "t5gemma2_text",
        "vaultgemma",
    )
)

# Granite exposes its factor directly on the config instead of deriving it.
_EMBEDDING_MULTIPLIER_MODEL_TYPES = frozenset(
    (
        "granite",
        "granite_swa",
        "granitemoe",
        "granitemoe_swa",
        "granitemoehybrid",
        "granitemoeshared",
        "hyperclovax",
        "minicpm3",
    )
)


@cache
def _verifier_model_type(name_or_path: str) -> str | None:
    """The verifier's effective model_type, or None when unresolvable."""
    try:
        verifier_config = get_verifier_config(name_or_path)
    except (KeyError, OSError, ValueError):
        logger.warning(
            "Could not resolve a config for verifier %s; assuming the plain "
            "final-norm convention (x * w).",
            name_or_path,
        )
        return None
    return str(getattr(verifier_config, "model_type", "") or "").lower()


def uses_gemma_style_final_norm(config) -> bool:
    """Whether the verifier's final norm applies gain ``1 + w`` (not ``w``)."""
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    name_or_path = getattr(verifier, "name_or_path", None)
    if not name_or_path:
        return False
    model_type = _verifier_model_type(name_or_path)
    return model_type is not None and model_type in GEMMA_STYLE_FINAL_NORM_MODEL_TYPES


@cache
def _embedding_scale_for(name_or_path: str) -> float:
    try:
        verifier_config = get_verifier_config(name_or_path)
    except (KeyError, OSError, ValueError):
        logger.warning(
            "Could not resolve a config for verifier %s; assuming it feeds "
            "layer 0 the unscaled embedding.",
            name_or_path,
        )
        return 1.0
    model_type = str(getattr(verifier_config, "model_type", "") or "").lower()
    if model_type in _SQRT_HIDDEN_SIZE_EMBEDDING_MODEL_TYPES:
        return float(verifier_config.hidden_size) ** 0.5
    if model_type in _EMBEDDING_MULTIPLIER_MODEL_TYPES:
        return float(getattr(verifier_config, "embedding_multiplier", 1.0))
    return 1.0


def verifier_embedding_scale(config) -> float:
    """Factor the verifier applies to its embedding before the first layer.

    ``1.0`` for the large majority of architectures. Pretraining multiplies the
    draft's frozen embedding by this to reproduce the verifier's layer-0
    auxiliary hidden state, which is captured after the scaling.
    """
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    name_or_path = getattr(verifier, "name_or_path", None)
    if not name_or_path:
        return 1.0
    return _embedding_scale_for(name_or_path)


def resolve_verifier_norm_class(config) -> type:
    """The RMSNorm class matching the verifier's final-norm weight convention.

    The frozen ``verifier_norm`` must apply the same gain convention the
    verifier was trained under (`x * (1 + w)` for the Gemma/Qwen3.5
    families, `x * w` otherwise), or the reconstructed verifier targets
    are silently mis-scaled.
    """
    if uses_gemma_style_final_norm(config):
        return Gemma3RMSNorm
    return Qwen3RMSNorm


def conditional_torch_compile(func=None, *args, **kwargs):
    if func is None:
        return partial(conditional_torch_compile, *args, **kwargs)
    if torch.cuda.is_available() and hasattr(torch, "compile"):
        return torch.compile(func, *args, **kwargs)
    return func


def get_verifier_config(
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> PretrainedConfig:
    verifier_config = AutoConfig.from_pretrained(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if hasattr(verifier_config, "text_config"):
        verifier_config = verifier_config.text_config
    return verifier_config


DEFAULT_TARGET_LAYER_IDS_WARNING = (
    "--target-layer-ids is not explicitly set. Setting target "
    "layers to {target_layer_ids}. If custom target layers were used "
    "when launching vllm datagen, please set them explicitly."
)


def resolve_target_layer_ids(
    target_layer_ids: list[int] | None,
    verifier_name_or_path: str,
    trust_remote_code: bool = False,
) -> list[int]:
    num_layers = get_verifier_config(
        verifier_name_or_path,
        trust_remote_code=trust_remote_code,
    ).num_hidden_layers

    if target_layer_ids is None:
        explicit = False
        target_layer_ids = [2, num_layers // 2, num_layers - 3]
    else:
        explicit = True

    # Layer id ``num_layers`` (the final hidden state) is valid, matching the
    # ids scripts/launch_vllm.py emits with --include-last-layer.
    invalid = (
        not target_layer_ids
        or min(target_layer_ids) < 0
        or max(target_layer_ids) > num_layers
        or len(set(target_layer_ids)) != len(target_layer_ids)
    )
    if invalid:
        if explicit:
            raise ValueError(
                f"target_layer_ids must be distinct and within [0, {num_layers}] "
                f"for a verifier with {num_layers} hidden layers, "
                f"got {target_layer_ids}"
            )
        raise ValueError(
            f"Default target layer ids {target_layer_ids} are invalid for a verifier "
            f"with {num_layers} hidden layers; pass --target-layer-ids explicitly."
        )

    if not explicit:
        warnings.warn(
            DEFAULT_TARGET_LAYER_IDS_WARNING.format(target_layer_ids=target_layer_ids),
            stacklevel=3,
        )
    return target_layer_ids


def flatten_rope_parameters(config: PretrainedConfig) -> PretrainedConfig:
    """Flatten nested per-layer-type ``rope_parameters`` for rotary embedding init.

    Models like Laguna store separate rope configs per layer type
    (``sliding_attention``, ``full_attention``). Rotary embedding classes expect
    a flat dict with ``rope_type``/``rope_theta`` at the top level. This helper
    selects the ``sliding_attention`` variant when nested parameters are detected
    and returns a deep-copied config; otherwise returns the original unchanged.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or "sliding_attention" not in rope_params:
        return config
    config = deepcopy(config)
    config.rope_parameters = rope_params["sliding_attention"]
    return config


def resolve_draft_intermediate_size(verifier_config: PretrainedConfig) -> int:
    """Resolve a dense draft MLP ``intermediate_size`` from a verifier config.

    The draft is an independent small *dense* decoder, so its FFN width is a design
    choice rather than something to reconcile with the verifier's routed capacity:

    * Dense verifiers expose ``intermediate_size`` directly; the draft mirrors it.
    * MoE verifiers have no dense ``intermediate_size`` (their FFN is a routed set of
      small experts), so the draft falls back to the widely used ``3 * hidden_size``
      gated-MLP ratio -- the Qwen3 dense convention that the dflash draft decoder
      follows. Pass ``--draft-config`` to set it explicitly instead.

    :raises ValueError: when the verifier config exposes neither ``intermediate_size``
        nor ``hidden_size`` (degenerate config; pass ``--draft-config``).
    """
    dense = getattr(verifier_config, "intermediate_size", None)
    if dense is not None:
        return int(dense)

    hidden_size = getattr(verifier_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Verifier config exposes neither `intermediate_size` nor `hidden_size`, "
            "so a draft intermediate_size cannot be inferred. Pass --draft-config to "
            "set the draft architecture explicitly."
        )

    intermediate_size = 3 * int(hidden_size)
    warnings.warn(
        "Verifier config has no dense intermediate_size (likely MoE); using draft "
        f"intermediate_size={intermediate_size} (3 x hidden_size = {hidden_size}). "
        "Pass --draft-config to override.",
        stacklevel=3,
    )
    return intermediate_size
