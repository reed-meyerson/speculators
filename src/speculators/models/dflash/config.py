from typing import Any, Literal

from pydantic import Field, field_serializer, field_validator, model_validator
from transformers import AutoConfig, PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)

from speculators import SpeculatorModelConfig

__all__ = [
    "DFlashSpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash")
class DFlashSpeculatorConfig(SpeculatorModelConfig):
    """
    Configuration for DFlash speculator with vocabulary mapping.

    DFlash features vocabulary mapping between draft (64K) and target (128K)
    vocabularies, enabling cross-tokenizer speculation.

    :param transformer_layer_config: Configuration for the transformer decoder layer
    :param draft_vocab_size: Size of draft model vocabulary for speculation
    """

    speculators_model_type: Literal["dflash"] = "dflash"
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlashSpeculator"],
        description="Model architectures that can load these weights",
    )

    transformer_layer_config: PretrainedConfig = Field(
        default_factory=Qwen3Config,
        description="Configuration for the transformer decoder layer",
    )

    draft_vocab_size: int = Field(
        default=32000,
        description="Size of draft model vocabulary for speculation",
    )

    block_size: int = Field(
        default=8,
        description=(
            "Default size of the draft block predicted with a forward pass of the model"
        ),
    )

    target_hidden_size: int | None = Field(
        default=None,
        description="Hidden size of the target model (if different from draft model)",
    )

    aux_hidden_state_layer_ids: list[int] | None = Field(
        default=None,
        description="Layer IDs of the DFlash auxiliary hidden state layers",
    )

    aux_hidden_state_scales: list[float] | None = Field(
        default=None,
        description=(
            "Per-slot pre-fc input scales, aligned index-for-index with "
            "aux_hidden_state_layer_ids in CONFIG order (not sorted). Applied "
            "to the distillation branch only (the token-only branch feeds "
            "embeddings, and the verifier-side target construction is never "
            "scaled). None = raw states; every existing checkpoint loads "
            "unchanged. Typical use: scale_l = rms(layer 0) / rms(layer l), "
            "measured offline on a captured pool (see "
            "scripts/estimate_layer_rms.py --emit-scales), to level the ~400x "
            "per-slot RMS spread of verifier hidden states before the fc."
        ),
    )

    mask_token_id: int | None = Field(
        default=None,
        description="Token ID used for masking",
    )

    sliding_window_non_causal: bool = Field(
        default=False,
        description="Use non-causal (bidirectional) masking within draft blocks for "
        "sliding window attention layers. Full attention layers are always "
        "bidirectional.",
    )

    sample_from_anchor: bool = Field(
        default=False,
        description=(
            "Whether to sample from the anchor position. "
            "False: anchor is the bonus token, only mask tokens predict "
            "(block_size-1 speculative tokens). "
            "True: sample from anchor and all mask positions "
            "(block_size speculative tokens). "
        ),
    )

    @field_serializer("transformer_layer_config")
    def serialize_transformer_config(self, value: PretrainedConfig) -> dict:
        """Serialize transformer config to dict."""
        return value.to_diff_dict()

    @field_validator("transformer_layer_config", mode="before")
    @classmethod
    def validate_transformer_config(cls, value: Any) -> PretrainedConfig:
        """Validate and convert transformer config."""
        if isinstance(value, dict):
            config_class: type[PretrainedConfig] = Qwen3Config
            if "model_type" in value:
                config_class = AutoConfig.for_model(
                    model_type=value["model_type"]
                ).__class__
            return config_class(**value)
        return value

    @model_validator(mode="after")
    def validate_aux_scales(self) -> "DFlashSpeculatorConfig":
        """Scales must be positive and aligned with the layer-id list."""
        scales = self.aux_hidden_state_scales
        if scales is None:
            return self
        ids = self.aux_hidden_state_layer_ids
        if ids is None:
            raise ValueError(
                "aux_hidden_state_scales requires aux_hidden_state_layer_ids "
                "to be set."
            )
        if len(scales) != len(ids):
            raise ValueError(
                f"aux_hidden_state_scales has {len(scales)} entries but "
                f"aux_hidden_state_layer_ids has {len(ids)}; they must be "
                "aligned index-for-index in config order."
            )
        bad = [s for s in scales if s <= 0]
        if bad:
            raise ValueError(
                f"aux_hidden_state_scales must be strictly positive; got {bad}."
            )
        return self

    @property
    def target_vocab_size(self) -> int:
        """Get target vocabulary size from transformer config."""
        return self.transformer_layer_config.vocab_size
