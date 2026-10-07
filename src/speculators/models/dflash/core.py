import logging
from typing import ClassVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from transformers import PretrainedConfig
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)

from speculators.losses import LossConfig, resolve_loss_config
from speculators.model import DraftVocabMixin, SpeculatorModel
from speculators.models.attention import create_float_mask
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.metrics import compute_metrics
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from speculators.models.dflash.utils import (
    get_base_indices_for_anchored_blocks,
    select_anchors,
)
from speculators.models.utils import (
    conditional_torch_compile,
    flatten_rope_parameters,
    resolve_target_layer_ids,
    resolve_verifier_norm_class,
)

logger = logging.getLogger(__name__)

# Compile so the mask builds block-sparse instead of materializing DFlash's huge
# dense [Q, KV] grid every step. (No benefit for EAGLE3's small autoregressive mask.)
_compiled_create_block_mask = torch.compile(create_block_mask)


@SpeculatorModel.register("dflash")
class DFlashDraftModel(DraftVocabMixin, SpeculatorModel):
    config_class: ClassVar[type[DFlashSpeculatorConfig]] = DFlashSpeculatorConfig  # type: ignore[misc]
    supports_gradient_checkpointing = True  # noqa: D003  # Qwen3DFlashDecoderLayer inherits GradientCheckpointingLayer
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = [  # type: ignore[misc]
        "embed_tokens.weight",
        "verifier_norm.weight",
        # verifier_lm_head is reloaded from the verifier (see load_verifier_weights)
        # and excluded on save, so it is expected to be absent from checkpoints.
        # lm_head is handled per-instance in __init__: omitted only for full-vocab
        # drafts, where it is the exact frozen verifier projection.
        "verifier_lm_head.weight",
        "t2d",
        "d2t",
    ]
    _keys_to_ignore_on_save: ClassVar[list[str]] = [  # type: ignore[misc,assignment]
        "verifier_lm_head.weight",
        "verifier_norm.weight",
    ]

    t2d: torch.Tensor | None
    d2t: torch.Tensor | None

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> nn.Module:
        """Build one draft decoder layer.

        DFlash-family variants override this factory when they wrap the shared
        attention and MLP with additional modules.
        """
        return Qwen3DFlashDecoderLayer(config.transformer_layer_config, layer_idx)  # type: ignore[arg-type]

    def __init__(
        self,
        config: DFlashSpeculatorConfig,
    ) -> None:
        # Forcibly override config settings
        if config.transformer_layer_config._attn_implementation is None:  # noqa: SLF001
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                "simple_flex_attention"
            )
        self._attn_impl = config.transformer_layer_config._attn_implementation  # noqa: SLF001
        self._create_mask_fn = (
            _compiled_create_block_mask
            if self._attn_impl == "simple_flex_attention"
            else create_float_mask
            if self._attn_impl == "eager"
            else create_mask
        )
        super().__init__(config=config)
        self._init_vocab(config)

        tl_config = config.transformer_layer_config

        # Number of draft layers is encoded in transformer_layer_config
        num_draft_layers = tl_config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                self._make_decoder_layer(config, layer_idx)
                for layer_idx in range(num_draft_layers)
            ]
        )
        # Both are optional: declared only for alternating/sliding attention.
        self.sliding_window = getattr(tl_config, "sliding_window", None)
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(
                getattr(tl_config, "layer_types", None) or []
            )
            if layer_type == "sliding_attention"
        ]
        self.uses_sliding_window_attn = bool(self.sliding_window_indices)
        self.uses_full_attn = bool(num_draft_layers - len(self.sliding_window_indices))
        self.sliding_window_non_causal = config.sliding_window_non_causal

        self.norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        rotary_config = flatten_rope_parameters(config.transformer_layer_config)
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)  # type: ignore[arg-type]

        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.transformer_layer_config.hidden_size,
            config.transformer_layer_config.hidden_size,
            bias=False,
        )
        # Entry 21: per-slot pre-fc input scales (config.aux_hidden_state_scales)
        # are applied ONLY in the distillation branch of _backbone_forward,
        # never to the token-only (embedding) branch or the verifier-side
        # target construction. The tensor is built lazily from the CONFIG (see
        # _aux_scale_tensor) rather than registered as a buffer:
        # transformers' from_pretrained constructs models under meta-device
        # init, where register_buffer produces meta tensors that to_empty()
        # later materializes as UNINITIALIZED memory for anything absent from
        # the checkpoint (non-persistent buffers are, by design). The config
        # object is always real, so it is the single source of truth.
        self.hidden_norm = Qwen3RMSNorm(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm = resolve_verifier_norm_class(config)(
            config.transformer_layer_config.hidden_size,
            eps=config.transformer_layer_config.rms_norm_eps,  # type: ignore[arg-type]
        )
        self.verifier_norm.weight.requires_grad = False
        self.block_size = config.block_size

        # Warn if using DFlash with sample_from_anchor=True (may not be supported)
        if type(self).__name__ == "DFlashDraftModel" and config.sample_from_anchor:
            logger.warning(
                "DFlash with sample_from_anchor=True may not be supported in "
                "all inference engines (e.g., vLLM). Verify compatibility with your "
                "deployment target."
            )

        self.post_init()
        # Verifier-owned weights are reconstructed on load. Keep a reduced-vocab
        # lm_head serialized because current runtimes cannot derive it from the
        # full verifier head.
        #
        # Shadow the ClassVar lists with per-instance copies so full- and
        # reduced-vocabulary siblings cannot mutate each other's save rules.
        keys_to_ignore_on_save = list(type(self)._keys_to_ignore_on_save)  # noqa: SLF001
        keys_to_ignore_on_load_missing = list(
            type(self)._keys_to_ignore_on_load_missing  # noqa: SLF001
        )
        keys_to_ignore_on_save.append("embed_tokens.weight")
        if not self.use_draft_vocab:
            keys_to_ignore_on_save.append("lm_head.weight")
            keys_to_ignore_on_load_missing.append("lm_head.weight")
        self.__dict__["_keys_to_ignore_on_save"] = keys_to_ignore_on_save
        self.__dict__["_keys_to_ignore_on_load_missing"] = (
            keys_to_ignore_on_load_missing
        )

    @property
    def target_layer_ids(self) -> list[int]:
        """Target layer IDs for auxiliary hidden states."""
        return self.config.aux_hidden_state_layer_ids

    def _aux_scale_tensor(self, device: torch.device, dtype: torch.dtype):
        """Lazily cached [1, 1, num_slots, 1] scale tensor for the fc input.

        Built from ``config.aux_hidden_state_scales`` (None -> returns None,
        meaning: no scaling). Cached in ``__dict__`` keyed implicitly by the
        requested (device, dtype); a plain tensor attribute so meta-device
        model init / ``to_empty()`` never touches it.
        """
        scales = self.config.aux_hidden_state_scales
        if scales is None:
            return None
        cached = self.__dict__.get("_aux_input_scales_cache")
        if (
            cached is None
            or cached.device != device
            or cached.dtype != dtype
        ):
            cached = (
                torch.tensor(scales, device=device, dtype=dtype)
                .view(1, 1, len(scales), 1)
                .contiguous()
            )
            self.__dict__["_aux_input_scales_cache"] = cached
        return cached

    def load_verifier_weights(self):
        """Reconstruct weights intentionally omitted from DFlash checkpoints."""
        self._load_verifier_weights(
            overwrite_embed_tokens=True,
            overwrite_lm_head=not self.use_draft_vocab,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: "PretrainedConfig",
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlashDraftModel":
        """Create DFlash model from training arguments.

        Args:
            verifier_config: Verifier model configuration. This should be a config
                with num_hidden_layers set to the number of DRAFT layers (created
                by create_transformer_layer_config in train.py).
            t2d: Target-to-draft vocabulary mapping tensor (optional)
            d2t: Draft-to-target vocabulary mapping tensor (optional)
            **kwargs: Training arguments with DFlash-specific params
                - draft_vocab_size: Size of draft vocabulary
                - block_size: Block size for draft predictions (default: 8)
                - verifier_name_or_path: Path to verifier model

        Returns:
            Initialized DFlashDraftModel

        Note:
            The number of draft layers is encoded in verifier_config.num_hidden_layers,
            following the same pattern as EAGLE3.
        """
        config = DFlashSpeculatorConfig(
            **cls._build_base_config_kwargs("dflash", verifier_config, **kwargs)
        )

        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def _build_base_config_kwargs(
        algorithm: str,
        verifier_config: "PretrainedConfig",
        **kwargs,
    ) -> dict:
        """Shared DFlash-family config kwargs for ``from_training_args``.

        DSpark reuses this and appends its Markov/confidence/loss fields.
        """
        from speculators.config import (  # noqa: PLC0415
            SpeculatorsConfig,
            VerifierConfig,
        )
        from speculators.proposals.greedy import (  # noqa: PLC0415
            GreedyTokenProposalConfig,
        )

        target_layer_ids = resolve_target_layer_ids(
            kwargs.get("target_layer_ids"),
            kwargs["verifier_name_or_path"],
            trust_remote_code=kwargs.get("trust_remote_code", False),
        )
        verifier_config._attn_implementation = kwargs.get(  # noqa: SLF001
            "draft_attn_impl", "simple_flex_attention"
        )
        block_size = kwargs.get("block_size", 8)

        default_sample_from_anchor = algorithm == "dspark"
        sample_from_anchor_arg = kwargs.get("sample_from_anchor")
        sample_from_anchor = (
            default_sample_from_anchor
            if sample_from_anchor_arg is None
            else sample_from_anchor_arg
        )
        default_non_causal = algorithm == "dflash2"
        non_causal_arg = kwargs.get("sliding_window_non_causal")
        sliding_window_non_causal = (
            default_non_causal if non_causal_arg is None else non_causal_arg
        )

        # Calculate speculative tokens based on sample_from_anchor
        # False: anchor is bonus token (block_size - 1 tokens)
        # True: sample from anchor too (block_size tokens)
        speculative_tokens = block_size if sample_from_anchor else block_size - 1

        return {
            "transformer_layer_config": verifier_config,
            "draft_vocab_size": kwargs["draft_vocab_size"],
            "block_size": block_size,
            "aux_hidden_state_layer_ids": target_layer_ids,
            "mask_token_id": kwargs.get("mask_token_id"),
            "sliding_window_non_causal": sliding_window_non_causal,
            "sample_from_anchor": sample_from_anchor,
            "speculators_config": SpeculatorsConfig(
                algorithm=algorithm,
                proposal_methods=[
                    GreedyTokenProposalConfig(speculative_tokens=speculative_tokens)
                ],
                default_proposal_method="greedy",
                verifier=VerifierConfig.from_pretrained(
                    kwargs["verifier_name_or_path"]
                ),
            ),
        }

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Get training and validation kwargs for DFlash.

        Args:
            **kwargs: Training arguments

        Returns:
            Tuple of (train_call_kwargs, val_call_kwargs)
        """
        loss_config = resolve_loss_config(
            kwargs["loss_fn"], kwargs.get("loss_implementation", "fused")
        )
        gamma = kwargs.get("dflash_decay_gamma", 4.0)
        max_anchors = kwargs.get("max_anchors", 512)
        per_position_loss_weight = kwargs.get(
            "per_position_loss_weight", "fixed-exp-decay"
        )
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        shared = {
            "loss_config": loss_config,
            "gamma": gamma,
            "max_anchors": max_anchors,
            "per_position_loss_weight": per_position_loss_weight,
            "dpace_alpha": dpace_alpha,
        }
        return dict(shared), dict(shared)

    @property
    def mask_token_id(self) -> int:
        if self.config.mask_token_id is None:
            raise ValueError(
                "mask_token_id is not set on the config. "
                "Pass --mask-token-id during training or ensure the config "
                "was saved with mask_token_id set."
            )
        return self.config.mask_token_id

    @torch.compiler.disable
    def _create_attention_mask(
        self,
        document_ids: torch.Tensor,
        total_seq_len: int,
        anchor_positions: torch.Tensor,
        device: torch.device,
        sliding_window: int | None = None,
        sliding_window_non_causal: bool = False,
    ):
        mask_mod, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids=document_ids.squeeze(0).to(device),
            total_seq_len=total_seq_len,
            anchor_positions=anchor_positions,
            block_size=self.block_size,
            sliding_window=sliding_window,
            sliding_window_non_causal=sliding_window_non_causal,
        )
        return self._create_mask_fn(
            mask_mod,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

    @torch.compiler.disable
    def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
        total_seq_len = loss_mask.shape[1]

        anchor_positions, anchor_valid = select_anchors(
            loss_mask, max_anchors, self.block_size
        )

        full_attn_mask = None
        if self.uses_full_attn:
            full_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=None,
            )

        sliding_window_attn_mask = None
        if self.uses_sliding_window_attn:
            sliding_window_attn_mask = self._create_attention_mask(
                document_ids=document_ids,
                total_seq_len=total_seq_len,
                anchor_positions=anchor_positions,
                device=device,
                sliding_window=self.sliding_window,
                sliding_window_non_causal=self.sliding_window_non_causal,
            )

        return full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid

    def _hard_target_mode(self, loss_config: LossConfig | None) -> bool:
        """Whether to train against token ids from the input sequence itself.

        Selected when ``--loss-fn ce_token`` meets an aux selection of layer 0
        alone: the layer-0 hidden state is the (unscaled, Qwen-family) input
        embedding, so features are computed from the frozen ``embed_tokens``
        and labels are gathered from ``input_ids`` — no verifier forward
        pass or captured hidden states are consumed. Any other config keeps
        the distillation path unchanged (``ce_token`` then behaves exactly
        like ``ce``).
        """
        return (
            loss_config is not None
            and "ce_token" in loss_config
            and self.target_layer_ids == [0]
        )

    def token_only_data(self, loss_fn: str | None) -> bool:
        """Whether this model trains from token ids alone under ``loss_fn``.

        CLI-boundary mirror of :meth:`_hard_target_mode` so the dataloader
        can skip verifier hidden states whenever they are never consumed.
        """
        if loss_fn is None:
            return False
        return self._hard_target_mode(resolve_loss_config(loss_fn))

    def _backbone_forward(
        self,
        input_ids: torch.Tensor,  # [1, total_seq_len]
        loss_mask: torch.Tensor,  # [1, total_seq_len]
        document_ids: torch.Tensor,  # [1, total_seq_len]
        hidden_states: torch.Tensor | None = None,  # [1, T, n_hidden*hidden]
        verifier_last_hidden_states: torch.Tensor | None = None,  # [1, T, hidden]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        hard_targets: bool = False,
        **kwargs,
    ):
        """Run the anchored-block draft transformer up to the draft logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices)``. DSpark reuses this and adds its Markov and
        confidence heads before computing its own loss.
        """
        # input_ids is [1, total_seq_len] on the batch device: source device
        # and length from it so hard-target (token-only) batches need no
        # hidden states at all.
        device = input_ids.device
        total_seq_len = input_ids.shape[1]
        num_anchors = kwargs.pop("max_anchors", 512)

        if position_ids is None:
            position_ids = torch.arange(
                total_seq_len, dtype=torch.long, device=device
            ).unsqueeze(0)

        full_attn_mask, sliding_window_attn_mask, anchor_positions, anchor_valid = (
            self._build_attention_mask(loss_mask, num_anchors, document_ids, device)
        )

        mask_tokens_size = num_anchors * self.block_size

        mask_token_ids = torch.full(
            (1, mask_tokens_size),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )  # shape: [1, num_anchors*block_size]
        mask_token_ids[:, :: self.block_size] = input_ids[:, anchor_positions]
        noise_embedding = self.embed_tokens(mask_token_ids)
        # shape: [1, num_anchors*block_size, hidden_size]

        if hard_targets:
            fc_output = self.fc(self.embed_tokens(input_ids))
        elif hidden_states is None:
            raise ValueError(
                "Distillation features require verifier hidden states."
                " Token-only batches are only supported with hard targets"
                " (--loss-fn ce_token --target-layer-ids 0)."
            )
        else:
            aux_scales = self._aux_scale_tensor(
                hidden_states.device, hidden_states.dtype
            )
            if aux_scales is not None:
                # Entry 21: scale each aux slot to a common RMS before the fc.
                # The flattened input is slot-major in config order, so
                # reinterpret as [..., num_slots, hidden] and broadcast-multiply.
                # The token-only branch above and the verifier target path
                # below are deliberately untouched.
                num_slots = len(self.target_layer_ids)
                hidden_states = (
                    hidden_states.reshape(
                        *hidden_states.shape[:-1], num_slots, -1
                    )
                    * aux_scales
                ).flatten(-2)
            fc_output = self.fc(hidden_states)
        fc_output = self.hidden_norm(fc_output)
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # rotary_emb only reads dtype and device from its first argument, so
        # the mask-token embedding stands in when hidden_states is absent
        # (token-only batches).
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            if hard_targets:
                # Labels gathered from the sequence itself. Aligned with the
                # distillation path: a soft target drawn from verifier position
                # p is a distribution over the token at p + 1. (Anchor
                # selection excludes the last block_size positions, so the +1
                # never wraps past the sequence end.)
                label_indices = (
                    (anchored_block_indices + 1) % total_seq_len
                    if self.config.sample_from_anchor
                    else anchored_block_indices
                )
                targets = input_ids[:, label_indices]
            elif verifier_last_hidden_states is None:
                raise ValueError(
                    "Distillation targets require verifier_last_hidden_states."
                    " Token-only batches are only supported with hard targets"
                    " (--loss-fn ce_token --target-layer-ids 0)."
                )
            elif anchored_block_indices.numel() < total_seq_len:
                target_indices = (
                    anchored_block_indices
                    if self.config.sample_from_anchor
                    else (anchored_block_indices - 1) % total_seq_len
                )
                targets = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states[:, target_indices])
                )
            else:
                verifier_logits = self.verifier_lm_head(
                    self.verifier_norm(verifier_last_hidden_states)
                )
                if not self.config.sample_from_anchor:
                    verifier_logits = torch.roll(verifier_logits, 1, dims=1)
                targets = verifier_logits[:, anchored_block_indices]
            # shape: [1, num_anchors*block_size, draft_vocab_size]

        for layer_idx, layer in enumerate(self.layers):
            noise_embedding = layer(
                hidden_states=noise_embedding,
                target_hidden=fc_output,
                attention_mask=sliding_window_attn_mask
                if layer_idx in self.sliding_window_indices
                else full_attn_mask,
                position_ids=position_ids,
                use_cache=False,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden = self.norm(noise_embedding)
        logits = self.lm_head(hidden)
        # shape: [1, num_anchors*block_size, vocab_size]

        aligned_loss_mask = loss_mask.clone()[:, anchored_block_indices]
        # shape: [1, num_anchors*block_size]

        # zero out any padded anchor blocks
        aligned_loss_mask = aligned_loss_mask * (
            anchor_valid.repeat_interleave(self.block_size)
            .unsqueeze(0)
            .to(aligned_loss_mask.dtype)
        )  # shape: [1, num_anchors*block_size]

        # For sample_from_anchor=False, mask slot 0 (anchor) since it's not trained
        if not self.config.sample_from_anchor:
            aligned_loss_mask[:, :: self.block_size] = 0

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @conditional_torch_compile
    def forward(
        self,
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        hidden_states: torch.Tensor | None = None,  # [1, T, n_hidden*hidden]
        verifier_last_hidden_states: torch.Tensor | None = None,  # [1, T, hidden]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            input_ids=input_ids,
            loss_mask=loss_mask,
            document_ids=document_ids,
            hidden_states=hidden_states,
            verifier_last_hidden_states=verifier_last_hidden_states,
            position_ids=position_ids,
            max_anchors=max_anchors,
            hard_targets=self._hard_target_mode(loss_config),
            **kwargs,
        )
        loss, metrics = compute_metrics(
            logits,
            targets,
            aligned_loss_mask,
            self.block_size,
            gamma=gamma,
            loss_config=loss_config,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            sample_from_anchor=self.config.sample_from_anchor,
        )
        return None, loss, metrics
