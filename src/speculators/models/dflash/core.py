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
from speculators.losses.targets import IGNORE_INDEX
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
    verifier_scales_input_embedding,
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
        self.sliding_window = tl_config.sliding_window
        self.sliding_window_indices = [
            i
            for i, layer_type in enumerate(tl_config.layer_types)
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

    def prepare_for_pretraining(self) -> None:
        """Zero the ``fc`` columns pretraining cannot train.

        Pretraining drives only the layer-0 (embedding) slot, so the remaining
        aux slots would otherwise reach the distillation run at their random
        init and swamp the pretrained features the moment real verifier hidden
        states arrive. Zeroed, they receive no input and therefore no gradient
        during pretraining, so the saved checkpoint starts distillation at an
        exact identity and grows the aux contributions from nothing. No
        checkpoint surgery is needed to move between the two.
        """
        if verifier_scales_input_embedding(self.config):
            raise ValueError(
                "Pretraining substitutes the draft's frozen embedding for the "
                "verifier's layer-0 hidden state, but this verifier scales its "
                "embedding before the first layer, so the two differ and the "
                "draft would train on mis-scaled features. See "
                "SCALED_EMBEDDING_MODEL_TYPES."
            )
        keep = self._embedding_fc_columns
        with torch.no_grad():
            mask = torch.zeros_like(self.fc.weight)
            mask[:, keep] = 1.0
            self.fc.weight.mul_(mask)

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

        return cls._finalize_training_model(config, t2d, d2t, **kwargs)

    @classmethod
    def _finalize_training_model(cls, config, t2d, d2t, **kwargs):
        """Build, load verifier-owned weights, and apply training-mode setup.

        Shared by every DFlash-family ``from_training_args``.
        """
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        if kwargs.get("training_mode") == "pretrain":
            model.prepare_for_pretraining()
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
            "training_mode": kwargs.get("training_mode", "distill"),
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

    @property
    def _embedding_fc_columns(self) -> slice:
        """Columns of ``fc`` that consume verifier layer 0, i.e. the embedding.

        Pretraining has only the embedding to project, so it drives exactly the
        aux slot that distillation fills with verifier layer 0. This is what
        lets a pretrained checkpoint load into a distillation run unchanged.
        """
        try:
            slot = list(self.target_layer_ids).index(0)
        except ValueError:
            raise ValueError(
                "Pretraining projects the frozen input embedding through the "
                "`fc` slot reserved for verifier layer 0, so layer 0 must be "
                f"among --target-layer-ids (got {list(self.target_layer_ids)}). "
                "Pass it explicitly, e.g. --target-layer-ids 0 "
                f"{' '.join(str(i) for i in list(self.target_layer_ids)[1:])}, "
                "and give the distillation run the same ids: they set the width "
                "of `fc`, so a checkpoint only loads back into a matching layer "
                "selection."
            ) from None
        return slice(slot * self.hidden_size, (slot + 1) * self.hidden_size)

    @staticmethod
    def _check_forward_inputs(pretrain, hidden_states, verifier_last_hidden_states):
        if pretrain or (
            hidden_states is not None and verifier_last_hidden_states is not None
        ):
            return
        raise ValueError(
            "Distillation requires `hidden_states` and "
            "`verifier_last_hidden_states`; pass training_mode='pretrain' to "
            "train from token ids alone."
        )

    def _project_features(self, hidden_states, input_ids, *, pretrain: bool):
        """Project the per-token features the draft layers attend over."""
        if not pretrain:
            return self.fc(hidden_states)
        # Equivalent to projecting [embedding, 0, ..., 0] through the full fc,
        # without materializing the zeros. The untouched columns get no
        # gradient, so prepare_for_pretraining's zeros survive to the
        # checkpoint and the distillation run resumes from an exact identity.
        return nn.functional.linear(
            self.embed_tokens(input_ids), self.fc.weight[:, self._embedding_fc_columns]
        )

    def _hard_targets(self, input_ids, anchored_block_indices, total_seq_len):
        """Token ids the draft must predict, plus which of them are learnable.

        Aligned to the distillation path: a soft target drawn from verifier
        position ``p`` is a distribution over the token at ``p + 1``.
        """
        label_indices = (
            (anchored_block_indices + 1) % total_seq_len
            if self.config.sample_from_anchor
            else anchored_block_indices
        )
        target_ids = input_ids[:, label_indices]
        if not self.use_draft_vocab:
            return target_ids, torch.ones_like(target_ids, dtype=torch.bool)
        # verifier_lm_head is the verifier head sliced by t2d, so draft index
        # is the running count of kept tokens below this id.
        in_draft = self.t2d[target_ids]
        draft_ids = (self.t2d.long().cumsum(0) - 1)[target_ids]
        return torch.where(in_draft, draft_ids, IGNORE_INDEX), in_draft

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

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor | None = None,  # [1, seq, num_hidden*hidden_size]
        input_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        loss_mask: torch.Tensor | None = None,  # [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor
        | None = None,  # [1, total_seq_len, hidden_size]
        document_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # [1, total_seq_len]
        **kwargs,
    ):
        """Run the anchored-block draft transformer up to the draft logits.

        Returns ``(hidden, logits, targets, aligned_loss_mask,
        anchored_block_indices)``. DSpark reuses this and adds its Markov and
        confidence heads before computing its own loss.

        Under ``training_mode="pretrain"`` the verifier tensors are absent:
        features come from the frozen embedding and targets are hard token ids.
        See :meth:`prepare_for_pretraining`.
        """
        training_mode = kwargs.pop("training_mode", "distill")
        pretrain = training_mode == "pretrain"
        self._check_forward_inputs(pretrain, hidden_states, verifier_last_hidden_states)

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

        fc_output = self.hidden_norm(
            self._project_features(hidden_states, input_ids, pretrain=pretrain)
        )
        # shape: [1, total_seq_len, hidden_size]

        mask_position_ids = get_base_indices_for_anchored_blocks(
            position_ids[0, anchor_positions], self.block_size
        )
        position_ids = torch.cat([position_ids, mask_position_ids.unsqueeze(0)], dim=1)
        # shape: [1, total_seq_len + num_anchors*block_size]

        # the fc_output shape doesn't match position_ids but doesn't need
        # to, as it is only used to set dtype and device in rotary_emb
        position_embeddings = self.rotary_emb(fc_output, position_ids)

        anchored_block_indices = get_base_indices_for_anchored_blocks(
            anchor_positions, self.block_size
        )  # shape: [num_anchors*block_size]

        with torch.no_grad():
            if pretrain:
                targets, hard_label_valid = self._hard_targets(
                    input_ids, anchored_block_indices, total_seq_len
                )
                # shape: [1, num_anchors*block_size]
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

        # Tokens outside a pruned draft vocabulary have no learnable label.
        if pretrain:
            aligned_loss_mask = aligned_loss_mask * hard_label_valid.to(
                aligned_loss_mask.dtype
            )

        return hidden, logits, targets, aligned_loss_mask, anchored_block_indices

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor | None = None,  # [1,seq,num_hidden*hidden_size]
        input_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor | None = None,  # [1, seq, hidden]
        document_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        gamma: float = 4.0,
        max_anchors: int = 512,
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        training_mode: str = "distill",
        **kwargs,
    ):
        _, logits, targets, aligned_loss_mask, _ = self._backbone_forward(
            hidden_states,
            input_ids,
            loss_mask,
            verifier_last_hidden_states,
            document_ids,
            position_ids,
            max_anchors=max_anchors,
            training_mode=training_mode,
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
