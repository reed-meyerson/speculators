# Experiment Log — DSpark pretraining on Qwen3.8-27B

- Fork: `reed-meyerson/speculators`
- Branch: `qwen38-27b-dspark-pretraining`, cut from `vllm-project/speculators` main @ `e3041dd`
- Prior art: RFC [#1127](https://github.com/vllm-project/speculators/issues/1127) and its implementation on `dflash-pretraining`

## 0. Purpose of this fork

Get **pretraining and finetuning (distillation) working end-to-end for DSpark on the Qwen3.8-27B verifier**, with the minimum machinery required to do so.

This is an experiment fork, not a generalization effort. The guiding rule is: **minimal compatibility, deliberately**. Concretely:

- **One verifier, one speculator.** Qwen3.8-27B and DSpark only. Everything that exists to support other targets — embedding-scaling conventions for Gemma/ Granite, DFlash/DFlash2 layering, EAGLE3, pruned-vocab generality, arbitrary layer sweeps — is out of scope and may be deleted or bypassed rather than maintained.
- **Hardcoding is fine.** Layer ids, block semantics, model-type dispatch, and anything else that would otherwise need a config-resolution path can be hardcoded to this model. If a check would only ever fire for some other architecture, it can go.
- **The upstreamable version already exists.** The `dflash-pretraining` branch is the general implementation of the RFC; this fork ports the minimum of it (and diverges freely) so we can run the experiment instead of reviewing compat. If results justify it, upstreaming happens later, from the general branch.

The experiment itself: a two-stage recipe — pretrain the DSpark draft on raw text (no verifier forward pass, no regeneration, no hidden-state extraction), then run ordinary on-policy distillation against Qwen3.8-27B warm-started from the pretrained checkpoint — and enough evidence (val loss, val EAL, test MAL) to judge whether the warm start helps at this scale.

### Target of record

`Qwen/Qwen3.8-27B` — `model_type=qwen3_5` (text backbone `qwen3_5_text`), 64 decoder layers, hidden_size 5120, vocab 248320. Qwen-family convention: layer 0 consumes the **unscaled** embedding, so pretraining can substitute `embed_tokens(input_ids)` for the layer-0 aux hidden state directly.

## Entries

Append-only, newest last. Format: `### N. <date> — <title>`, then what changed, what ran, what it showed. Divergences from `dflash-pretraining` are called out where they happen.

(none yet)
