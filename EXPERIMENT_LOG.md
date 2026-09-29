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

**No draft-vocab pruning.** The draft trains on the full 248320-token verifier vocab. Every piece of machinery that exists only for pruned vocabularies — the `t2d`/`d2t` mappings, hard-label remapping, the `IGNORE_INDEX`/`unlearnable` masking of corpus tokens outside the draft vocab — can be assumed away.

## Entries

Append-only, newest last. Format: `### N. <date> — <title>`, then what changed, what ran, what it showed. Divergences from `dflash-pretraining` are called out where they happen.

### 1. 2026-09-29 — hard-label pretraining path (`ce_token` + `[0]`): implemented and validated e2e

**What changed** (commits `ec47ac0`, `b189bbd`):

- `ec47ac0` — **ce_token loss.** Hard-label CE on verifier token ids, registered in both loss maps (name + JSON spec) so `--loss-fn ce_token` resolves cleanly. Fused and eager implementations; DSpark/DFlash metric heads (markov, confidence) train alongside. Mixed/compound configs degrade gracefully: distillation terms run as before; hard CE is skipped unless the target is layer 0.
- `b189bbd` — **token-only data path.** `token_only_data(loss_fn)` predicate on the model (true iff `ce_token` is in the resolved loss config and `target_layer_ids == [0]`; DFlash2 pinned to distillation). `ArrowDataset(require_hidden_states=False)` emits `{input_ids, loss_mask}` and returns before the hidden-states transfer is consulted; `create_train_val_loaders` skips the noise transform; draft forwards take hidden states as optional kwargs with a clear error if distillation targets are missing. No new CLI surface: the mode is the conjunction `--loss-fn ce_token --target-layer-ids 0`.

**What ran** (5k-conversation slice, single A100, `HF_HUB_OFFLINE=1`):

- Slice: `zcat train.jsonl.gz | head -n 5000` → 11.15M tokens (avg 2,231/conv; ~72% completion tokens).
- `prepare-data`: pretokenized passthrough, 13.5s → 4,926 rows, torch-format arrow, no `hidden_states/`.
- `speculators.train --loss-fn ce_token --target-layer-ids 0 --epochs 1`: healthy through step 365/1083 when killed — loss 6.49 → 2.98 (uniform over the 248k vocab would be 12.4), accept_rate 6e-06 → 0.076, confidence_abs_error 0.57 → 0.40, error_records=0.
- **No-verifier proof:** `--on-missing raise` with a dead vLLM endpoint; the transfer was never consulted (any consult would have crashed the run).
- Unit/GPU smoke: 417 unit tests across models/train suites; 8-check GPU smoke (token-only fwd/bwd, hard loss invariant to verifier tensor values, soft regression, clear error for soft-without-keys, DSpark heads, predicate matrix, dataset item keys, full loader feeding).

**Throughput** (per rank; the logged metric is rank-local, summed across ranks when distributed): ~3.3–4.5k packed tok/s. Step anatomy at 8192 packed tokens: fwd 136ms, bwd 945ms, opt 934ms (Muon ≈ half the step). Anchor cap trains 512×8 = 4096 of 8192 positions (×0.5), loss mask ×~0.72 → ~1.2–1.6k effective supervised tok/s. Full corpus (~4.2B tokens) projects to ~12 days/epoch on one A100, ~1.5 days on 8×A100 DDP.

**Divergences from `dflash-pretraining`:** deliberately narrower seam — no `pretrain_data.py` streaming loader, no `expand_aux_layers.py`; the layer-0 embedding substitution is inline in DFlash's fc path.

**Facts discovered:**

- `lm_head`/`verifier_lm_head` are frozen by design (verifier-tied); only the trunk, fc, and markov/confidence heads train.
- The corpus is temp-1.0 on-policy — fine for pretraining; the CE-distill equivalence check needs a greedily-sampled hidden-states dir (doesn't exist yet).
- The e2e prep used an uncommitted local tweak to `load_raw_dataset` (accept `.jsonl.gz`/`.json.gz` directly) — dataset-prep convenience, reverted after the run; not part of the branch.

**Outstanding:** equivalence check vs CE-distill; fc-width decision for the corpus-scale run.
