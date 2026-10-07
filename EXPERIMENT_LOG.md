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

### 2. 2026-09-29 — corpus-scale throughput: token-only pretraining (4×H200) vs hidden-states distillation (4+4×H200)

**What ran.** Same host (nm-frk-h200-02, 8× H200 141GB), same corpus (full 1,739,710-row regenerated arrow dataset, ~4.26B tokens, ~73% supervised), same draft recipe (DSpark, 5 layers, `block_size 8`, `sample_from_anchor`, Muon 1e-3, seed 42, FSDP × 4 ranks, 2 epochs, `train_data_ratio 0.99`). Three setups:

1. **Distillation** — upstream `main` @ `e3041dd` (this fork's base). 8 GPUs: 4×H200 vLLM hidden-states server (TP=4, `--max-model-len 8200`, 9 aux layers `[4,12,20,28,36,44,52,60,64]` → fc input `[T, 46080]`) + 4×H200 training. Batch 8192/1024, loss `{"ce":0.1,"tv":0.9}`, `on_missing generate`.
2. **Token-only** (this branch @ `1c1b145` = entry 1) — `--loss-fn ce_token --target-layer-ids 0`, fc input `[T, 5120]` from the frozen embedding, no server, no hidden states (`on_missing raise` tripwire; transfer never consulted). Batch 12288/1536.
3. **Token-only, 2× batch probe** — same as (2) at 24576/3072.

**Measured** (rank-0 `profile/*` at steady state; distillation = mean over steps 467–766; 12288 = steps 6–55; 24576 = first ~270 steps):

| | distill 8192/1024 | token-only 12288/1536 | token-only 24576/3072 |
|---|---|---|---|
| GPUs | 8 (4 server + 4 train) | 4 | 4 |
| steps/s | 1.01 | 1.41 | ~0.94 |
| global tok/s | ~30,100 | ~63,100 | ~90,000 |
| step_ms | ~990–1080 | 709 | ~1060 |
| fwd / bwd / opt / fetch ms | 185 / ~215 / ~320 / ~22 | 171 / 272 / 227 / ~4 | ~305 / ~524 / ~228 / ~4.5 |
| unaccounted step time | ~300 ms | ~0 | ~0 |
| packing efficiency | 91% | 89% | ~99% |
| peak train mem (GB / 141) | 87.5 | ~93 | 140.5 |
| µs per rank-token | ~133 | ~65 | ~44 |
| projected 2-epoch wall | ~3.3 d | ~1.6 d | ~25 h |

**Findings:**

- **3× the throughput on half the GPUs** (6× per GPU): ~90k vs ~30k global tok/s. Per-rank-token cost drops 133 → 44 µs.
- **The hs pipeline was the overhead, not the model**: in token-only mode the step components sum exactly to `step_ms` (fwd+bwd+opt+fetch ≈ 1060 at 24576), while distillation carried ~300 ms/step unaccounted (hs collation/staging) plus a fetch path through a live server.
- **fc input shrinks 9×** (`[T, 46080]` → `[T, 5120]`): step time at 24576 (1060 ms) is barely above distillation's at 8192 (~1000 ms) for 3× the tokens.
- **Bigger bins pack tighter**: 91% → 89% → ~99% of `total_seq_len` real tokens/step (the multipack sampler's tail waste amortizes).
- **24576/3072 fits, at the edge**: peak 140.5/143.8 GB on ranks 6/7 (~3 GB margin). The caching allocator logged 3 non-fatal OOM-retry warnings during warmup (24.4 GB transient alloc, recovered after cache flush; no recurrence in ~270 steps). 12288/1536 sits at ~93 GB with wide margin.
- Loss at 12288: 6.59 → 4.30 by step 55 (matches entry 1's slice trajectory); accept_rate 5e-06 → 0.008 by step 55.

**Reproducibility:**

- Distillation: upstream `vllm-project/speculators` `main` @ `e3041dd`; server `launch_vllm.py Qwen/Qwen3.8-27B --target-layer-ids 4 12 20 28 36 44 52 60 --hidden-states-path <dir> -- --port 8400 --tensor-parallel-size 4 --max-model-len 8200` (vllm 0.30.0); training `torchrun --standalone --nproc_per_node 4 -m speculators.train --config qwen38-dspark-regen-fresh-2ep.yaml`.
- Token-only: this branch @ `1c1b145` (+ this commit); training `CUDA_VISIBLE_DEVICES=<4 held GPUs> torchrun --standalone --nproc_per_node 4 -m speculators.train --config qwen38-dspark-pretrain-2ep.yaml`, no server. Both configs live untracked in the checkout's `configs/` (fields as listed above); the 24576 probe is the same file with `total_seq_len`/`max_anchors` bumped.
- Env: torch 2.13.0+cu130, Python 3.12, editable install from the working tree.
- Metric caveat: `profile/tokens_per_s` is rank-0-local (the profile dict is not all-reduced; only rank 0 logs) — global numbers here are 4× rank-0.

**Outstanding:** 24576/3072 long-run stability (3 GB margin on ranks 6/7); warm-start distillation (stage 2) from this run's checkpoint.

### 3. 2026-09-29 — multilingual raw-text corpus staged (FineWeb-Edu + FineWeb-2)

**What/why.** Stage-1 pretraining wants a raw-text, off-policy corpus (per the fork purpose: pretrain on text, then distill on-policy). Corpus decision: **1/2 English + 1/16 × 8 other languages** by token count — exactly balanced halves. English side: `HuggingFaceFW/fineweb-edu` `sample/100BT`. Other side: 8 subsets of `HuggingFaceFW/fineweb-2` (train splits only; each subset also ships a `test/` split, excluded by construction).

**Language selection** (rationale: top-volume tier — every pick has ≥98 GB total so a 1/16 share is obtainable at any sane budget — plus script/family diversity and Qwen-tokenizer fit):

| Language | Subset | FW-2 total | Chosen (seeded random) |
|---|---|---|---|
| Mandarin Chinese | `cmn_Hani` | 1622 GB / 370 shards | 13 shards, 45.5 GB |
| Japanese | `jpn_Jpan` | 717 GB / 175 shards | 10 shards, 48.4 GB |
| Korean | `kor_Hang` | 106 GB / 25 shards | 10 shards, 45.3 GB |
| Russian | `rus_Cyrl` | 1988 GB / 440 shards | 10 shards, 48.4 GB |
| Standard Arabic | `arb_Arab` | 106 GB / 25 shards | 10 shards, 45.3 GB |
| Spanish | `spa_Latn` | 638 GB / 146 shards | 10 shards, 47.5 GB |
| French | `fra_Latn` | 540 GB / 135 shards | 12 shards, 45.1 GB |
| German | `deu_Latn` | 772 GB / 181 shards | 11 shards, 49.6 GB |

Hindi was considered and excluded: 32 GB total (~5–8B tokens) is insufficient for a 1/16 share against an ~80B-Qwen-token English side. CJK picks double as tokenizer-efficiency coverage (Han/Kana/Hangul); Cyrillic and Arabic cover the remaining scripts.

**Acquisition.** Everything landed on `/data` (56 TB LV, 51 TB free) under `/data/playground/reed-meyerson/` — `/home` (842 GB free) cannot hold parquet + datasets arrow cache + tokenized output together. `/data` root is root-owned; `/data/playground` (1777) is the shared scratch. Downloads via `huggingface_hub` snapshot with `hf_transfer` (Rust chunked downloads) + an authenticated token: ~1 GB/s aggregate, full edu subset in ~5 min. Inventory: `fineweb-edu/` 140 parquet, 286.4 GB (= 267 GiB); `fineweb-2/` 86 parquet, 375.1 GB (= 349 GiB); 616 GiB total.

**Sampling method.** FineWeb-2 shards are CC-MAIN-crawl-ordered (~4.8 GB each, named `NNN_NNNNN.parquet`), so **first-N would be a temporally biased slice** (earliest crawls only). Instead: seeded random shard subset per language (seed 42, sorted-then-shuffled for determinism), targeting 45 GB/language vs the ~10B Qwen-token 1/16 requirement — margin covers per-script token/byte variance. Chosen files recorded in `fineweb-2/manifest.json` (repo-external; the pull script `pull_finetweb2.py` sits next to it on /data). Final mix ratios will be enforced **exactly at prep time** by row-level subsampling after tokenization (per-source token counts are only known then); the shard pull just guarantees sufficient volume.

**Prep-path design (planned; code follows in the next commit).** Minimal fork diff, all in `data_generation/preprocessing.py` (~45 lines, no CLI changes, no vLLM server):

1. `load_raw_dataset`: accept local `.parquet` file / directory of parquets (raw `text` column).
2. New `_preprocess_raw_text` batch fn: local tokenizer (no chat template, no render endpoint), `add_special_tokens=False` + append `eos_token_id` per doc, `loss_mask = ones`, reuse `_append_row` for `--seq-length` truncation and `--minimum-valid-tokens` filtering.
3. Dispatch in `_preprocess_batch` on a `text` column **before** the `render_endpoint is required` raise.
4. Tokenizer plumbed into the map `fn_kwargs` via `get_tokenizer(processor)`.

Everything else stays untouched — anchor/boundary logic, packer, `ce_token` + `target_layer_ids [0]` path — so the raw-text run stays comparable to the chat-corpus run of entry 2. Token-exact mixing, val holdout, and downscaling live in a local driver script outside the fork. `HF_DATASETS_CACHE` must point at /data for the runs (datasets materializes a full arrow copy of the parquet, ~330 GB for the text side).

**Facts discovered:**

- FineWeb-Edu's "100BT" is GPT-2-tokenizer counted; Qwen's larger vocab yields ~20–25% fewer tokens on the same text → English side ≈ 75–80B Qwen tokens, each 1/16 share ≈ 10B.
- Full mix ≈ 160B Qwen tokens ≈ 20 days at the entry-2 rate of ~90k tok/s; the driver's token budgets make a smaller first pass (e.g. 40B: 20B EN + 2.5B/lang) a one-line change.
- fineweb-2 hub layout is `data/<lang_Script>/train/*.parquet` + `test/*.parquet` (the tree API needs the split subdirectory; languages are ISO 639-3 + script codes, e.g. `cmn_Hani` not `zho`).

**Outstanding:** write the preprocessing diff + mixing driver; tokenize and stage the mixed arrow dataset; 24576/3072 long-run stability; GPU hold renewal before ~02:40 UTC.

### 4. 2026-09-29 — `scheduler_type: constant` (flat LR after warmup)

**What/why.** The planned FineWeb raw-text pretraining run (entry 3) wants a **constant learning rate with a short warmup**: schedule-free means the run can be stopped at any token budget without a truncated-decay confound. The trainer's existing `scheduler_type: "none"` builds *no scheduler at all* — base LR from step 0 and `scheduler_warmup_steps` silently ignored — so "constant + warmup" was not expressible. Added a `"constant"` type that wraps transformers' `get_constant_schedule_with_warmup` (one scheduler per optimizer, so the Muon and AdamW groups warm up independently, same as linear/cosine).

**Change.** `Literal["linear", "cosine", "constant", "none"]` in both `train/config/schema.py` (`SchedulerArgs`) and `trainer.py` (`TrainerConfig`), plus the `make_scheduler` branch. No CLI changes needed — the argparse layer is generated from the schema. `scheduler_total_steps` is unused for `constant` (warmup resolution unchanged: explicit steps > ratio > 1% default).

**Planned run values.** `scheduler_type: constant`, `scheduler_warmup_steps: 100` ≈ 10M tokens at ~98K global tokens/step (24576 × 4 ranks, ~99% packing).

**Verification.** LR trajectory on a dummy optimizer: 0 → 0.5×peak at step 50 → peak at step 100 → flat through step 249. `TrainConfig` round-trips `scheduler.scheduler_type: constant` through `flatten()`; `--scheduler-type` choices now include it.

**Outstanding:** none for this piece; part of the raw-text pretraining setup (entries 3, 5, 6).

### 5. 2026-09-29 — keep-N checkpoint rotation for sub-epoch checkpointing

**What/why.** The raw-text run checkpoints hourly via `checkpoint_freq < 1` (see entry 6 planning in entry 3's setup), over a multi-day single epoch. As implemented, sub-epoch saves **overwrite `path/<epoch>/` in place**: only the latest snapshot exists, and a crash mid-write corrupts the only copy. Added rotation: before an overwrite, shift `path/<epoch>/` → `<epoch>.prev1/` → `<epoch>.prev2/` → …, keeping the `checkpoint_keep` most recent snapshots (fresh save + N−1 prevs) and deleting older ones.

**Change.** `BaseCheckpointer.rotate_previous()` + `keep` ctor arg, called at the top of both `save_checkpoint` implementations (rank-0 contexts only). New `--checkpoint-keep` (default 3) plumbed schema → `TrainerConfig` → checkpointer. The dotted `.prevK` names are deliberately not int-parseable: auto-resume (`_get_previous_epoch`) only ever finds the fresh `<epoch>/` dir, so resume semantics are unchanged; prevs are manual-rename recovery points (each carries `training_state.json` with its `global_step`). Whole-epoch checkpointing is unaffected — each epoch writes a fresh dir, so rotation no-ops. End-of-epoch saves after mid-epoch ones rotate the last mid-epoch snapshot to `.prev1` (desirable: the final state lands in `<epoch>/`, the last hourly snapshot is retained as a prev).

**Verification.** Simulated 6 successive mid-epoch saves (rotate-then-write order): final state exactly `0/` (step 6), `0.prev1/` (5), `0.prev2/` (4); auto-resume scan returns only epoch 0; `keep=1` deletes the current dir; rotation on a fresh epoch dir is a no-op; `keep=0` raises. Schema round-trip + CLI flag present.

**Caveat (accepted).** Saves are still not atomic — a kill during a save leaves `<epoch>/` corrupt, but recovery is now `rm -rf <epoch>/ && mv <epoch>.prev1 <epoch>` instead of data loss. Hourly cadence ≈ 3,300 steps at ~98K global tok/s (~324M tokens/h).

**Outstanding:** none for this piece; part of the raw-text pretraining setup (entries 3, 4, 6).

### 6. 2026-09-29 — mid-epoch validation at checkpoint boundaries

**What/why.** `run_training` validates only at **epoch end** — for the raw-text run (single ~160B-token epoch, stopped prematurely after days), that means validation would never fire, and the stop decision would have to ride on train-loss alone. Added: at each mid-epoch checkpoint boundary (the existing `checkpoint_freq < 1` save site in `train_epoch`), run `val_epoch` after the checkpoint is written, save `val_metrics.json` into the (rotating, entry 5) epoch dir, and track best val loss in the log. Metrics are also logged to the metric backend keyed by `global_step` (same path as epoch-end val), so trackio gets an hourly val curve.

**Change.** ~20 lines in `trainer.py` `train_epoch`: save → `val_epoch` → `save_val_metrics` (rank-0, barrier) → best-loss log line → **`self.model.train()`** — the critical fix, since `val_epoch` sets eval mode and the mid-epoch loop would otherwise continue in eval mode (`train_epoch`'s own `model.train()` only fires on epoch entry). No-ops when `val_loader is None` or `save_best=True` (mid-epoch checkpointing is disabled under `save_best`). Ordering: checkpoint **before** val, so a death during the ~3-min val pass still leaves the checkpoint on disk.

**Verification.** Stubbed-loop test of the boundary block: fires only at `local_step % step_interval == 0` (and only when the `MIN_STEP_PCT` tail guard passes), save→val→metrics→`train()` ordering, `best_val_loss` tracking, `val_metrics.json` content. Full module + CLI import clean.

**Run parameters this enables.** Hourly val at ~324M-token boundaries; val slice sized ~50M tokens (~0.03% of the mix, ~3 min/pass) — the stop decision gets a val-loss signal roughly every 20 train-side loss-logged... every hour, aligned with each kept checkpoint.

**Caveats (accepted).** `best_checkpoint` symlink is not created in this path (rotation would make a standing symlink lie about "best"); on resume `best_val_loss` restarts at ∞ (informational only). Val cost is serial with training (~5% overhead at 3 min/hour).

**Outstanding:** none for this piece; part of the raw-text pretraining setup (entries 3–5).

### 7. 2026-09-29 — token-denominated scheduler horizon (cool-down path)

**What/why.** After the raw-text pretrain run is stopped prematurely (entry 3 plan), the next step is a **cool-down**: anneal the checkpoint on the on-policy chat corpus (the stage-2 distillation data, `~/qwen38-regen-training/data`), still token-only, with a **linear descending LR over a configurable number of tokens**. The trainer only understands steps; expressing the horizon in tokens is the natural unit for the budget ("anneal over N tokens") and matches how the pretrain stop decision is made. Added `scheduler_total_tokens`: resolved at launch to optimizer steps and used as both the schedule horizon and (by default) the run's `max_steps`, so the LR lands at ~0 exactly when the run stops.

**Change.** `SchedulerArgs.scheduler_total_tokens` (`--scheduler-total-tokens`, schema → argparse auto-generation) + a resolution block in `cli.py` `main()` (reads `cfg.scheduler`/`cfg.data`/`cfg.trainer` directly, per the phase-1 adapter note): `total_steps = ceil(tokens / (total_seq_len × world_size))` — 98,304 global tok/step at 24576×4 ranks, ~99% packing efficiency makes the realized budget accurate to ~1%. Precedence: an explicit `scheduler_total_steps` or a differing `max_steps` logs a warning and the token-derived steps win for the schedule; `max_steps` defaults to the same horizon. `_resolve_scheduler_steps`/`make_scheduler` are untouched (they consume the resolved `TrainerConfig.scheduler_total_steps`). Works with any scheduler type — linear for the cool-down; for constant it just bounds the run length.

**Verification.** Schema round-trip (`flatten()` carries the field), `ge=1` enforced, `--scheduler-total-tokens` in `--help` (after fixing an argparse `%`-format crash in the field description — help strings go through `help % params`). Resolution math: 2B tokens → 20,346 steps (~6.0 h at 0.94 steps/s). Linear LR trajectory on a dummy optimizer (warmup 0, total 20346): peak at step 0 → 4.9e-8 at step 20345 → exactly 0 at 20346.

**Cool-down config.** `configs/qwen38-dspark-cooldown.yaml` (untracked by convention — machine-specific paths; documented here): warm start via `draft.from_pretrained` (pretrain checkpoint; geometry + `target_layer_ids [0]` come from the checkpoint config; verifier-owned embed/lm_head reconstructed from `Qwen/Qwen3.8-27B`), chat corpus, `loss_fn: ce_token`, `scheduler: linear` + `scheduler_total_tokens: 2e9` + `warmup 0`, `epochs: 1` (upper bound only — the chat corpus is ~42.9K steps/epoch), `checkpoint_freq: 0.079` (~hourly), `checkpoint_keep: 3`, fresh `save_path` (auto-resume inert on empty; a stopped cool-down resumes from its own checkpoints). Fresh Muon moments at peak LR is a known transient — the config notes bumping warmup to ~100 if unstable.

**Outstanding:** run it when the pretrain run is stopped; GPU launch awaits reservation + user go.

### 8. 2026-09-29 — zero-expansion converter for the stage-1 → stage-2 fc seam

**What/why.** Stage-1 pretrains token-only: the draft's `fc` consumes only the verifier's layer-0 hidden state (the unscaled embedding), so `fc.weight` is `[hidden, hidden]`. Stage-2 distillation feeds the concatenation of several verifier layers (`aux_hidden_state_layer_ids`), so `fc` must be `[hidden, n_ids * hidden]` — a shape change that blocks warm-starting stage 2 from the pretrained draft. Added `scripts/expand_target_layers.py`: zero-expands the fc input dim, placing each old column block at the position of its layer id in the new id list and zeros elsewhere. Because the forward is `fc(concat(h_id))` and the zero blocks contribute nothing, **the converted model behaves identically to the stage-1 checkpoint** whenever the input places the old layers' hidden states in their blocks — stage-2 finetuning then grows weights into the zeroed columns from a warm trunk.

**Change.** Standalone CPU script (no GPU, no model class needed): reads `config.json` (`aux_hidden_state_layer_ids`, nested `transformer_layer_config.hidden_size`) + `model.safetensors` from a trainer-written checkpoint, validates the fc shape against the config, expands, and writes a fresh `--from-pretrained`-loadable dir (`model.safetensors` + rewritten `config.json` + `config.py` if present; optimizer/training state deliberately not copied — the output is a fresh model, not a resumable run). New ids must be a duplicate-free superset of the old ids (dropping a trained layer would change behavior → hard error). Position mapping is general (`new_ids.index(old_id)`, not "0 is first"). Single-file safetensors only — the ~1.5B-param bf16 draft is ~3GB, under the 5GB shard threshold; sharded checkpoints get an explicit error. Usage: `python scripts/expand_target_layers.py CKPT --new-target-layer-ids 0 4 12 20 28 36 44 52 60 --output OUT` (only `fc.weight` changes shape; trunk, norms, lm_head, Markov/confidence heads are hidden-size-based and copy unchanged).

**Verification** (tiny synthetic geometry, CPU, float32: fake 8-layer Qwen3 verifier + 2-layer dspark draft, ids `[0]` → `[0, 3, 6]`): (1) weight placement — old block at index 0, zeros elsewhere; (2) general mapping with 0 not first (`[3, 0, 6]`); (3) fc-level: `B.fc([h0; h3; h6]) == A.fc(h0)`; (4) **backbone-forward equivalence**: A in token-only mode (`hard_targets=True`, hs=None → `fc(embed(input_ids))`) vs B in distillation mode (`hard_targets=False`, hs=`[embed(input_ids); junk]`) produce identical trunk hidden states and draft logits (same anchors via re-seeding — `select_anchors` draws `randperm`); (5) error cases (dropped layer, duplicate ids). The warm-start `from_pretrained` load path (checkpoint dir + verifier name, `_attn_implementation` re-applied, mirroring cli.py) is exercised by the same test — the cooldown config (entry 7) uses it too.

**Operational notes for stage 2.** The new id list must include 0 (where the pretrained weights live) — e.g. `[0, 4, 12, 20, 28, 36, 44, 52, 60]` (9 ids, fc in = 46080 at hidden 5120) — and the verifier hs server (`launch_vllm.py --target-layer-ids …`, which appends the final layer 64 itself) and the training config's `draft.target_layer_ids` must use the **same list in the same order** (concat order). The existing chat-corpus hs cache was captured at the old width (8 aux layers) — stage 2 with the converted model needs a **fresh `hidden_states_path`** (or cache wipe) so hs are re-captured at the new width; stale-width hs would fail at the fc matmul.

**Outstanding:** run it on the real pretrain checkpoint when stage 1 is stopped; then stage-2 distillation (warm start via `--from-pretrained` on the converted dir) with a fresh hs server + cache.

## Entry 9: parallelize prepare-data's save_to_disk

**Problem.** `speculators prepare-data`'s final `dataset.save_to_disk(output)`
ran single-process on a dataset that carries a shuffle indices mapping (the
preprocessing pipeline shuffles post-map). The save gathers rows through that
mapping one at a time: observed ~3-12K rows/s (varies with indices locality
and page cache) on the 97.27M-row FineWeb-Edu EN corpus — a projected 2-11h
for a ~1.1TB / 2393-shard write, with eight more FineWeb-2 language saves
behind it. The filesystem itself was exonerated (dd direct-write 8.2 GB/s).

**Change.** `prepare_data.py` now passes `num_proc` to `save_to_disk`, reusing
the same worker-count resolution as the map (`--num-preprocessing-workers`
when given, else `default_preprocessing_workers()`). datasets ≥2.14 shards the
save across workers, each writing whole output shards; the shuffled-order
gather now runs N-way parallel.

**Verification.** Tiny end-to-end run (single fineweb-edu parquet,
`--max-samples 200000`, 16 workers): 200K rows -> 16 shards in 13s at
~50K rows/s (~30x the observed single-process rate; EN projects to ~5 min
at 120 workers). `load_from_disk` round-trip: 200,000 rows, columns
`[input_ids, loss_mask, seq_len]`, 207.3M tokens, per-row
`len(input_ids) == seq_len` spot checks pass.

**Operational note.** The restart replays the map from the datasets
fingerprint cache (the ~1.1TB map output lives in the HF datasets cache on
/data), so a killed single-process save costs only the re-save. Restart
hygiene: `prepare-data` skips an output dir that contains any `*.arrow`, so a
partial save must be wiped before relaunch (the driver does not pass
`--overwrite`).

## Entry 10: mixed pretraining corpus staged — custom parallel writer + val-tail microcosm

**Problem.** With all nine sources prepared (entry 9's parallel save finished
the job: EN 97.27M rows / 99.55B tokens in ~7 min, the eight FineWeb-2
languages 209.77M docs total in ~2h), the mix phase — select rows to the
token-exact budget (1/2 EN + 1/16 × 8, 199.11B tokens, 212,281,873 rows) and
write ONE combined dataset dir — stalled in `datasets`' `save_to_disk`
machinery: py-spy showed the parent grinding serially inside the
`kwargs_per_job` generator (`shard()` → `select` → `Dataset.__init__` →
`update_metadata_with_features` → `ConcatenationTable` rebuild →
`table.to_batches()`), ~30-60s+ per shard × 4,653 shards = days. The forked
write workers were healthy; the parent-side per-shard metadata rebuild over
the ~6,800-block combined table is simply O(shards × blocks). `flatten_indices`
and `num_shards` variants share the same parent-side path.

**Key enabler.** The trainer's `MultipackDistributedBatchSamplerV2.__iter__`
re-permutes ALL rows every epoch (`rng.permutation`, seed + epoch). On-disk
row order is therefore irrelevant to training — the mix can be written in
sequential coarse-interleaved order with no random gather and no datasets
indices machinery at all.

**Design (driver `~/fineweb-pretrain-data/prep_and_mix.py`, outside the fork
per convention; config+script paths untracked).**

- Per source: seeded first-crossing row selection to the token budget, then
  **sorted** row ids — every source read becomes a sequential scan (the
  selection itself already fixes WHICH rows; sorting only fixes the order).
- Interleave: selected rows chunked into ~45K-row (~500MB) single-source
  files, round-robin across sources (en, cmn, jpn, kor, rus, arb, spa, fra,
  deu, repeat; depleted sources drop out).
- Write: plain stdlib `multiprocessing.Pool` (120 workers) of
  `datasets.arrow_writer.ArrowWriter` shard writers. Each worker takes its
  file's sorted rows **per memory-mapped block** (`t.table.take(m - a)`, then
  `pa.concat_tables`): a take on the combined >2GB source table overflows
  pyarrow's int32 list offsets ("offset overflow while concatenating
  arrays"), while per-block takes stay under the limit, and sorted ids make
  the per-block parts concatenate directly in order. `state.json` +
  `dataset_info.json` are written by hand in the exact layout
  `save_to_disk` produces (`_data_files` list + torch row format;
  `load_from_disk` only reads `_data_files` — the `dataset_info` splits
  block, e.g. the stale 984-entry `shard_lengths` in the EN dir, is
  informational).
- **Val tail:** the trainer splits val as the contiguous tail
  `data[int(len × train_data_ratio):]` — with round-robin interleaving, the
  file-order tail is ALL-EN (EN has 2,162 chunks vs ~300-370 per other
  language), so a naive tail slice would be an English-only val. Instead the
  FINAL file is a proportional microcosm: per source, the last rows of its
  selection summing to its token share of the 50M-token val budget
  (53,432 rows / 50,363,514 tokens), one multi-source file.
  `train_data_ratio = (rows − val_rows)/rows = 0.9997482969259462` lands the
  split exactly on that file boundary (verified on the staged dir).

**Verification.** (a) Mechanism dry-run on real data: two sources, 4
interleaved files via the per-block take + ArrowWriter path — row-level
equality against direct takes, torch format carried, `load_from_disk`
round-trip clean. (b) 1B- and 2B-token end-to-end smoke runs: exact
rows/tokens vs the report; per-file average `seq_len` matches each source's
signature in round-robin order; final-file sliding-window bands show the
multilingual microcosm; trainer-split lands exactly on the val boundary.
(c) Full run: 212,281,873 rows / 199,106,618,466 tokens / 4,723 files /
2.39TB in 19 min (peak 4.1 files/s ≈ 2 GB/s), deterministic across reruns
(identical totals). `mix_report.json` at `fineweb-prepared/` carries
per-source `{fraction, token_budget, tokens_used, rows}` + totals + val
tail + `train_data_ratio`.

**Corpus (staged, final).** EN takes ALL 99.55B tokens (budget-binding at
2×EN); each FineWeb-2 language 12.444B. Train side 199.056B tokens
(~2.03M steps at 98,304 global tokens/step, ~99% packing), val 50.36M
(~512 fwd-only steps, ~3 min per hourly boundary).

**Run config (untracked, `configs/qwen38-dspark-pretrain-fineweb.yaml`) +
launcher (`~/launch-pretrain.sh`, canhazgpu-run integrated with a
MANUAL_HOLD escape hatch; `~/launch-smoke.sh` for the 50-step smoke).**
Constant LR 1e-3 + warmup 100 steps (~10M tokens); epochs 1 as an upper
bound only (~25 days at 0.94 steps/s — premature stop planned, then entry
7's cooldown); `checkpoint_freq: 0.001655` (~3,383 steps ≈ hourly),
`checkpoint_keep: 3`; validation rides each checkpoint boundary (entry 6).
Batch geometry 24576/3072 (1:8), FSDP shard, Muon 1e-3, trackio logging;
artifacts under `/data/playground/reed-meyerson/fineweb-pretrain-run/`.

**Outstanding:** GPU smoke test (50 steps, saves at 20/40, boundary val)
then the launch — both via `canhazgpu run`, on user go.

## Entry 11: columnar seq_len read in `_compute_approx_lengths`

**Problem.** The fineweb smoke launch (50-step preflight for entry 10's mix)
sat ~25 min at 100% CPU with idle GPUs before training ever started. py-spy:
`ArrowDataset._compute_approx_lengths` — `list(ds.with_format(None)["seq_len"])`
— grinding inside the datasets formatter (`extract_row` ← `format_row` ←
`format_table` ← `Dataset.__iter__`). In datasets 5.x, `ds["col"]` on an
indices-mapped dataset returns a lazy column view whose iteration walks every
row through the formatter: ~100-200K rows/s. Unnoticed on the 1.74M-row chat
corpus (~15s), it is ~35 min on the 212M-row mixed corpus — paid by every
launch, train and val alike.

**Change.** `ArrowDataset._compute_approx_lengths` (train/data.py) now reads
the column columnar — `np.asarray(self.data.data.column("seq_len"))` —
returning an ndarray (the only consumer,
`MultipackDistributedBatchSamplerV2.__init__`, immediately does
`np.array(lengths)`, so behavior is identical). Measured on the staged mix:
0.7s for the full 212M-row column (~3000x). In datasets 5.x a
contiguous-range `select` materializes directly into `.data` (no indices
mapping), so `.data.column` is already exactly this split's rows; a
non-contiguous select leaves a one-column-table indices mapping, which is
gathered explicitly (`indices.column(0).to_numpy(zero_copy_only=False)` —
`np.asarray(indices)` is WRONG, it nests to shape (1, n)).

**Regression caught by the smoke run (first attempt).** The initial version
of this change sliced the full pre-select column by
`[start_file_idx : start_file_idx + len(data)]`, on the belief that
`select(range(start, stop))` keeps `.data` untouched and maps rows via
indices. Wrong for datasets 5.x: select MATERIALIZES. Consequence: the train
split (start=0) was coincidentally correct, but the val split's `.data` was
already only 53,432 rows, so slicing at `[212,228,441 : ...]` yielded ZERO
lengths → `len(val_loader) == 0` → every val pass ran 0 batches and wrote
`{}` to val_metrics.json while training happily proceeded. The 50-step smoke
run surfaced it immediately (train metrics fine, val absent) — exactly what
a smoke run is for. An equivalence check against the old path had passed
before launch, but it exercised the pre-select dataset's `.data`, not the
post-select `.data` the class actually reads — lesson recorded: verify the
exact class codepath, not a hand-replicated expression.

**Verification (post-fix).** Exact class expression on the real mix:
train 212,228,441 lengths / 199,056,254,952 tokens and val 53,432 /
50,363,514 tokens (exactly the staged val-tail file, entry 10); a 3000-row
window per split is element-equal to the old row-by-row path; the
non-contiguous-select gather is element-equal to the old path; val sampler
len() = 523 rank-0 batches.

**Also measured (startup budget for the real run).** The multipack
`_assign_to_packed_batches` loop over the full 212M rows extrapolates to
~12 min per rank (13.8s on a 4M-row subsample; ranks pack independently in
parallel, cached per epoch) — now the dominant one-time startup cost after
the fix.

**Outstanding:** relaunch the smoke test on the corrected path.

## Entry 12: no CUDA contexts in dataloader workers for file/token-only paths

**Problem.** Smoke run 2 (val fix from entry 11 in, everything else identical)
trained 20 steps fine, saved the step-20 checkpoint, then died the moment the
boundary validation started: rank1 and rank2 both crashed with
`torch.AcceleratorError: CUDA error: out of memory` raised from
`torch.accelerator.set_device_index(local_rank)` inside `_worker_init_fn`
(DataLoader worker process 6, val loader spawn). With 141GB H200s at ~23GB
training usage this is not device-memory exhaustion; it is a driver-level
allocation failure at context creation while a fresh 12-worker val pool
spawns concurrently with the persistent train-worker pool, the main process
pinning the first val batches, and workers fetching. (A bare torch CUDA
context on this box measures 616 MiB — 24 workers/rank ≈ 15GB — so capacity
was never the issue; kernel log clean, /dev/shm and fds fine, no cgroup
limits from canhazgpu.) Smoke run 1 never hit it only because its val pass
was empty (entry 11's regression): zero val batches meant workers spawned
but never fetched/pinned.

**Root cause.** `_worker_init_fn` binds each worker's CUDA device
unconditionally — added upstream in #1168 ("Spawn mooncake clients on
dataloader's associated rank") for hidden-states backends whose workers
touch CUDA in-process (mooncake's transfer engine allocates its local
segment on the rank's device). For file-backed/token-only paths the workers
are pure CPU: arrow reads → CPU tensors → collate on CPU; pinning happens in
the main process and H2D in the trainer. The device binding there is pure
waste — 24 dead 616 MiB contexts per GPU — and, as observed, a crash vector
during concurrent pool spawn.

**Change.** `create_train_val_loaders` computes
`worker_bind_device = isinstance(transfer, MooncakeTransfer)` and passes it
through `_setup_dataloader` to a `partial(_worker_init_fn, bind_device=...)`
worker init (picklable under the spawn context). Mooncake runs keep upstream
behavior exactly; file/token-only workers never touch CUDA at all
(`torch.accelerator.is_available()` is short-circuited behind `bind_device`,
so workers do not even query the driver).

**Verification.** `_worker_init_fn` semantics: default `bind_device=True`
reproduces the upstream behavior (device set to LOCAL_RANK, matching
upstream's unit tests); `bind_device=False` never calls the accelerator API.
The bound partial round-trips through pickle (spawn requirement). Modules
import clean. Upstream `test_dataloader.py` expectations (bind by default)
are unchanged.

**Outstanding:** smoke run 3 with both fixes (entry 11's lengths + this) —
expect the step-20 boundary: checkpoint save → val pass over 523 batches →
resume training to step 40 → save/val → finish at 50.

### 13. 2026-09-30 — real run launched: stage-1 token-only pretraining on 8×H200

**What's running.** The staged FineWeb mix (entry 10) on all 8 H200s under a
24h manual `canhazgpu` hold (renewable — multi-day run), launched from tmux
via `MANUAL_HOLD=1 CUDA_GPUS=0,...,7 ~/launch-pretrain.sh` →
`torchrun --standalone --nproc_per_node 8`, stdout tee'd to
`fineweb-pretrain-run/logs/train.log` (rotated per launch). Config is the
staged pretrain config with two 8-rank adaptations: warmup **51 steps**
(10.03M tokens at 196,608 global tok/step — the ~10M-token warmup budget
preserved in tokens, since the global batch doubles) and
`checkpoint_freq 0.0064` (see the cadence note below).

**Throughput (steady state, steps ~100–400):**

| | distill 4+4 (entry 2) | token-only 4×H200 (smoke) | token-only 8×H200 (this) |
|---|---|---|---|
| global tok/s | ~30,100 | ~90,000 | **~190–195K** |
| per-GPU tok/s | ~3,760 | ~22,500 | ~24,200 |
| step_ms | ~990–1080 | ~1060 | ~1010–1060 |
| fwd/bwd/opt/fetch ms | 185/215/320/22 | 305/524/228/4.5 | 308/533/193/7 |

- **2.08× on 2× GPUs** vs the 4-rank smoke (~94K → ~195K global tok/s,
  slightly superlinear): per-rank work per step is unchanged (same 24,576
  tokens/rank), so the scaling arrives as 2× tokens per step at the same
  ~1.0 s step time, with FSDP 8-way halving the live shard footprint.
- **6.5× the baseline distillation setup** (entry 2's 4 vLLM server + 4
  training) on the same 8 GPUs: ~195K vs ~30.1K global tok/s. The
  verifier-free design is why stage 1 can chew 199B tokens in ~12 days
  instead of ~77: ~16.9B tokens/day; full epoch ≈ 11.8 days (premature stop
  planned).
- Step components sum to step_ms (no unaccounted time — the hs-pipeline
  overhead of the distill baseline stays absent at 8 ranks).

**Early trajectory** (first ~400 steps ≈ 79M tokens): train loss 6.55 → 2.87,
accept_rate → 0.063, eal 1.26, position accs ~0.15. (Smoke's step-50 numbers
for contrast: loss 3.96, accept 0.013 — this run has seen ~40× more tokens.)

**Operational notes:**

- Startup at 8 ranks: ~3 min model load, 0.7 s lengths (entry 11), ~16 min
  multipack (8 ranks contend for memory bandwidth; 12.5 min at 4), 96
  dataloader workers spawning, ~2 min compile → first step ~22 min after
  launch (11:56:17 → ~12:18).
- **Checkpoint cadence landed at ~108 min + val, not ~60**: the 8-rank
  `checkpoint_freq` was computed under the assumption that 2× throughput
  means 2× steps/s — it actually means 2× tokens per step at unchanged
  ~1.0 s/step. `checkpoint_freq` is fixed at startup; decision was to leave
  it running (~1.8 h boundary cadence: 6,480 steps ≈ 108 min train + ~30 s
  save + ~2 min val over ~256 fwd batches/rank) rather than pay a ~25 min
  restart. First boundary: step 6,480.
- Memory: 108–132 GB/GPU steady, asymmetric by rank (0–3 high) — consistent
  with entry 2's 24576/3072 allocator growth on packed-shape variety
  (mostly cache; live state is small: frozen embed, 8-way shards, Muon/AdamW
  states on ~0.46B trainable params). No allocator OOM-retry warnings so
  far. Val at boundaries reuses the cache (observed at 4 ranks, smoke run 3).
- Entry 12 fix in production: 96 dataloader workers with **zero** CUDA
  contexts (would have been ~7.4 GB/GPU of dead contexts and the val-spawn
  crash vector at 8 ranks too).
- The 24h reservation expires ~11:55 next day; renew with
  `canhazgpu reserve` before expiry for the multi-day run.

**Reproducibility:** branch @ this commit (untracked config
`configs/qwen38-dspark-pretrain-fineweb.yaml`: 24576/3072, Muon 1e-3,
constant LR + 51-step warmup, `checkpoint_freq 0.0064`, keep 3, epochs 1,
FSDP ×8, `train_data_ratio 0.9997482969259462`); dataset = entry 10's mix.
Env: torch 2.13.0+cu130, Python 3.12, CUDA 13.0, 8× H200 141GB.

**Outstanding:** first checkpoint+val boundary at step 6,480 (~108 min in)
— verify rotation, non-empty val metrics, and val-loss trend for the
premature-stop decision; reservation renewal.

### 14. 2026-10-01 — `trainer.skip_steps`: data-stream continuation; pretrain stopped at V16, FineWeb linear cool-down launched

**Decision.** The 40h constant-LR pretrain (entry 13) was deliberately
stopped ~32.6h in (step 108,004, ~21.2B tokens) per user call, and the
annealing experiment pulled forward ONTO THE SAME WEB STREAM: a linear
cool-down 1e-3 → ~0 over 2B tokens (10,173 steps at 8 ranks / 196,608
tok/step) warm-started from **V16** (step 104,768, 20.60B tokens, val loss
2.1582 / eal 1.6181 / p0 0.3564 — the 16th consecutive new-best boundary).
This is the clean ablation of entry 7's chat-corpus cool-down: pure
on-distribution LR annealing, no distribution-transfer confound. The
pretrain remains resumable with ZERO repeated data (recipe below).

**The stop.** Ctrl-C via `tmux send-keys` killed the torchrun process group
WITHOUT the graceful-shutdown `interrupted/` save (workers died on group
SIGINT before `TrainingInterruptedError` could fire; no partial writes —
the last boundary save to `0/` was V16 at 18:29–18:31). The ~3.2K steps
trained past V16 exist only in discarded (never-written) state, so the
retained trajectory is exactly V16-weights + data-from-batch-104,769.
Operational lesson: deliberate stops should rely on boundary checkpoints
(or signal the driver PID alone), not tmux-forwarded group SIGINT.

**The feature — `trainer.skip_steps` (this commit).** On a fresh run (no
checkpoint consumed), fast-skip the first N train batches of the run's
first epoch before step 0, reusing the mid-epoch resume fast-skip: the
sampler's pre-generated batch list is sliced `_generate_batches(epoch)[N:]`
and re-cached — no data is loaded for skipped batches, and `local_step`
bookkeeping continues from the true epoch position (mid-run checkpoints
record true positions, so the skipping run is itself mid-epoch-resumable).
Run-local counting (global_step, scheduler horizon, max_steps) starts at 0.
Setting it alongside a resume logs a warning and is ignored; skip ≥ epoch
length raises. Why the stream is EXACT: the epoch-0 batch sequence is a
pure function of (dataset+split, DP size, batch_max_length, epoch) —
permutation `np.default_rng(0 + epoch)` over valid_indices (the sampler
seed is not plumbed from train seed; it is the default 0), split is the
deterministic prefix cut `int(len × train_data_ratio)`, LPT packing is
RNG-free. With identical data_path / ratio / 24576 / 8 DP ranks /
max_batches=None, `batches[104768:]` is bit-identical to what the pretrain
would have trained next: the cool-down's first batch is the pretrain's
batch 104,769.

**The cool-down run.** Untracked config
`configs/qwen38-dspark-cooldown-fineweb.yaml` + `~/launch-cooldown-fineweb.sh`
(manual-hold pattern, 8 ranks); fresh save_path
`/data/playground/reed-meyerson/fineweb-cooldown-run/` (pretrain dir
untouched, per the keep-it-resumable requirement); warm start = reflink
snapshot of V16 into `warm-start/` (decouples from pretrain checkpoint
rotation); fresh Muon at peak LR, warmup 0; linear descent over
scheduler_total_tokens 2e9; checkpoint_freq 0.0024 (~2,456 steps ≈ 42 min;
boundaries at cooldown steps ~840 / 3,296 / 5,752 / 8,208 + final);
same val split → series directly comparable to V1–V16. Smoke run
(8 ranks, skip 200, max_steps 3, own save_path/run-name) validated the full
path — and caught two things: (1) explicit draft-definition keys (e.g.
`num_layers`) are REJECTED alongside `from_pretrained` — a latent bug in
entry 7's never-launched chat-cooldown config, fixed in both configs; (2) a
fresh-optimizer transient on the confidence head: after 3 steps at peak LR,
val `loss_epoch` was 2.179 vs V16's 2.158 — but `loss_epoch` = ce_token +
confidence_loss, and decomposing shows the CE component essentially
IDENTICAL (1.9946 vs 1.9963) with all argmax slots / eal / accept_rate at
or slightly above V16 (trunk warm-start exact; p-slots +0.0005 from the 3
steps). The +0.021 is entirely the confidence head (a small AdamW-only
head whose pred_mean swung 0.157 → 0.067 → 0.148 across the 3 train steps
with empty optimizer state). Applied entry 7's contingency:
`scheduler_warmup_steps: 100` (~1% of budget), and the cooldown tracks
`loss − confidence_loss` / eal / p0 as its clean early signals.

**Pre-registered expectations (40h-stop analysis, entries 13/14):** the
constant-LR fit put the loss floor at ~2.118 and current trend at
~2.155–2.158 at the 40h mark. A 2B linear anneal should capture part of the
0.04-nat constant-LR→floor gap: final val loss ~2.13–2.15 (at least
−0.008 vs V16), monotone-ish descent tracking LR; eal 1.615–1.630 (flat to
+0.012 — V16 was the first non-positive eal cycle; annealing usually
sharpens argmax alignment); p0 0.356 → 0.360–0.365.

**Picking the pretrain back up (no repeated data):** copy the pretrain's
`checkpoints/0` (V16) to a fresh dir, edit its `training_state.json`
local_step/global_step 104768 → 114941 (= 104,768 + 10,173 cool-down
steps), keep optimizer/scheduler state, relaunch the pretrain config with
`save_path` pointed there (same 8-rank geometry; this commit is purely
additive vs the run's launch SHA e7c7248, so either tree works).
Auto-resume fast-skips to 114,941 and continues constant-LR training; Muon
moments are 10,173 steps stale (negligible at constant LR this late).

**Reproducibility:** branch @ this commit; dataset = entry 10's mix; warm
start = V16 snapshot (`fineweb-cooldown-run/warm-start/`); env unchanged
(torch 2.13.0+cu130, 8× H200 141GB, manual hold).

**Outstanding:** monitor the 4 boundaries + final (~2.9h train + ~15 min
overhead); compare against pre-registered expectations; then decide the
stage-1b endpoint (this anneal vs entry 7's chat cool-down) before stage-2
conversion (entry 8) + hs distillation.

### 15. 2026-10-02 — cool-down results: the anneal far outperformed the pre-registration

**Run.** The entry-14 cool-down ran to completion on 2026-10-01: launched
20:04, first step ~20:22 (~18 min startup), finished 23:24 — 10,173 steps
(~2.9 h at ~992 ms/step) + 4 boundary save/val overheads, ~3.3 h wall. The
skip landed as designed: "Fast-skipping 104768 batches", trained the
pretrain's batches 104,769–114,941 (2.000B tokens, zero repeated data);
mid-run checkpoints record true stream positions (0.prev1 = local_step
112,976 = cooldown step 8,208). Clean exit; final save at
`fineweb-cooldown-run/checkpoints/0` (global_step 10,173, `epoch0_end` /
`checkpoint_best` → 0; 8,208 and 5,752 boundaries kept in `0.prev1/2`).

**Results** (same 50.36M-token multilingual val tail; directly comparable
to the V1–V16 series):

| point | step (cooldown) | LR | val loss | p0 | eal |
|---|---|---|---|---|---|
| V16 start | 0 | 1e-3 const | 2.1582 | 0.3564 | 1.6181 |
| b1 | 840 | ~0.92e-3 | 2.141 | 0.362 | 1.637 |
| b2 | 3,296 | ~0.68e-3 | 2.111 | 0.374 | 1.675 |
| b3 | 5,752 | ~0.44e-3 | 2.087 | 0.383 | 1.703 |
| b4 | 8,208 | ~0.20e-3 | 2.075 | 0.388 | 1.721 |
| **final** | **10,173** | **→0** | **2.0726** | **0.3888** | **1.7229** |

Every metric monotone at every boundary. Final slots
0.3888/0.3403/0.2888/0.2592/0.2417/0.2304/0.2222/0.2160 (V16:
0.3564/0.3142/0.2673/0.2411/0.2258/0.2160/0.2091/0.2036); full_acc 0.2734
(V16 0.2542); accept_len 1.4437 (V16 1.3684); accept_rate 0.1559. The
confidence head re-equilibrated by b1 (conf loss 0.162 ≈ V16's 0.162;
final 0.166) — the entry-14 smoke transient was purely a fresh-optimizer
warm-up artifact; decomposing, the CE component fell 1.996 → 1.907 (−0.090)
and the anneal's total gain was −0.086.

**vs the entry-14 pre-registration** (final loss 2.13–2.15, eal
1.615–1.630, p0 0.360–0.365): actual 2.0726 / 1.7229 / 0.3888. The loss
gain was ~3–10× the predicted −0.008…−0.028, landing 0.045 BELOW the
fitted constant-LR floor (~2.118, entry 13's curve-fit). Two lessons:
(1) a constant-LR asymptote fit is an upper bound on the annealed
endpoint, not an estimate of it — the sharp-minima payoff of LR→0 is
invisible to constant-LR extrapolation; (2) the V16 "eal stall" (first
non-positive cycle, entry 13) was a constant-LR artifact, not a capacity
limit — under annealing every argmax slot sharpened and eal recovered
+0.105, landing mid-band in entry 13's 1.70–1.80 ceiling estimate. Anneal
shape was textbook: −0.047 of the loss gain by b2 (LR ~0.68e-3), only
−0.002 over the final ~2,000 steps at near-zero LR — consistent with
most of the payoff coming from mid-range LR reduction, not the last
epsilon of descent.

**Stage-1b is done.** Downstream conversion endpoint:
`fineweb-cooldown-run/checkpoints/0`. The pretrain remains resumable with
zero repeated data (recipe in the entry-14 config header). Open: optional
HF push of the annealed checkpoint; stage-2 conversion (fc zero-expansion,
entry 8) + hidden-state distillation.

**Reproducibility:** warm start = V16 snapshot (`fineweb-cooldown-run/
warm-start/`); untracked config `configs/qwen38-dspark-cooldown-fineweb.yaml`
(linear 1e-3 → 0 over 2e9 tokens, warmup 100, skip_steps 104,768, fresh
Muon, checkpoint_freq 0.0024, 8 ranks); launcher `~/launch-cooldown-fineweb.sh`;
dataset = entry 10's mix; code @ 63fe7df + this commit; env unchanged
(torch 2.13.0+cu130, 8× H200 141GB).

### 16. 2026-10-02 — stage-2 hidden-states pool: one capture serves the whole layer-ID search

**What/why.** The layer-ID search (stage 2) trains 5-layer-subset drafts
against pre-captured on-policy verifier hidden states, comparing val eal.
Rather than re-prefilling the corpus per candidate subset, capture **one
17-slot pool** — the union of every subset worth trying — and slice slots
by id at train time (entry 17). Pool ids: `0` (embeddings — Qwen layer 0
consumes the unscaled embedding, so id 0 is an exact `embed_tokens` gather)
and `4, 8, …, 64` (all 16 full-attention block outputs; ≡0 mod 4 because
full-attention sits at 0-indexed 3, 7, …, 63 and hidden state after layer
i carries id i+1). Id 64 doubles as the verifier target, matching the
online path's convention.

**Row sets.** `scripts/capture_pool.py compute` replicates the training
stream exactly: the sampler's epoch permutation (seed 0 — the class
default, not the config seed) over the entry-10 arrow dataset, packed by
the same LPT window logic (`_assign_to_packed_batches` outer loop with an
early break; the window is rank-independent, so the prefix of permuted
rows is the union over all 8 DP ranks). Cross-validated against the real
`MultipackDistributedBatchSamplerV2` (full packing × 8 ranks, union of
first-N batches == simulated prefix; exact match on synthetic data and on
the real train/val splits). Budgets: 1,606 train steps = 100.07M tokens
(95% LPT utilization — nominal 65,536 tok/step is wrong for step math),
161 val steps = 10.00M tokens, +8-step prefetch margins. Total: 45,591
rows / 111.06M tokens ≈ 19.3 TB bf16 (17 slots × 5,120 × 2 B/token).

**Capture.** vLLM (TP8) with the `extract_hidden_states` speculator
config, staging on /data; the async client driver (same request pattern
as `generate-offline-data`) moves each finished file into the pool by
rename and validates it (token-id match, finiteness, 17 slots).
Resumable via existing-file skip. Measured 15.2 rows/s at concurrency
32 — server prefill+extraction bound (concurrency 64 gave 14.3, disk
only ~5.6 of ~13 GB/s), so the full pool is ~50 min. **Slot-0 identity
verified bit-exact**: `hs[:, 0, :] == embed_tokens[token_ids]` (max|diff|
0.0 on sampled files) — the capture mechanism is sound end-to-end.
`pool_manifest.json` in the pool dir records the capture order of layer
ids + verifier id; `ArrowDataset` reads it (entry 17).

**Reproducibility:** rows at `/data/playground/reed-meyerson/
layer-id-search/rows.json`, pool + manifest under `layer-id-search/
hidden_states/`; server via `launch_vllm.py … --target-layer-ids 0 4 8
12 16 20 24 28 32 36 40 44 48 52 56 60` (launch auto-appends 64);
untracked launcher `~/launch-pool-server.sh`; code @ this commit.

### 17. 2026-10-02 — bounded mini-experiments + manifest-based slot selection

**What/why.** Two surgical library changes so the layer-ID search runs as
100M-token mini-experiments over the entry-16 pool:

1. **`max_train_batches` / `max_val_batches`** (DataArgs → cli →
   `create_train_val_loaders`; train side was already plumbed in the
   dataloader, val side and the config fields are new). 1,606 train
   batches makes `len(train_loader)` the true horizon — the LR schedule
   resolves from it (epochs × len), and the run consumes exactly the
   captured 100.07M-token prefix (no `scheduler_total_tokens`: it assumes
   nominal 65,536 tok/step vs the actual ~62.3K LPT-packed, and would
   misalign schedule and pool). 161 val batches gives one fixed ~10.0M-
   token val subset — the same first-161-permutation-batches for every
   candidate, since the sampler is seeded and epoch-0 deterministic.
   `on_missing=skip` was rejected for val bounding: it emits placeholder
   batches that pollute eal.

2. **Id-based slot selection** (`ArrowDataset.target_layer_ids`). The old
   `[:, :-1]` / `[:, -1]` slicing assumes files carry exactly this
   draft's layers + verifier. With `pool_manifest.json` present, slots
   are selected by layer id (config order preserved — the same order
   `expand_target_layers.py` warm-start assumes), with validation (ids ⊆
   captured; per-file slot count). No manifest → legacy positional
   convention; the online-generation path is untouched. Token-only
   datasets skip resolution entirely.

**Tests.** Slot selection bit-identical to manual file slices (5-layer
subset → fc `[seq, 25,600]`, verifier `[seq, 5,120]`); loader lengths 4/2
under `max_train_batches=4`/`max_val_batches=2`; collate widths and
bfloat16 dtype through a real packed batch; CLI mirror
(`--max-train-batches`/`--max-val-batches`); error paths (missing id,
manifest without layer ids). Embedding identity in entry 16.

**Reproducibility:** configs for the search runs live in `configs/`
(untracked); code @ this commit.

### 18. 2026-10-03 — layer-ID search resolved: depth wins; S3 {0,36,44,52,60} → full on-policy distillation launched

**What ran.** Five bounded mini-experiments (entry 17's harness), one per
candidate layer subset, each 100M tokens (1,606 batches) of on-policy
distillation from the same zero-expanded entry-15 cool-down draft, pool-
based (`on_missing=raise` — the entry-16 pool was complete), Muon 1e-4
linear (10% warmup), noise off, single end-of-run val on the fixed 161-
batch (~10M-token) subset. Subsets (all include layer 0 — the warm-start
seam, entry 8):

- S1 {0,4,20,36,52} reference-subsample; S2 {0,8,16,24,60} stride-8
  ladder; S3 {0,36,44,52,60} high-lean; S4 {0,8,16,52,60} bimodal
  (vs S2: only 24↔52); S5 {0,8,24,40,60} even+endpoints.

**Results** (val eal; every metric — eal/accept_len/p0/ce/loss — agrees
on the ordering; smoke's untrained-expansion baseline: 1.621):

| run | val eal | p0 | ce |
|-----|---------|-----|-----|
| **S3 high-lean** | **2.639** | 0.686 | 1.367 |
| S4 bimodal | 2.624 | 0.685 | 1.374 |
| S5 even | 2.589 | 0.681 | 1.391 |
| S2 ladder | 2.575 | 0.679 | 1.399 |
| S1 refsub | 2.550 | 0.646 | 1.406 |

**Read.** S3 beats S1 by +0.089 (~18× the ±0.005 subset noise floor).
Both controlled swaps favor depth: S4−S2 = +0.049 (24→52) and S3−S5 =
+0.050 (high-lean vs spread). S1 — the only set without slot 60 — lands
last with distinctly weak p0 (0.646 vs ~0.68): the top full-attention
output (nearest the layer-64 verifier target) is load-bearing. The
EAGLE3-style low-layer bias does not transfer to this model pair; the
verifier's deepest layers carry the most draft-predictive signal.
S3 vs S4 (+0.015, ~3× the floor) is the one comparison inside polite
striking distance of noise; accepted S3 as the round-1 winner and moved
to the full run rather than spend a refinement round on a ±1-slot grid.

**Follow-on (the full run).** Re-did the cool-down on-policy before
distilling: V16 → 100M-token token-only cool-down on the regen stream
(509 steps, Muon 1e-3 linear→0, warmup 100; val on the regen tail:
eal 1.916, p0 0.444, loss 1.849 — different distribution from entry 15's
FineWeb val, not comparable) → fc zero-expansion to S3 (slot-0 block
bit-exact, 4 zero blocks) → 1-epoch on-policy distillation: 4 train
ranks + TP4 hidden-states server, ce 0.1/tv 0.9, Muon 1e-3, linear with
cold-parity absolute warmup 2,867 steps over the 4.695B-token epoch,
noise off, checkpoint/val every 7,165 steps (keep 3). Launched
2026-10-03 20:05:47 UTC; measured 0.92 s/step (vs 1.014 on the entry-13
8-slot cold-start — the 5-slot fc is cheaper), ETA ~44 h. Two server
launch gotchas fixed en route: vLLM's JIT needs `ninja` (venv bin must
be in PATH for the server process), and native max_model_len 262144
demands 324 GB KV > the ~111 GB available at TP4 — cap `--max-model-len`
at 8,200 (training requests are ≤8,192).

**Reproducibility:** untracked configs `configs/qwen38-dspark-layerid-
S{1..5}.yaml`, `configs/qwen38-dspark-cooldown-regen-100m.yaml`,
`configs/qwen38-dspark-distill-regen-s3-1ep-linlr.yaml`; warm start =
`layer-id-search/warmstarts/S{1..5}` (from entry-15 ckpt) and
`s3-distill/warmstart` (from the on-policy cool-down); pool per entry 16;
code @ 059165b; env unchanged.

### 19. 2026-10-05 — full S3 on-policy distillation complete: val eal 4.372 in one epoch

**What ran.** The entry-18 plan, executed end-to-end: V16 → 100M-token
on-policy token-only cool-down on the regen stream (end val eal 1.916,
token-only accept semantics) → fc zero-expansion to S3 {0,36,44,52,60} →
1-epoch on-policy distillation: ce 0.1/tv 0.9, Muon 1e-3 linear→0 with
cold-parity absolute warmup 2,867 steps, noise off, 4 train ranks + TP4
verifier hidden-states server (max_model_len capped 8,200 — native
262144 wants 324 GB KV > ~111 GB at TP4), checkpoint+val every 7,165
steps (keep 3). Exactly 143,291 steps × 32,768 tokens = 4.695B tokens,
one pass, no repeats. Wall 44.1 h — 0.92 s/step train, 1.11 s/step
effective including 20 vals at ~23 min each.

**Results** (val = full regen tail, distillation-mode accept semantics —
draft argmax vs verifier argmax from captured layer-64 states):

- **final: eal 4.372**, accept_len 4.234, position-0 acc 0.838,
  full-sequence acc 0.619, loss 0.425 (ce 0.708, conf 0.209)
- trajectory (eal at each val): 3.686 @ 7.2K steps (5% in) → 3.984 @
  21.5K (15%) → 4.282 @ 71.6K (50%) → 4.366 @ 114.6K (80%) → 4.372 @
  129.0K (90%) → 4.372 final. Classic saturation: +0.001 over the final
  14K steps at near-zero LR (last three vals 4.372/4.373/4.372).
- epoch-mean train eal 5.90 vs val 4.37 — the draft fits each stream
  segment as it passes (one-epoch memorization); val is the honest
  number.

**Read.**
- vs the S3 mini-experiment (entry 18: 2.639 after 100M tokens @ 1e-4):
  **+1.73**. Same warm-start lineage and val distribution — the delta is
  recipe (47× tokens, 10× LR, proper warmup-to-peak), not architecture;
  the slot choice was already settled by the search.
- The first val (5% through, LR ~0.9e-3) already read 3.686 — above
  every mini-experiment endpoint. Most of the mini-vs-full gap accrues
  in the first fraction of the epoch.
- The cool-down endpoint's 1.916 is a proxy, not a like-for-like
  baseline: token-argmax match (token-only semantics) vs
  verifier-argmax agreement. On this on-policy corpus (verifier-sampled)
  they are kin in spirit but not numerically comparable.
- No cold-start comparison exists at full scale — the fresh-2ep
  reference died at step 1,273 of 286,582 with no checkpoints/vals.

**Ops notes.** Post-run server teardown: C-c stops the API server but
orphaned TP workers (131 GB each) survive it — kill by PID, then the
reservation wrapper exits and the hold releases. 7.3 GB of unconsumed
in-flight hs cache files remained after on_generate=delete consumed the
rest; cleared. The 18T layer-ID pool retained for now.

**Reproducibility:** untracked config
`configs/qwen38-dspark-distill-regen-s3-1ep-linlr.yaml`; warm start =
`s3-distill/warmstart` (entry 18); dataset = regen stream; run tree
archived in `s3-distill/checkpoints/` (`speculators.patch`, `run.yaml`;
code @ 059165b — no code changes since entry 17); checkpoints
`{0, 0.prev1, 0.prev2, epoch0_end}`; log `s3-distill/logs/train.log`;
trackio `qwen38-dspark-distill-regen-s3-1ep-linlr`; env unchanged.

### 20. 2026-10-06 — verifier hidden-state RMS grows ~400× with depth (pool measurement)

**What/why.** A CPU-only pass over the entry-16 pool
(`scripts/estimate_layer_rms.py`) measured the RMS norm of the
verifier's hidden states as a function of layer id — the raw magnitudes
the draft's fc consumes (hidden states enter the fc unnormalized).
Method: seeded sample of ~1,000 tokens (50 files × 20 contiguous
positions, lazy safetensors slice reads — no full-file loads, no GPU),
the SAME token set for every layer; stable to ~1% across seeds
42/7/123.

**Results** (RMS over sampled (token, dim) elements; mean per-token L2
≈ RMS × 71.5):

| layer | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 | 44 | 48 | 52 | 56 | 60 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RMS | .014 | .34 | .48 | .67 | .83 | .85 | .92 | 1.03 | 1.13 | 1.16 | 1.24 | 1.38 | 1.58 | 2.00 | 2.83 | 3.68 | 5.66 |

Monotone growth, 407× end to end, super-linear in the tail (L52→L64
nearly triples; L60→L64 alone +54%).

**Read.**
- The layer-ID search's "depth wins" (entry 18) has a magnitude
  counterpart: the deepest states are also the highest-energy ones —
  the fc's per-slot signal energy tracks depth. S3 {0,36,44,52,60}
  leans into exactly the high-energy range.
- Layer 0's RMS (0.014 — ~25× below L4, ~250× below deep layers) is an
  embedding-scale artifact: the model divides embeddings down hard.
  Its slot is in every subset as the warm-start seam (entry 8), not for
  signal energy; per-slot input scaling (or a norm ahead of the fc) is
  an obvious future lever.
- The steep L60→L64 jump is the readout approach: the verifier's final
  state — the distillation target proper — is simultaneously the
  largest-magnitude and most draft-predictive representation.

**Reproducibility:** `scripts/estimate_layer_rms.py` @ this commit;
pool per entry 16; JSON record `layer-id-search/layer_rms_sample.json`
(seed 42); ran on CPU alongside the in-flight token-only run (entry 22).

### 21. 2026-10-06 — design note: per-slot pre-fc scaling of verifier hidden states (NOT implemented; recipe for later)

**Motivation.** Entry 20: the draft's fc consumes raw verifier states
with a ~400x per-slot RMS spread (L0 0.014 vs L60 3.68). Nothing
normalizes the fc input (`_backbone_forward` applies `hidden_norm` only
AFTER the projection), so the imbalance is absorbed by fc weight
magnitudes — and since the output is immediately RMSNormed, it manifests
as per-slot conditioning / effective-LR asymmetry (L0's columns must grow
~250x to match deep slots' energy). Scaling every aux slot to layer-0's
std would level the field — and, because L0's state IS the unscaled
embedding (core.py:381) and the token-only branch feeds `embed(input_ids)`
into the same fc, it would also make token-only -> distillation
warm-starts distribution-consistent.

**Design (agreed 2026-10-06, deliberately deferred).**
- Scale INSIDE the model, ahead of the fc: inference inherits via
  config.json (from_pretrained); zero changes to the data pipeline,
  the hs server, or the 18T pool (scales apply at consume time).
- `DFlashSpeculatorConfig.aux_hidden_state_scales: list[float] | None =
  None`, aligned index-for-index with `aux_hidden_state_layer_ids` in
  CONFIG order (the dataset flattens slots in config order — not sorted
  order). `None` = raw states = every existing checkpoint loads
  unchanged.
- `_backbone_forward` distillation branch: reshape [1,T,n*5120] ->
  [1,T,n,5120], multiply by the per-slot scales, flatten, then fc.
  Token-only branch (embeddings = the reference scale) and the
  verifier-side target construction (`verifier_norm` + L64) untouched.
- Scales measured offline by extending `scripts/estimate_layer_rms.py`
  with `--emit-scales` (scale_l = rms_0 / rms_l). Seed-42 values:
  L0 1.0, L4 0.041, L8 0.029, L36 0.012, L52 0.0070, L60 0.0038.
  L64 is never scaled (verifier target, not an aux slot).
- `expand_target_layers.py`: propagate the field when present (new zero
  slots: scale 1.0 or the measured value).
- Estimated ~120-140 lines across 5 files: config ~15, core ~25, script
  ~10, expand ~5, tests ~70 (distill path scales; token-only path
  doesn't; teacher target untouched; order alignment; config
  round-trip). One commit + entry.

**Caveats.**
- A function-preserving retrofit of an existing checkpoint exists —
  fc(s*hs) == (W diag(s)) hs, so dividing slot-l fc columns by s_l and
  setting the config scales is bit-identical — but it is NOT
  dynamics-neutral under Muon (spectral-normalized updates see the
  rescaled geometry), so experiments should start from a fresh fc or a
  zero-expansion warm-start rather than a retrofit.
- Scales are a fixed prior (measured once, ~1% sample noise across
  seeds); per-layer RMS is a stable model property but not adaptive
  across domains.

**Trigger to implement:** any future layer-subset experiment where input
conditioning is suspected — e.g. revisiting low-layer subsets post-S3,
or training a fresh fc from scratch.

### 22. 2026-10-06 — full-epoch token-only ablation complete: val eal 2.479; follow-on S3 distillation launched from it

**What ran.** The entry-20-announced ablation: V16 (the FineWeb-cooldown
warm-start, NOT the S3-pipeline cooldown) -> one full epoch of on-policy
token-only training — layer-0 only, `ce_token`, Muon 1e-3 linear->0 over
the epoch, 1% warmup, noise off, 8 ranks, no server. 22,040 steps
(~4.3B tokens, one pass of the regen train stream; 196,608 nominal
tok/step, ~95% LPT utilization), 20 mid-epoch vals on the full regen
tail (token-only accept semantics — directly comparable to the 100M
cooldown's 1.916). Wall 6h 50m (2026-10-06 13:59:24 -> 20:49:30 UTC,
0.92 s/step, val pause ~52 s each). Launched + monitored via
`canhazgpu run --gpus 8`; GPUs auto-released on exit.

**Results** (final val, step 22,040): eal 2.479, p0 0.5472, p1 0.4960,
full_acc 0.4100, accept_len 2.0059, loss 1.4967 (confidence head 0.2114).
Trajectory: 2.022 @ 5% -> 2.290 @ 30% -> 2.408 @ 55% -> 2.470 @ 80% ->
2.479 final; p0 0.466 -> 0.547.

- **vs the 100M cooldown endpoint (1.916, same semantics): +0.563.**
  The 43x token budget buys a real but strongly saturating gain —
  ~50% of it lands in the first 5% of the epoch, ~80% by 30%.
- **vs entry 19's distill final (4.372): not comparable** — hard-label
  match vs verifier-argmax agreement. The follow-on run below measures
  this checkpoint under distill semantics.

**Forecast verification.** Saturating-curve fits (exp-approach, power,
sat-power, log, geometric-gain-decay) with per-family bias correction
calibrated on the entry-19 distill run's own first-N vals (known final)
predicted 2.48 +- 0.02 eal / 0.546 +- 0.003 p0 at the 80% mark; actuals
2.479 / 0.547. The shape-transfer estimate (distill covered 99.1% of
its eal gain by 80%) was the most accurate single predictor. The
forecast history across updates (2.42 -> 2.43 -> 2.45 -> 2.46 -> 2.47 ->
2.48) converged monotonically to the truth — useful procedure to reuse.

**Follow-on launched: full 1-epoch S3 distillation from THIS endpoint.**
Identical recipe to entry 19 (stream, geometry 8192/1024, ce 0.1/tv 0.9,
Muon 1e-3, linear with cold-parity warmup 2,867 steps, noise off,
143,291 steps = one epoch, ckpt+val every 7,165 steps keep 3) — the ONLY
change is the warm start:

  entry 19: V16 -> 100M-token token-only cooldown -> expand -> distill
  this run: V16 -> FULL-EPOCH token-only (this entry) -> expand -> distill

This isolates the token-only stage's budget: does 4.3B tokens of
token-only pretraining beat 100M once the teacher epoch runs, and where
does the combined pipeline land vs 4.372?

- Expansion verified (same protocol as entry 18): slot-0 fc block
  bit-exact vs the token-only endpoint, 4 zero blocks (layers 36/44/52/
  60), config aux ids [0, 36, 44, 52, 60]; warmstart at
  `tokenonly-s3distill/warmstart`.
- Server: TP4 vLLM hs server (max-model-len 8200) via
  `canhazgpu run --gpus 4`, healthy in ~2.3 min (JIT cache warm).
- Training: 4 ranks via `canhazgpu run --gpus 4`, launched 2026-10-06
  20:53 UTC. Verified healthy: 0.91 s/step (matches entry 19's 0.92),
  LR exactly on the 2,867-step warmup ramp, hs fetch_ms ~15-37,
  error_records 0. ETA ~44 h -> ~2026-10-08 17:00 UTC.

**Reproducibility:** configs `configs/qwen38-dspark-tokenonly-regen-1ep.yaml`
(entry-22 run) and `configs/qwen38-dspark-distill-s3-1ep-linlr-fromtokenonly.yaml`
(follow-on); outputs `onpolicy-tokenonly-1ep/` (checkpoints epoch0_end ->
0, keep-3, val_metrics.json) and `tokenonly-s3distill/{warmstart,
checkpoints,logs,hidden-states}`; trackio runs
`qwen38-dspark-tokenonly-regen-1ep` and
`qwen38-dspark-distill-s3-1ep-linlr-fromtokenonly`; code @ this commit.

**Update (abort).** Killed at 35% (step 50,702, 2026-10-07) on user
decision. Seven vals in, the warm-start ablation had answered its
question: a consistent but shrinking lead over entry 19 —
+0.041 / +0.029 / +0.024 / +0.024 / +0.021 / +0.020 / +0.019 eal at
5-35% (3.727 -> 4.210 vs 3.686 -> 4.191 at identical boundaries).
Gap decay ~0.003/boundary projects final ~= parity with entry 19's
4.372 — the 43x-larger token-only budget before the distill epoch buys
~+0.02 mid-run but not a better endpoint. Not worth the remaining ~29 h.
Both reservations released cleanly (8/8 AVAILABLE, baseline memory);
keep-3 checkpoints from 25/30/35% preserved under
`tokenonly-s3distill/checkpoints/`.

Ops note: the server went down via an over-broad `pkill -9 -f
"speculators.train"` (dots match any char; it SIGKILL'd a vLLM
ApiServer worker whose cmdline matched the regex, and the caller's own
shell). Intended outcome (full teardown) but wrong mechanism — kill
run processes by exact pattern or PID, and C-c the tmux windows (which
worked cleanly for torchrun).

### 23. 2026-10-07 — entry-21 per-slot pre-fc input scaling implemented; normalized S3 distillation launched from the token-only endpoint

**What was built.** The entry-21 design, implemented exactly as specified:
`DFlashSpeculatorConfig.aux_hidden_state_scales: list[float] | None`
(aligned index-for-index with `aux_hidden_state_layer_ids` in CONFIG
order; validated positive + length-matched; `None` = raw states = every
existing checkpoint loads unchanged), applied ONLY in the
`_backbone_forward` distillation branch — reshape [1,T,n*5120] ->
[1,T,n,5120], multiply by the per-slot scales, flatten, then fc. The
token-only (embedding) branch and the verifier-side target construction
are untouched. Inference inherits the scales via config.json
(from_pretrained), so the data pipeline, hs server, and 18T pool are
unmodified — scales apply at consume time. Tooling:
`estimate_layer_rms.py --emit-scales` (scale_l = rms(lowest id)/rms_l,
recorded in the JSON) and `expand_target_layers.py --slot-scales`
(aligned with the new ids; when omitted, propagates source scales with
new slots at 1.0; warns when an old slot's effective scale changes —
NOT function-preserving).

**Engineering note (meta-device init).** First implementation registered
a non-persistent fp32 buffer — and from_pretrained loaded it as
UNINITIALIZED memory (garbage values): transformers 5.x constructs
models under a `torch.device("meta")` context, where tensors created in
`__init__` land on meta and `to_empty()` later materializes anything
absent from the checkpoint (non-persistent buffers are, by design) as
uninitialized bits. Fix: the scale tensor is now derived from the CONFIG
at forward time (`_aux_scale_tensor`, lazily cached in `__dict__` keyed
by device/dtype) — config is the single source of truth and is immune
to meta init. Regression test simulates the exact failure (meta-context
construct + to_empty + check). Lesson: never register config-derived
constants as buffers in this codebase.

**Scales measured** (seed 42, `layer-id-search/layer_rms_sample_s3scales.json`,
from the entry-16 pool): for S3 ids [0, 36, 44, 52, 60] ->
[1.0, 0.01197847, 0.01005724, 0.00696475, 0.00377651]. L44 = 0.01005724
(rms 1.3817) — the one value missing from entry 21's table.

**Verification before launch.**
- 9 new unit tests (`tests/unit/models/test_dflash_aux_scales.py`):
  distill-path scaling in config order (unsorted ids [4,0]), token-only
  path unaffected (non-unity scale on single-slot), verifier targets
  untouched, config validator, meta-init survival, config round-trip
  via save_pretrained/from_pretrained, expand script scales/propagation/
  warning via subprocess.
- Full suite: 265 passed (tests/unit/models + test_config), run under
  `canhazgpu run` (CUDA-dependent tests included).
- Real warmstart end-to-end: from_pretrained loads scales [1.0, 0.012,
  0.010, 0.007, 0.0038]; eager forward == manual pre-scaling of slot
  blocks (max logits diff 0.0); compiled forward on CUDA bf16 works,
  deterministic, compiled == eager (0.0) — compile compatibility of the
  new ops confirmed before the multi-day launch.
- Expansion verified (entry-18/22 protocol): slot-0 fc block bit-exact
  vs the token-only endpoint (scale 1.0 keeps the warm start
  function-preserving — zero-expansion satisfies entry-21's Muon
  caveat), 4 zero blocks (36/44/52/60), all 61 non-fc weights identical,
  config ids+scales present. Warmstart at
  `tokenonly-s3distill-norm/warmstart`.

**Run launched.** Entry-22's ablation re-run WITH normalization: full
1-epoch on-policy S3 distillation from the token-only full-epoch
endpoint (eal 2.479), scales active. Identical recipe to entry 19 /
the killed follow-on (stream, geometry 8192/1024, ce 0.1/tv 0.9, Muon
1e-3, linear LR with cold-parity warmup 2,867 steps, noise off,
143,291 steps = one epoch, ckpt+val every 7,165 keep 3) — the ONLY
change is per-slot pre-fc scaling.

- Question: does leveling the ~400x per-slot RMS spread at the fc input
  beat the un-normalized trajectory (killed follow-on: parity with
  entry 19's 4.372, projected ~4.21; entry 19 = 4.372)?
- Server: TP4 vLLM hs server (S3 ids, max-model-len 8200) via
  `canhazgpu run --gpus 4`, healthy in ~150 s.
- Training: 4 ranks via `canhazgpu run --gpus 4`, launched 2026-10-07
  14:18 UTC (tmux `tokenonly-s3distill-norm`). Verified healthy at step
  200: LR exactly on the 2,867-step warmup ramp (6.98e-05), Muon params
  36/26 (identical to the un-normalized run — same shapes),
  error_records 0, fetch_ms ~15. Step rate 0.66 s/step (faster than
  the 0.91-0.92 family — ETA ~26 h at this rate vs 36-44 h prior;
  monitor).

**Reproducibility:** config
`configs/qwen38-dspark-distill-s3-1ep-linlr-fromtokenonly-norm.yaml`;
scales JSON `layer-id-search/layer_rms_sample_s3scales.json`; outputs
`tokenonly-s3distill-norm/{warmstart,checkpoints,logs,hidden-states}`;
trackio run `qwen38-dspark-distill-s3-1ep-linlr-fromtokenonly-norm`;
code @ this commit.
