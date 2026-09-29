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
