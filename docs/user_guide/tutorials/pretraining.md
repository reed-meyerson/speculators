# Pretraining a Draft Model

Speculator drafts are normally trained by distillation: the verifier runs over every token of the training data, and the draft learns to match the distribution it produces. That verifier pass is the expensive part, and it caps how much data a draft can afford to see.

Pretraining removes it. Instead of matching the verifier's distribution, the draft predicts the corpus's own next tokens from its frozen copy of the verifier's input embedding. No verifier forward pass, no vLLM server, no hidden-state extraction, no prepared dataset — just raw text. Tokens become cheap enough to spend billions of them, and the result is a checkpoint that an ordinary distillation run warm-starts from.

Pretraining is a *mode*, not an algorithm. It applies to [DFlash](../algorithms/dflash.md), [DFlash2](../algorithms/dflash2.md) and [DSpark](../algorithms/dspark.md), all of which project auxiliary verifier hidden states through the same `fc` layer.

## The three steps

```bash
# 1. Pretrain. One auxiliary layer -- the embedding is all there is to learn from.
speculators train \
    --training-mode pretrain \
    --speculator-type dflash \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --target-layer-ids 0 \
    --pretrain-dataset HuggingFaceFW/fineweb \
    --pretrain-dataset-config sample-10BT \
    --pretrain-token-budget 1000000000 \
    --total-seq-len 8192 \
    --save-path ./output/pretrain

# 2. Widen the projection to the layers distillation will use.
speculators expand-aux-layers ./output/pretrain/checkpoint_best 0 18 33 \
    --output ./output/pretrain-wide

# 3. Ordinary distillation, warm-started.
speculators train \
    --speculator-type dflash \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --from-pretrained ./output/pretrain-wide \
    --data-path ./output/data \
    --vllm-endpoint http://localhost:8000/v1 \
    --total-seq-len 8192 \
    --save-path ./output/distill
```

`examples/train/dflash_qwen3_8b_pretrain_then_distill.sh` runs all three end to end.

Pretraining learns from the embedding alone, so the checkpoint carries no commitment to which auxiliary layers a finetune will use, or how many. Step 2 is where that choice is made, which means one pretraining run can be widened several different ways and feed a whole layer-selection sweep.

`--from-pretrained` takes the auxiliary layer ids from the checkpoint, so `--target-layer-ids` is ignored in step 3. The selection is whatever you passed to `expand-aux-layers`, and vLLM must be launched extracting those same layers.

## How the warm start works

Pretraining trains the `fc` columns fed by verifier layer 0 -- the embedding. Widening adds columns for the remaining auxiliary layers and leaves them at exactly zero.

That is what makes the handoff exact. When distillation starts, the auxiliary hidden states arrive and their contribution grows from nothing, rather than from a random initialization that would swamp the features pretraining just learned. Distillation begins precisely where pretraining left off.

A checkpoint that has not been widened is still a valid single-auxiliary-layer drafter, so it can be served and evaluated on its own before you commit to a selection.

## The token budget sets the run length

`--pretrain-token-budget` counts tokens across all ranks, and fixes how many packed sequences each rank produces. The epoch therefore ends on its own; there is no need to pass `--max-steps`, and the learning-rate schedule resolves from the budget like any other run.

Documents are streamed, tokenized and packed to `--total-seq-len`, separated by the verifier's EOS token. Position ids restart at each document and attention never crosses a document boundary, so packing does not leak context between unrelated documents.

## Using a local corpus

`--pretrain-data-files` streams local files instead of pulling from the Hub. The `--pretrain-dataset` flag then names the loader rather than a dataset:

```bash
speculators train \
    --training-mode pretrain \
    --pretrain-dataset json \
    --pretrain-data-files /data/corpus-00.jsonl /data/corpus-01.jsonl \
    ...
```

Use `--pretrain-text-column` if the documents are not under `text`.

## Verifiers that scale their embedding

The draft's frozen embedding stands in for the verifier's layer-0 hidden state, so the two have to match. For most architectures they do: Llama, Mistral, Qwen, DeepSeek, GLM, Phi, gpt-oss, MiniMax, OLMo, Nemotron and Kimi all feed layer 0 the unscaled embedding.

Two families scale it first — Gemma by `sqrt(hidden_size)` and Granite by `config.embedding_multiplier` — so their layer-0 hidden state is the scaled embedding. Pretraining applies the same factor and works normally; no flag is needed. The factor is resolved from the verifier's config by `verifier_embedding_scale` in `src/speculators/models/utils.py`, and an architecture that is not listed there is assumed to be unscaled.

## Loss

Pretraining scores hard token ids, so it uses cross-entropy; `--loss-fn` is defaulted to `ce` for you and a distributional loss such as `kl_div` is rejected. Everything else — D-PACE position weighting, block size, anchors, sliding-window attention — behaves exactly as it does in distillation.
