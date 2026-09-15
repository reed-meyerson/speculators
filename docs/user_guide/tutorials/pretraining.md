# Pretraining a Draft Model

Speculator drafts are normally trained by distillation: the verifier runs over every token of the training data, and the draft learns to match the distribution it produces. That verifier pass is the expensive part, and it caps how much data a draft can afford to see.

Pretraining removes it. Instead of matching the verifier's distribution, the draft predicts the corpus's own next tokens from its frozen copy of the verifier's input embedding. No verifier forward pass, no vLLM server, no hidden-state extraction, no prepared dataset — just raw text. Tokens become cheap enough to spend billions of them, and the result is a checkpoint that an ordinary distillation run warm-starts from.

Pretraining is a *mode*, not an algorithm. It applies to [DFlash](../algorithms/dflash.md), [DFlash2](../algorithms/dflash2.md) and [DSpark](../algorithms/dspark.md), all of which project auxiliary verifier hidden states through the same `fc` layer.

## The two stages

```bash
# Stage 1 -- no vLLM, no data preparation
speculators train \
    --training-mode pretrain \
    --speculator-type dflash \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --target-layer-ids 0 18 33 \
    --pretrain-dataset HuggingFaceFW/fineweb \
    --pretrain-dataset-config sample-10BT \
    --pretrain-token-budget 1000000000 \
    --total-seq-len 8192 \
    --save-path ./output/pretrain

# Stage 2 -- ordinary distillation, warm-started
speculators train \
    --speculator-type dflash \
    --verifier-name-or-path Qwen/Qwen3-8B \
    --target-layer-ids 0 18 33 \
    --from-pretrained ./output/pretrain/checkpoint_best \
    --data-path ./output/data \
    --vllm-endpoint http://localhost:8000/v1 \
    --total-seq-len 8192 \
    --save-path ./output/distill
```

`examples/train/dflash_qwen3_8b_pretrain_then_distill.sh` runs both end to end.

There is no conversion step between the stages, and there is not meant to be: a pretrained checkpoint *is* a DFlash checkpoint. It declares itself as `dflash`, carries no trace of the training mode, and loads with plain `--from-pretrained`.

## Layer 0 must be in `--target-layer-ids`

This is the one rule that catches people, and it is not the default.

Pretraining has only the embedding to feed the draft, so it drives exactly the `fc` slot that distillation fills with verifier layer 0 — and layer 0 *is* the embedding. Without layer 0 among the target layers there is no such slot, and the run stops with an error rather than training something meaningless.

Both stages must also pass the **same** ids. The target layers set the width of `fc`, so a stage-2 run with a different selection cannot load the stage-1 checkpoint at all.

## How the warm start works

Pretraining trains only the `fc` columns fed by layer 0 and holds the rest at exactly zero. Those columns receive no input during pretraining and therefore no gradient, so they reach the checkpoint still zero.

That is what makes the handoff exact. When distillation starts, the auxiliary hidden states arrive and their contribution grows from nothing, rather than from a random initialization that would swamp the features pretraining just learned. Stage 2 begins precisely where stage 1 left off.

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

## Supported verifiers

Pretraining assumes the verifier feeds its first layer the unscaled embedding, which is what makes the draft's frozen copy a faithful stand-in for layer 0. That holds for Llama, Mistral, Qwen, DeepSeek, GLM, Phi, gpt-oss, MiniMax, OLMo, Nemotron and Kimi.

It does not hold for the Gemma family, which multiplies embeddings by `sqrt(hidden_size)`, or for Granite, which applies an `embedding_multiplier`. Those verifiers are rejected rather than silently trained against mis-scaled features. See `SCALED_EMBEDDING_MODEL_TYPES` in `src/speculators/models/utils.py`.

## Loss

Pretraining scores hard token ids, so it uses cross-entropy; `--loss-fn` is defaulted to `ce` for you and a distributional loss such as `kl_div` is rejected. Everything else — D-PACE position weighting, block size, anchors, sliding-window attention — behaves exactly as it does in distillation.
