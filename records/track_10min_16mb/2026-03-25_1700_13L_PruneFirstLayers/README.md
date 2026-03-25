# Exp 92b: Progressive Pruning -- Remove First 2 Layers (13L->11L)

## Base
Exp 87 v2 (Progressive Pruning, BI-based layer removal) at
`records/track_10min_16mb/2026-03-25_1100_13L_StructuredPruning/`

## Changes
- **Force remove layers [0, 1]** instead of BI-lowest layers. Controlled by
  `PRUNE_FIRST_LAYERS=1` (default enabled).
- BI scores are still measured and logged for diagnostics, but not used for
  layer selection when `force_first=True`.
- `PRUNE_AT_FRAC` changed from 0.60 to **0.50** (same as Exp 92a) for more
  recovery time after pruning.

## Rationale
In Exp 87, BI consistently identified layers 9 and 10 (deep layers) as having
the lowest influence. However, the first layers (0, 1) handle low-level token
processing that might be redundant with the bigram/smear embedding pipeline.
Removing them forces the model to rely more on the embedding pipeline and
tests whether the early transformer layers are truly necessary or if their
work is already captured by the input embeddings.

## Expected Results
Experimental -- may be worse than BI-based pruning, but worth testing the
hypothesis that early layers are redundant with the embedding pipeline.
