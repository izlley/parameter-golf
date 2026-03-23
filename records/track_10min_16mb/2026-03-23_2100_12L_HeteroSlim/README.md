# 12L Hetero Slim (Exp 49)

## Purpose
Test a heterogeneous (mixed) layer architecture to fit 12 layers within 16MB.
Uses 6 slim encoder layers (MLP 2.0x, KV 1) followed by 6 full decoder layers (MLP 3.0x, KV 4).
The slim encoder layers save parameters by using smaller MLP expansion and single KV head,
allowing more total layers for deeper representation while staying within the size budget.

## Approach
- 12 layers total with heterogeneous per-layer configs via `layer_configs`
- Layers 0-5: Slim encoder (mlp_mult=2.0, num_kv_heads=1) -- lightweight feature extraction
- Layers 6-11: Full decoder (mlp_mult=3.0, num_kv_heads=4) -- full capacity for generation
- Block class modified to support mlp_mult=0 (attn-only layers) for future flexibility
- GPT class accepts per-layer (mlp_mult, num_kv_heads) tuples
- swa_start_frac: 0.50 -> 0.35 (earlier SWA averaging for more smoothing with deeper model)
- prune_quantile: 0.05 (magnitude pruning enabled)

## Changes (vs SOTA)
- num_layers: 10 -> 12
- swa_start_frac: 0.50 -> 0.35
- Added layer_configs: [(2.0, 1)] * 6 + [(3.0, 4)] * 6
- Added prune_quantile hyperparameter
- Block supports conditional MLP (mlp_mult=0 -> attn-only)

## Expected Results
- Target ~16.0MB model size
- Deeper model should improve representation quality
- Slim encoder layers provide efficient early processing
- Full decoder layers maintain generation quality

## Actual Results
(to be recorded after training)
