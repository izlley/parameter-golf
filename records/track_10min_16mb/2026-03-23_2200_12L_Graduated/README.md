# 12L Graduated (Exp 50)

## Purpose
Test a 3-tier graduated heterogeneous layer architecture to fit 12 layers within 16MB.
Uses progressively wider layers: 4 mini, 4 slim, 4 full.
This mirrors the intuition that early layers need less capacity (simple features)
while later layers benefit from more capacity (complex reasoning/generation).

## Approach
- 12 layers total with 3-tier graduated per-layer configs via `layer_configs`
- Layers 0-3: Mini (mlp_mult=1.5, num_kv_heads=1) -- minimal feature extraction
- Layers 4-7: Slim (mlp_mult=2.5, num_kv_heads=2) -- intermediate processing
- Layers 8-11: Full (mlp_mult=3.0, num_kv_heads=4) -- full capacity generation
- Block class modified to support mlp_mult=0 (attn-only layers) for future flexibility
- GPT class accepts per-layer (mlp_mult, num_kv_heads) tuples
- swa_start_frac: 0.50 -> 0.35 (earlier SWA averaging for more smoothing with deeper model)
- prune_quantile: 0.0 (no pruning, since graduated model is smaller)

## Changes (vs SOTA)
- num_layers: 10 -> 12
- swa_start_frac: 0.50 -> 0.35
- Added layer_configs: [(1.5, 1)] * 4 + [(2.5, 2)] * 4 + [(3.0, 4)] * 4
- Added prune_quantile hyperparameter (set to 0.0, pruning disabled)
- Block supports conditional MLP (mlp_mult=0 -> attn-only)

## Expected Results
- Smaller total model size than HeteroSlim due to more aggressive early-layer compression
- 3-tier graduation may better match natural layer importance distribution
- No pruning needed since model already fits comfortably in budget

## Actual Results
(to be recorded after training)
