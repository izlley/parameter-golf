# 14L Mixed AttnOnly (Exp 52)

## Purpose
Test heterogeneous (mixed) layer architecture with attention-only layers to maximize depth within 16MB.
4 attention-only layers (MLP 0, KV 2) + 3 slim layers (MLP 2.0x, KV 2) + 7 full decoder layers (MLP 3.0x, KV 4).

## Approach
- Attention-only layers have zero MLP parameters, providing very cheap additional depth
- Slim layers use reduced MLP (2.0x) and fewer KV heads (2) for intermediate capacity
- Full decoder layers handle the heavy lifting with standard 3.0x MLP and 4 KV heads
- This allows 14 total layers vs 10 baseline, testing if depth > width tradeoff is favorable
- Earlier SWA start (0.35) for deeper model
- Higher prune quantile (0.06) to compensate for more parameters

## Changes (vs base 10L SWA050)
- num_layers: 10 -> 14
- swa_start_frac: 0.50 -> 0.35
- prune_quantile: 0.06 (was hardcoded 0.05)
- Block class: supports mlp_mult=0 (attn-only layers)
- GPT class: accepts layer_configs for per-layer MLP mult and KV heads
- layer_configs: [(0, 2)] * 4 + [(2.0, 2)] * 3 + [(3.0, 4)] * 7

## Expected Results
- Target ~16.0MB total submission size
- Maximum depth (14L) may capture more complex patterns despite some layers being attn-only

## Actual Results
(to be recorded after training)
