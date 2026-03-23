# 13L Hetero Mini (Exp 51)

## Purpose
Test heterogeneous (mixed) layer architecture to fit more layers within 16MB.
7 mini encoder layers (1.5x MLP, KV 1) + 6 full decoder layers (3.0x MLP, KV 4).

## Approach
- Use smaller MLP multiplier and fewer KV heads in early encoder layers to reduce parameter count
- This frees budget for more total layers (13 vs 10), potentially improving representation depth
- Earlier SWA start (0.35) to average over more checkpoints with the deeper model
- Lower prune quantile (0.04) since the model is already compact

## Changes (vs base 10L SWA050)
- num_layers: 10 -> 13
- swa_start_frac: 0.50 -> 0.35
- prune_quantile: 0.04 (was hardcoded 0.05)
- Block class: supports mlp_mult=0 (attn-only layers)
- GPT class: accepts layer_configs for per-layer MLP mult and KV heads
- layer_configs: [(1.5, 1)] * 7 + [(3.0, 4)] * 6

## Expected Results
- Target ~16.0MB total submission size
- More layers may improve loss despite smaller per-layer capacity in encoder

## Actual Results
(to be recorded after training)
