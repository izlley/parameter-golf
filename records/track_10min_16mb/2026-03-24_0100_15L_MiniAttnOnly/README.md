# 15L Mini+AttnOnly (Exp 53)

## Purpose
Test extreme depth with heterogeneous (mixed) layer architectures to fit more layers within 16MB by using lightweight attention-only and mini layers.

## Approach
- 15 total layers with 3 layer types:
  - 5 AttnOnly layers (MLP mult=0, KV heads=2) -- no MLP, minimal attention
  - 3 Mini layers (MLP mult=1.5x, KV heads=1) -- reduced MLP and single KV head
  - 7 Full layers (MLP mult=3.0x, KV heads=4) -- standard full layers
- SWA start fraction lowered to 0.35 (from 0.50) to allow more averaging time
- Magnitude pruning at 5% quantile
- All other settings inherited from 10L SWA050 baseline

## Expected Results
- Target model size: ~16.0MB
- More representational depth from 15 layers while staying within budget
- AttnOnly layers add cheap feature mixing; mini layers add lightweight nonlinearity
