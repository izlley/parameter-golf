# 16L AttnOnly (Exp 54)

## Purpose
Test extreme depth with a clean encoder-decoder split using attention-only layers for the encoder half and full layers for the decoder half, fitting 16 layers within 16MB.

## Approach
- 16 total layers with 2 layer types:
  - 8 AttnOnly encoder layers (MLP mult=0, KV heads=2) -- pure attention feature mixing
  - 8 Full decoder layers (MLP mult=3.0x, KV heads=4) -- standard full layers
- SWA start fraction lowered to 0.35 (from 0.50) to allow more averaging time
- Magnitude pruning at 5% quantile
- All other settings inherited from 10L SWA050 baseline

## Expected Results
- Target model size: ~16.1MB
- Clean separation: encoder layers build representations cheaply, decoder layers do the heavy lifting
- 8 attention-only layers add significant depth with minimal parameter cost
