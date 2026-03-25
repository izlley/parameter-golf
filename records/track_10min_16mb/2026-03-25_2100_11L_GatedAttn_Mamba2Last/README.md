# Exp 96: GatedAttn + Mamba2 LastLayer

## Base
Exp 90 (HybridMamba2 LastLayer, val_bpb 1.1208) + Exp 93 (GatedAttn, val_bpb 1.1198)

## Changes
- Added per-head gated attention (bias=4.0) from Exp 93 to attention layers
- Bigram8192 (from Exp 93)
- Last layer (10) uses Mamba2 (from Exp 90)

## Expected
Combining two best techniques: ~1.117x-1.119x
