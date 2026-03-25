# Exp 90: HybridMamba2 -- Mamba at Last Layer

## Base
Exp 83 v2 (HybridMamba2, mamba_layers=[0], val_bpb 1.1219)

## Change
MAMBA_LAYERS default: "0" -> "10" (last layer of 11 layers, index 10)

## Rationale
The first layer processes raw embeddings which are less semantic in nature. The last layer produces the final logits and may benefit more from sequential modeling. Mamba's strength is capturing long-range sequential dependencies, which matter more for the final prediction layer than for initial embedding processing.

## Expected
~1.1190-1.1220
