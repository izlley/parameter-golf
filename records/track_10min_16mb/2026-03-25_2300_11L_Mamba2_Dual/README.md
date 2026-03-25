# Exp 98: Mamba2 Dual Layers (0 + 10)

## Base
Exp 90 (HybridMamba2 LastLayer, val_bpb 1.1208)

## Changes
- MAMBA_LAYERS: "10" -> "0,10"
- Layer 0 (first): captures local/sequential patterns from raw embeddings
- Layer 10 (last): captures long-range sequential patterns for final prediction
- Layers 1-9: attention (complex feature extraction)

## Rationale
"Mamba sandwich" architecture. Layer 0 handles low-level sequential patterns
where Mamba excels (raw embeddings are less semantic). Layer 10 handles high-level
sequential dependencies for prediction. Attention layers in between handle complex
relational reasoning.

Exp 83 (Mamba layer 0) = 1.1241, Exp 90 (Mamba layer 10) = 1.1208.
Both showed Mamba adds value at different positions -> combining may be complementary.

## Expected
~1.118x-1.120x
