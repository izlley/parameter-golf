# Exp 91: SharedFFN 12L — Wider Dimension (dim=656)

## Base
Exp 84 v2 (SharedFFN 12L, dim=512, val_bpb 1.1856, size 9.8MB)

## Change
- MODEL_DIM 512 -> 656

## Rationale
v2 was only 9.8MB, wasting ~6MB of the 16MB budget. Wider dim increases per-layer capacity significantly.

## Estimates
- Estimated params: ~24.8M
- Estimated size: ~15.5MB

## Expected
~1.1500-1.1700 (improved from 1.1856 but SharedFFN inherently limits representation diversity)
