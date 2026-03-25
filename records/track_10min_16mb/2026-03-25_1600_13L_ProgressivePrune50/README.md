# Exp 92a: Progressive Pruning at 50% Wallclock (13L->11L)

## Base
Exp 87 v2 (Progressive Pruning at 60%, val_bpb 1.1687)

## Change
- PRUNE_AT_FRAC: 0.60 -> 0.50 (prune at 5min instead of 6min, giving 5min recovery vs 4min)

## Rationale
Exp 87 v2 post_ema was 1.1468 but int6_SW was 1.1687. More recovery time after pruning should allow better convergence and improved quantized performance.

## Expected
~1.1550-1.1650
