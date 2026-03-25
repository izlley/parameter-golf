# Exp 95: GatedAttn + Legal TTT

## Base
Exp 93 (GatedAttn PerHead, val_bpb 1.1198)

## Changes
- Add Legal Score-First TTT evaluation (from Exp 89)
- TTT_FREEZE_BLOCKS=0 (matches SOTA#2)
- TTT_LR=0.002, TTT_EPOCHS=3, TTT_CHUNK_TOKENS=32768

## Expected
Exp 93's 1.1198 + TTT -> ~1.118x or better
