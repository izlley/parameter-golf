# Exp 104: 12L Frozen SharedFFN (SupSup-style) hidden=6144

## Purpose
Test whether a single wide frozen MLP (hidden=6144) with per-layer binary masks (STE) can outperform per-layer MLPs. Frozen weights eliminate gradient interference between layers sharing the same weight matrix.

## Approach
- Based on Exp 100 (13L SupermaskFFN, 1.1276 bpb)
- Key changes from Exp 100:
  - NUM_LAYERS: 13 → 12
  - MLP_MULT: 4.0 → 12.0 (hidden=6144 for dim=512)
  - VE_LAYERS: "11,12" → "10,11"
  - **SharedFFN fc/proj weights are FROZEN** (requires_grad=False, orthogonal init)
  - Only per-layer mask_scores (12 x 6144 = 73,728 params) are trainable
- Architecture: 12L, 512dim, 8 heads (4 KV), GatedAttn PerHead
- Frozen MLP weights serve as a random feature basis; binary masks select per-layer subnetworks

## Expected Results
- Wider hidden (6144 vs 2048) provides richer random feature space for mask selection
- Frozen weights prevent gradient interference → cleaner per-layer specialization
- Target: < 1.1250 bpb (improve over Exp 100's 1.1276)
- Size: ~15.5MB (frozen weights + masks fit within 16MiB budget)
