# Exp 105: 11L SupermaskTTT with MLP + Attention Masking

## Purpose
Extend Supermask TTT from MLP-only to both MLP and Attention output channels. Test whether learning per-channel masks on attention outputs during test-time provides additional adaptation capacity.

## Approach
- Based on Exp 99 (SupermaskTTT MLP-only, 1.1209 bpb) which is based on Exp 93 (GatedAttn PerHead SOTA, 1.1198 bpb)
- Key changes from Exp 99:
  - mask_scores now includes BOTH MLP hidden masks (even indices) and Attn output masks (odd indices)
  - Attn output mask: per-channel sigmoid mask applied to CausalSelfAttention output
  - Both mask types initialized to 5.0 (sigmoid ≈ 0.993, near-identity start)
  - Total mask params: 11 * (MLP_hidden + model_dim) per chunk
- All model weights remain frozen during TTT; only masks are learned via SGD
- Monkey-patching pattern: factory functions to avoid Python closure bugs

## Expected Results
- Attention masking provides orthogonal adaptation to MLP masking
- Target: < 1.1200 bpb (improve over Exp 99's 1.1209)
- No size increase (masks are ephemeral, not saved in artifact)
