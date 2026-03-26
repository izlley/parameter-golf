# Exp 106: 11L Mamba2 Last 2 Layers [9,10]

## Purpose
Test whether placing Mamba2 SSM blocks at the last 2 layers (9,10) improves over mid-placement (5,10). Prior results showed single Mamba at layer 10 was optimal; this tests if adding layer 9 provides complementary sequential modeling.

## Approach
- Based on Exp 101 (Mamba2 Mid [5,10], 1.1250 bpb)
- Key changes from Exp 101:
  - MAMBA_LAYERS: "5,10" → "9,10"
  - VE_LAYERS: "8,9" → "7,8" (shifted to avoid collision with Mamba layer 9)
- Architecture: 11L (9 Attn + 2 Mamba2), 512dim, 8 heads (4 KV), GatedAttn PerHead
- U-Net skip connections: 5 encoder (layers 0-4) + 6 decoder (layers 5-10)
- Mamba layers correctly excluded from RoPE and XSA

## Expected Results
- Consecutive Mamba layers at the end may enable deeper sequential feature extraction
- Risk: Exp 97 showed Mamba at layer 10 alone achieved 1.1217; adding more Mamba layers has consistently worsened results
- Target: < 1.1250 bpb (improve over Exp 101's 1.1250)
