# Exp 97: SharedFFN U-Net 13L

## Base
Exp 84 (SharedFFN 14L) + Exp 93 (GatedAttn)

## Architecture
- 13 layers, 512 dim, 8 heads, 4 KV heads
- Paired FFN sharing: [0,1][2,3][4,5][6,7][8,9][10,11] + layer 12 independent
- U-Net skip connections: [0,1]->[10,11], [2,3]->[8,9], [4,5]->[6,7]
- 7 unique FFNs (6 shared + 1 independent) -> saves ~6 FFNs worth of params
- Gated attention (per-head, bias=4.0)
- Bigram8192

## Size estimate
- 13 layers attention: ~10.2M params
- 7 unique FFNs: ~11M params
- Gated attention: ~53K params
- Bigram, embeddings, etc.: ~2.5M params
- Total: ~23.7M -> at int6+zstd ~ ~15MB (within budget)

## Expected
More depth (13 vs 11) with parameter sharing -> potential improvement
