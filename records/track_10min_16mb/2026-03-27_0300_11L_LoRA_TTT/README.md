# Exp 125: 11L LoRA TTT (Rank-1 LoRA Adapters)

## Purpose
SupermaskTTT의 sigmoid channel mask를 rank-1 LoRA adapter로 교체. Mask는 스케일링만 가능하지만 LoRA는 additive update로 더 높은 표현력.

## Paper
"Test-Time Learning for LLMs" (ICML 2025) + "In-Place TTT" (ICLR 2026)
- LoRA adapter는 mask보다 더 expressive한 적응 가능
- 원본 가중치 동결 + adapter만 학습 → 안정적

## Base
Exp 115 (11L SupermaskTTT V3, val_bpb 1.11962742 sliding+TTT)

## Changes from V3
- **Supermask → LoRA rank-1**: per block MLP.proj와 Attn.proj에 rank-1 LoRA
  - MLP: A=[hidden, 1], B=[1, dim] → hidden×1 + 1×dim = ~2048 params/block
  - Attn: A=[dim, 1], B=[1, dim] → 512×1 + 1×512 = 1024 params/block
  - Total: ~3072 params/block × 11 blocks = ~33K params (V3 mask: ~45K)
  - Zero-init (identity at start)
- **ttt_lr**: 0.005 → **0.001** (LoRA는 더 expressive, 낮은 LR 필요)
- additive update: `out += (h @ A) @ B`

## Architecture (Exp 93 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15

## TTT Hyperparameters
- ttt_lr: **0.001** (V3: 0.005)
- ttt_epochs: 3, ttt_chunk_tokens: 32768
- LoRA rank: 1
- LR schedule: warmup 5% + constant 65% + cosine decay 30%

## Expected Results
- Target: < 1.1196 bpb (V3 대비 개선)
- LoRA의 additive update가 mask의 multiplicative scaling보다 유연
- Artifact: ~16.70MB (동일)
