# Exp 122: 11L WSD Schedule (Warmup-Stable-Decay)

## Purpose
WSD LR Schedule 논문 적용: stable phase를 최대화하고 decay를 짧고 급격하게.

## Paper
"Optimal Learning-Rate Schedules" (Feb 2026), ICML 2025
- Cosine보다 체계적으로 우수
- Longer high-LR phase → more learning, shorter sharp decay → better convergence

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Base
- **warmdown_iters**: 3500 → **2000** (decay 구간 단축)
  - Exp 93: stable ~44%, decay ~56% (너무 긴 decay)
  - Exp 122: stable ~68%, decay ~32% (WSD 스타일)
  - 더 긴 full-LR 구간에서 더 많은 학습 → 짧은 sharp decay로 수렴

## LR Schedule 비교
| Phase | Exp 93 (3500) | Exp 122 (2000) |
|-------|---------------|----------------|
| Stable (LR=1.0) | ~step 2800 (44%) | ~step 4300 (68%) |
| Decay (linear→0) | 3500 steps (56%) | 2000 steps (32%) |

## Architecture (Exp 93 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15

## Hyperparameters
- **Warmdown: 2000 iters** (핵심 변경)
- Muon: LR=0.025, momentum=0.99, WD=0.04, NS5
- AdamW: LR=0.035, beta1=0.9, beta2=0.95, WD=0.04

## Expected Results
- Target: < 1.1198 bpb (-0.001~0.003 개선)
- step_avg: ~95ms (동일)
- Artifact: ~16.72MB (동일)
