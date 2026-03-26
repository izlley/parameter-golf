# Exp 123: 11L Differential Attention

## Purpose
Differential Transformer 논문 적용: 각 head를 2개 sub-head로 분할, attention score 차이로 noise 제거.

## Paper
"Differential Transformer" (ICLR 2025, Microsoft)
- 각 head를 2개 sub-attention으로 분할 (Q,K,V를 head_dim 반으로 split)
- 두 attention 결과의 차이(y1 - λ*y2)로 attention noise 제거
- λ: learnable per-head scalar (sigmoid로 0~1 범위)

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Base
- **Differential Attention**: CausalSelfAttention.forward 수정
  - Q,K,V를 head_dim 반으로 split → 2배 heads로 reshape
  - 단일 flash_attn_3_func 호출 (2*H heads, D/2 dim = 동일 FLOPS)
  - 출력: diff(y1-λ*y2) + sum(y1+y2) concat → 원래 차원 복원
  - `diff_lambda`: per-head learnable scalar (init=0.8, sigmoid≈0.69)
- 추가 파라미터: 8개 (num_heads × 1)
- GatedAttn과 결합 유지

## Architecture
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x
- **DiffAttn** (λ init=0.8) + GatedAttn PerHead
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15

## Expected Results
- Target: < 1.1198 bpb (-0.001~0.002)
- step_avg: ~95ms (동일 FLOPS, reshape만 추가)
- Artifact: ~16.72MB (동일)
