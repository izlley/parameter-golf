# Exp 113: 11L Progressive Attention Looping

## Purpose
Exp 93 (GatedAttn PerHead, 1.1198) 아키텍처에서 학습 후반에 중간 layer를 2회 실행하여 effective depth를 증가. Sparse Growing Transformer 논문에서 영감.

## Paper
[Sparse Growing Transformer](https://arxiv.org/abs/2603.23998)

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Exp 93

### 1. Progressive Attention Looping
- **학습 후반 20%** (lr_scale < 0.2)에서 layer 4,5 (중간 layers)를 2-pass로 실행
- 동일 weight 재사용 → 추가 파라미터 없음
- Effective depth: 11L → 13L (후반에만)
- 구현: `GPT.forward`에서 조건부 layer 반복
- 학습 초반에는 비활성 → 기본 학습에 영향 없음

### 2. wandb logging 추가
- final eval metrics (roundtrip, sliding_window, s64) wandb.log 추가

## Architecture
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + U-Net skip
- **Progressive Looping**: layers 4,5 2x (학습 후반 20%)

## Expected Results
- step_avg: ~95ms (초반) → ~105ms (후반, layer 2개 추가 실행)
- Target: < 1.1198 bpb
- Size: ~16.72MB (동일, weight sharing)
