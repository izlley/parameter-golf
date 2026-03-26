# Exp 114: 11L Online Self-Distillation (EMA Teacher)

## Purpose
Exp 93 (GatedAttn PerHead, 1.1198)에서 이미 유지 중인 EMA 모델을 teacher로 활용하여 student(학습 모델)에 KL divergence loss 추가. Self-Distillation for MTP 논문에서 영감.

## Paper
[Self-Distillation for Multi-Token Prediction](https://arxiv.org/abs/2603.23911)

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Exp 93

### 1. Online Self-Distillation
- **학습 후반 50%** (lr_scale < 0.5)에서 EMA teacher의 logits과 student logits 간 KL loss 추가
- `loss = ce_loss + alpha * T² * KL(student || teacher)`
- alpha=0.1, T=2.0 (temperature)
- EMA 모델은 이미 메모리에 있으므로 추가 메모리 최소
- forward_logits 메서드 활용하여 teacher inference
- teacher forward는 `torch.no_grad()` 하에서 실행

### 2. wandb logging 추가
- final eval metrics (roundtrip, sliding_window, s64) wandb.log 추가

## Architecture (동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + U-Net skip

## Expected Results
- step_avg: ~95ms (초반) → ~115ms (후반, teacher forward 추가)
- Target: < 1.1198 bpb
- Size: ~16.72MB (동일)
