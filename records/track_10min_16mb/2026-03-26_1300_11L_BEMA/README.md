# Exp 112: 11L BEMA (Bias-Corrected EMA)

## Purpose
Exp 93 (GatedAttn PerHead, 1.1198)의 EMA에 bias correction 적용. 표준 EMA는 초기 step에서 bias가 있어 최종 가중 평균이 부정확. Adam의 bias correction과 동일한 원리로 보정.

## Paper
[Bias-Corrected Exponential Moving Average](https://arxiv.org/abs/2508.00180)

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Exp 93

### 1. BEMA (Bias-Corrected EMA)
- **기존**: `ema = decay * ema + (1-decay) * param` (bias 있음)
- **변경**: 동일 업데이트 후, 최종 적용 시 `corrected = ema / (1 - decay^step)` 보정
- decay=0.997, ~6300 steps 학습 시 초기 ~333 steps에서 bias 존재
- 최종 모델 저장 시 corrected EMA 사용
- 구현: EMA step counter 추가 + 최종 적용 시 보정

### 2. wandb logging 추가
- final eval metrics (roundtrip, sliding_window, s64) wandb.log 추가

## Architecture (동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + BEMA correction + U-Net skip

## Expected Results
- step_avg: ~95ms (bias correction은 최종 적용 시에만 연산, 학습 중 오버헤드 0)
- Target: < 1.1198 bpb
- Size: ~16.72MB (동일)
