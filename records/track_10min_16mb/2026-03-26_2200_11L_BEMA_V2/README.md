# Exp 120: 11L BEMA V2 (Higher Decay + Effective Correction)

## Purpose
Exp 112(BEMA)에서 bias correction이 사실상 무효한 문제 수정.
decay=0.997, 6298 steps → 0.997^6298 ≈ 0 → correction=1.000000 (보정 없음).

## Base
Exp 112 (11L BEMA, val_bpb 1.1208 sliding_window)

## 문제 분석
| decay | steps | decay^steps | correction |
|-------|-------|-------------|------------|
| 0.997 | 6298 | ≈ 5.8e-9 | 1.000000 (무효) |
| 0.999 | 6298 | ≈ 0.002 | 1.002 (미미) |
| 0.9995 | 6298 | ≈ 0.042 | **1.044** (유효) |
| 0.9999 | 6298 | ≈ 0.532 | **2.135** (과도) |

## Changes from Exp 112
- **ema_decay**: 0.997 → **0.9995**
  - 0.9995^6298 ≈ 0.042 → correction ≈ 1.044
  - 초기 EMA 편향을 ~4.4% 보정하여 모든 step의 기여도 균등화
  - EMA가 초기 가중치에 편향되는 문제 해결
- 더 높은 decay = 더 넓은 가중치 평균 → 일반화 향상 기대

## Architecture (Exp 93 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.9995), LN Scale (1/sqrt(layer+1))
- Late QAT threshold 0.15, Warmdown 3500

## Hyperparameters
- Muon: LR=0.025, momentum=0.99, WD=0.04, NS5
- AdamW: LR=0.035, beta1=0.9, beta2=0.95, WD=0.04
- Embed LR: 0.035 (tied), init_std=0.005
- Batch: 786,432 tokens, seq_len=2048
- Grad clip: 0.3
- SWA: every 50 steps (scale < 0.2)

## Expected Results
- Target: < 1.1208 bpb (Exp 112 개선)
- 유효한 bias correction으로 EMA 초기 편향 해소
- Artifact: ~16.72MB (Exp 93과 동일)
- step_avg: ~95ms
