# Exp 111: 11L SupermaskTTT V2 (Mask + Bias, Lower LR)

## Purpose
Exp 105 (SupermaskTTT_Attn, sliding 1.1200, TTT 1.1203)에서 TTT 효과를 증폭. 마스크 파라미터를 2배로 확대하고, TTT hyperparameter를 조정.

## Base
Exp 105 (11L SupermaskTTT_Attn, sliding 1.1200 / TTT 1.1203)

## Changes from Exp 105

### 1. Mask Parameter 확대 (22,528 → ~45,056)
- **기존**: MLP mask(11×1536) + Attn mask(11×512) = 22,528 params
- **추가**: MLP bias(11×1536) + Attn bias(11×512) = 22,528 params
- **총**: ~45,056 params (2배)
- MLP bias: hidden space에서의 additive shift (init=0)
- Attn bias: output space에서의 additive shift (init=0)
- mask는 multiplicative, bias는 additive → 상호보완

### 2. TTT Hyperparameter 조정
| HP | Exp 105 | Exp 111 | 근거 |
|----|---------|---------|------|
| TTT lr | 0.002 | 0.001 | 파라미터 2배 → lr 절반으로 과적합 방지 |
| TTT epochs | 3 | 2 | 에포크 감소 → 과적합 방지 + 속도 향상 |

## Architecture (학습 시 동일, TTT만 변경)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + U-Net skip

## Expected Results
- sliding_window: ~1.1200 (학습은 동일)
- TTT: < 1.1200 (bias 추가로 TTT 효과 증폭 기대)
- Target: < 1.1194 (SOTA 돌파)
- Size: ~16.73MB (학습 모델 동일)
