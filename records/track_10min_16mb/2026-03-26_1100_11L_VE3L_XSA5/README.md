# Exp 110: 11L VE 3 Layers + XSA 5 + GatedAttn Bias 3.0

## Purpose
Exp 93 (GatedAttn PerHead, 1.1198) 아키텍처 미세조정. Value Embedding 범위 확장, XSA 범위 확장, GatedAttn 초기 게이트 값 조정.

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Exp 93

| 변경 | Exp 93 | Exp 110 | 근거 |
|------|--------|---------|------|
| VE layers | 9,10 | 8,9,10 | VE 범위 확장 → 더 많은 레이어에 토큰 정체성 주입 |
| XSA last_n | 4 (layers 7-10) | 5 (layers 6-10) | XSA 범위 확장 → self-value projection 제거 확대 |
| GatedAttn bias | 4.0 | 3.0 | 초기 sigmoid(3.0)≈0.953 → 더 적극적 게이팅 |

## Architecture
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x
- GatedAttn PerHead (bias=3.0) ← 변경
- XSA last 5 ← 변경
- Partial RoPE 16/64
- BigramHash 8192
- VE layers 8,9,10 ← 변경
- EMA(0.997) + U-Net skip

## Expected Results
- step_avg: ~95ms (VE 1개 추가 영향 무시)
- Target: < 1.1198 bpb
- Size: ~16.72MB (VE layer scale 1개 추가 = 4bytes, 무시)
