# Exp 115: 11L SupermaskTTT V3 (Higher LR, Warmup Schedule)

## Purpose
Exp 111 (SupermaskTTT V2, sliding 1.1200, TTT 1.1202)에서 TTT가 sliding_window보다 나쁜 결과. 원인 분석: mask가 초반에 너무 느리게 수렴하여 초기 chunk들의 높은 loss가 누적 평균을 끌어올림. TTT lr 5x 증가, epoch 증가, lr schedule 개선, mask 초기값 변경으로 빠른 수렴 유도.

## Base
Exp 111 (11L SupermaskTTT V2, sliding 1.1200 / TTT 1.1202)

## Changes from Exp 111

### 1. TTT Hyperparameter 변경
| HP | Exp 111 | Exp 115 | 근거 |
|----|---------|---------|------|
| TTT lr | 0.001 | 0.005 | 5x 증가 → 초반 mask 수렴 가속 |
| TTT epochs | 2 | 3 | chunk당 48 grad steps (vs 32) → 더 충분한 학습 |

### 2. LR Schedule 변경
- **기존**: pure cosine `lr * 0.5 * (1 + cos(pi * ci / N))` → chunk 0에서 최대, 이후 계속 감소
- **변경**: warmup 5% + constant 65% + cosine decay 30%
  - 0~5%: linear warmup (0 → lr)
  - 5~70%: constant (lr 유지)
  - 70~100%: cosine decay (lr → 0)
- 근거: 기존 schedule은 chunk 946(50%)에서 이미 lr이 절반. mask가 충분히 학습되기 전에 lr이 줄어듦

### 3. Mask 초기값 변경
- **기존**: init=5.0 → sigmoid(5.0)≈0.993 (mask 거의 1, 사실상 미적용)
- **변경**: init=3.0 → sigmoid(3.0)≈0.953 (더 적극적 초기 masking)
- 근거: 0.993에서 시작하면 gradient가 매우 작아 학습 느림. 0.953에서 시작하면 sigmoid gradient가 더 크고 mask 효과도 즉시 발생

## Architecture (학습 시 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)^2, MLP 3x, GatedAttn PerHead
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + U-Net skip

## TTT Mask Structure (동일)
- Per block: mlp_mask(1536) + mlp_bias(1536) + attn_mask(512) + attn_bias(512)
- Total: 45,056 params

## Expected Results
- sliding_window: ~1.1200 (학습 동일)
- TTT: < 1.1200 (빠른 mask 수렴으로 초반 chunk 손실 감소)
- TTT 시간: ~600s (epochs 3 → Exp 111 대비 ~50% 증가)
- Size: ~16.65MB (학습 모델 동일)
