# Exp 124: 11L EWQ (Entropy-Weighted Quantization)

## Purpose
레이어별 mixed precision 양자화: 초기 레이어(덜 민감)는 int5, 후기 레이어(더 민감)는 int6 유지.

## Paper
"Entropy-Weighted Quantization" (Mar 2025)
- 레이어별 weight entropy 분석 → 민감도에 따라 precision 할당
- 덜 민감한 레이어의 낮은 precision은 regularization 효과

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window, artifact 16,722,423B)

## Changes from Base
- **EWQ Mixed Precision**: layers 0-3은 int5 (clip_range=15), layers 4-10은 int6 (clip_range=31)
  - int5: [-15, 15] → 5비트 효과, int6: [-31, 31] → 6비트 효과
  - 초기 4개 레이어: 임베딩에 가까워 feature extraction이 주 역할, 정밀도 덜 필요
  - 후기 7개 레이어: task-specific representation, 높은 정밀도 필요
- **QAT도 EWQ 반영**: early layer의 CastedLinear는 int5 QAT 적용
- 예상 크기 절감: layer당 ~200KB × 4 layers × (1 - 5/6) ≈ ~130KB 절감

## Architecture (Exp 93 동일 + EWQ)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead
- **Layers 0-3: int5**, Layers 4-10: int6
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10

## Expected Results
- Artifact: ~16.59MB (~130KB 절감)
- Target: ≤ 1.1198 bpb (품질 유지 또는 소폭 개선)
- step_avg: ~95ms (동일)
