# Exp 126: 11L CERWU (Compression-Aware Weight Regularization)

## Purpose
양자화-압축 공동 최적화: warmdown 동안 가중치가 int6 양자화 그리드에 가깝도록 유도하여 zstd 압축 효율 향상.

## Paper
"Compression with Entropy-Regularized Weight Updates" (May 2025)
- 양자화 후 가중치의 entropy를 줄여 더 잘 압축되게 함
- 같은 품질에서 20-40% 더 작은 artifact

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198, artifact 16,722,423B)

## Changes from Base
- **CERWU Loss**: warmdown 시작(scale<1.0) 후 양자화 잔차 페널티 추가
  - `cerwu_loss`: 가중치를 int6 그리드로 라운딩한 잔차의 MSE
  - `loss = loss + cerwu_alpha * cerwu_loss(model)`
  - cerwu_alpha: 0.001 (약한 정규화)
  - 가중치가 그리드에 스냅 → 양자화 후 값 분포 집중 → zstd 압축률 향상
- 학습 초반(scale=1.0)에서는 비활성 → 품질 영향 최소

## Architecture (Exp 93 동일)
- 11L, 512dim, 8H/4KV, GQA, LeakyReLU(0.5)², MLP 3x
- GatedAttn PerHead, XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15, Warmdown 3500

## Hyperparameters
- **cerwu_alpha: 0.001** (핵심 추가)
- 나머지: Exp 93 동일

## Expected Results
- Artifact: ~16.0~16.5MB (기존 16.72MB 대비 절감)
- Target: ≤ 1.1198 bpb (품질 유지)
- step_avg: ~96ms (+1ms, CERWU loss 계산)
