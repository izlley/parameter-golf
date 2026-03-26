# Exp 121: 11L NorMuon (Neuron-wise Normalized Muon)

## Purpose
NorMuon 논문 적용: Muon의 Newton-Schulz 후 neuron별 row norm 불균형을 해소하여 학습 효율 향상.

## Paper
"NorMuon: Making Muon more efficient and scalable" (Oct 2025)
- modded-nanogpt speedrun에서 World Record 달성 (-15 steps)
- 학습 효율 11% 향상

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Base
- **NorMuon**: Muon step에서 NS5 후 `F.normalize(g, dim=1)` 추가
  - 각 neuron(row)의 L2 norm을 1로 정규화
  - NS5 출력의 row norm variance 해소 → 모든 neuron에 동일한 effective LR
  - 한 줄 추가, 속도 영향 거의 없음 (~0.1ms)

## Architecture (Exp 93 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15, Warmdown 3500

## Hyperparameters (Exp 93 동일)
- Muon: LR=0.025, momentum=0.99, WD=0.04, NS5 + **NorMuon**
- AdamW: LR=0.035, beta1=0.9, beta2=0.95, WD=0.04

## Expected Results
- Target: < 1.1198 bpb (-0.001~0.005 개선)
- step_avg: ~95ms (거의 동일)
- Artifact: ~16.72MB (동일)
