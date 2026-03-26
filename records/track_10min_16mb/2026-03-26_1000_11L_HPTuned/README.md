# Exp 109: 11L HP Tuned (EMA/LR/Warmdown/QAT/Clip)

## Purpose
Exp 93 (GatedAttn PerHead, 1.1198)에서 SOTA(1.1194)까지 0.0004 bpb 격차. Hyperparameter 미세조정으로 격차를 메우는 것이 목표.

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Exp 93

| HP | Exp 93 | Exp 109 | 근거 |
|----|--------|---------|------|
| EMA decay | 0.997 | 0.998 | 후반 가중평균 강화 — 최근 체크포인트 비중 ↑ |
| Muon lr | 0.025 | 0.028 | 약간 공격적 학습 → 동일 스텝에서 더 많이 수렴 |
| Warmdown | 3500 | 3000 | 짧은 warmdown → 본 학습 시간 확보 |
| Late QAT | 0.15 | 0.12 | QAT 조기 시작 → 양자화 적응 시간 확보 |
| Grad clip | 0.3 | 0.5 | 클리핑 완화 → gradient 정보 더 보존 |

## Architecture (동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x
- GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.998) + U-Net skip

## Expected Results
- step_avg: ~95ms (동일)
- Target: < 1.1198 bpb (Exp 93 개선)
- Size: ~16.72MB (동일)
