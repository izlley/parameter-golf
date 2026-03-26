# Exp 118: 11L Early QAT (HPTuned_Fit v2)

## Purpose
Exp 116(HPTuned_Fit)에서 late_qat_threshold 방향 오류 수정. threshold=0.06은 scale<0.06일때 QAT 활성 → 매우 늦은 QAT. 0.25로 올려 진짜 earlier QAT 적용.

## Base
Exp 116 (11L HPTuned_Fit, val_bpb 1.1198 sliding_window, artifact 17,638,707B **초과**)

## Changes from Base
- **late_qat_threshold**: 0.06 → **0.25** (scale < 0.25일때 QAT 활성)
  - Exp 116: step 6145에서 QAT 활성 (scale=0.06, QAT 학습 ~178 steps)
  - 예상: ~step 4700에서 QAT 활성 (scale=0.25, QAT 학습 ~1600 steps)
  - 더 긴 QAT 학습으로 quantization-aware fine-tuning 충분히 수행 → 크기 감소

## Architecture (Exp 109/116 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4 layers [7,8,9,10]
- Partial RoPE 16/64
- BigramHash 8192, dim=128
- VE layers 9,10 (dim=128)
- EMA(0.998), LN Scale (1/sqrt(layer+1))

## Hyperparameters
- Muon: LR=0.028, momentum=0.99, WD=0.04, NS5
- AdamW: LR=0.035, beta1=0.9, beta2=0.95, WD=0.04
- Grad clip: 0.5
- Warmdown: 3000 iters
- SWA: every 50 steps (scale < 0.2)
- **Late QAT threshold: 0.25** (핵심 변경)

## Expected Results
- Target: < 1.1198 bpb (Exp 116과 유사 성능)
- Artifact: < 16,777,216 bytes (16MiB 이내 — Exp 116의 17.6MB 문제 해결)
- step_avg: ~95ms
