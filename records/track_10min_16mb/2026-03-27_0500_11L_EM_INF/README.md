# Exp 127: 11L EM-INF (Entropy Minimization at Inference)

## Purpose
모델 출력 logits에 entropy minimization을 적용하여 예측 신뢰도 향상. 모델 가중치는 변경 없이 logit 벡터만 GD로 최적화.

## Paper
"The Unreasonable Effectiveness of Entropy Minimization" (NeurIPS 2025)
- Logits를 free parameter로 취급, GD로 entropy 최소화
- 모델 backward 불필요 — vocab_size(1024) 차원 벡터에 5 GD steps
- Cost-free로 시도 가능 (추가 학습 불필요)

## Base
Exp 115 (11L SupermaskTTT V3, val_bpb 1.11962742 sliding+TTT)

## Changes from V3
- **EM-INF 후처리**: sliding window 평가 시 logits에 entropy minimization 적용
  - 각 배치의 logits를 detach → float → 5 GD steps (SGD lr=0.05)
  - Entropy = -Σ p·log(p) 최소화 → 분포 sharpening
  - TTT 후 추가 후처리로 실행
- **별도 평가 패스**: 기존 sliding_window + TTT 결과는 그대로 유지
  - EM-INF 결과는 추가 metric으로 리포트
- 추가 시간: ~30-60초 (logit-level GD만, model forward 없음)

## Architecture (Exp 93/V3 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- SupermaskTTT V3 포함

## EM-INF Hyperparameters
- eminf_steps: 5
- eminf_lr: 0.05
- 적용 시점: TTT 후 sliding window 재평가

## Expected Results
- Target: < 1.1196 bpb (-0.0002~0.001)
- Artifact: ~16.70MB (동일, 모델 변경 없음)
- 추가 eval 시간: ~30-60초
