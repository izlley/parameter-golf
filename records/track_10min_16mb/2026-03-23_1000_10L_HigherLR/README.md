# 10L + Higher Learning Rate

## 목적
더 공격적인 학습률(0.025)과 더 긴 warmup(30 steps), 넓은 momentum warmup(0.80→0.99)으로 학습 탐색 강화.

## 변경 사항 (vs SOTA)
- matrix_lr: 0.02 → 0.025
- warmup_steps: 20 → 30
- muon_momentum_warmup_start: 0.85 → 0.80

## 예상 결과
- val_bpb: 1.140~1.144
- 크기: ~15.8MB (변화 없음)

## 실제 결과
(학습 후 기록 예정)
