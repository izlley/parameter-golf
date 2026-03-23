# 9L x 560dim (Exp 62)

## 목적
10L x 512dim과 동일한 파라미터 예산을 9L x 560dim으로 재배분.
Exp 61(544dim)보다 더 넓은 차원으로 per-layer 표현력 추가 확대.
head_dim = 560/8 = 70으로 per-head 표현력 증가.

## 변경 사항 (vs SOTA 10L)
- num_layers: 10 -> 9
- model_dim: 512 -> 560 (head_dim 64->70)
- swa_start_frac: 0.50 -> 0.35
- prune_quantile: 0.05 -> 0.08 (크기 제어를 위해 pruning 강화)
- 나머지 동일 (MLP 3x, KV 4, bigram 10240)

## 예상 결과
- val_bpb: ~1.140~1.145
- 크기: ~16.0MB (pruning 8%로 크기 제어)
- step_avg: ~85ms (레이어 1개 감소, 차원 증가로 상쇄)

## 실제 결과
(학습 후 기록 예정)
