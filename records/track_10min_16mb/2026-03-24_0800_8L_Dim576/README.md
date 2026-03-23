# 8L x 576dim (Exp 60)

## 목적
10L x 512dim과 동일한 파라미터 예산을 8L x 576dim으로 재배분.
더 적은 레이어 x 더 넓은 차원으로 per-layer 표현력 증가.
레이어 감소로 step_avg 유지 -> 학습 스텝 손실 없이 표현력 확대.

## 변경 사항 (vs SOTA 10L)
- num_layers: 10 -> 8
- model_dim: 512 -> 576 (head_dim 64->72)
- swa_start_frac: 0.50 -> 0.35
- 나머지 동일 (MLP 3x, KV 4, pruning 5%, bigram 10240)

## 예상 결과
- val_bpb: ~1.140~1.145
- 크기: ~16.0MB
- step_avg: ~85ms (10L과 유사, 레이어 감소가 차원 증가 상쇄)

## 실제 결과
(학습 후 기록 예정)
