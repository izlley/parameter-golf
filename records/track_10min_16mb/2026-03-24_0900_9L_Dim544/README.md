# 9L x 544dim (Exp 61)

## 목적
10L x 512dim과 동일한 파라미터 예산을 9L x 544dim으로 재배분.
레이어를 1개만 줄이고 차원을 소폭 확대하여 보수적 접근.
head_dim = 544/8 = 68로 per-head 표현력 소폭 증가.

## 변경 사항 (vs SOTA 10L)
- num_layers: 10 -> 9
- model_dim: 512 -> 544 (head_dim 64->68)
- swa_start_frac: 0.50 -> 0.35
- 나머지 동일 (MLP 3x, KV 4, pruning 5%, bigram 10240)

## 예상 결과
- val_bpb: ~1.140~1.145
- 크기: ~16.0MB
- step_avg: ~85ms (레이어 1개 감소, 차원 소폭 증가로 상쇄)

## 실제 결과
(학습 후 기록 예정)
