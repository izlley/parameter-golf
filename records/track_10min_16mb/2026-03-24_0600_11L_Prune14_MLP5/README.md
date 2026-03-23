# 11L Prune14% MLP5 (Exp 58)

## 목적
Exp 45 (11L, 16.08MB, +76KB 초과)에서 pruning을 13% → 14%로 강화하여 16MB 이내 달성.

## 변경 사항 (vs Exp 45)
- pruning quantile: 0.13 → 0.14 (~80KB 절약)
- 나머지 동일: bigram_vocab_size 10240, encoder_mlp_layers 5, encoder_kv_heads 2

## 예상 결과
- val_bpb: ~1.143 (pruning 1%p 증가로 미미한 영향)
- 크기: ~16.00MB (76KB - 80KB 절약 = 예산 내)

## 실제 결과
(학습 후 기록 예정)
