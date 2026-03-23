# 11L Prune13% Bigram9216 (Exp 57)

## 목적
Exp 45 (11L, 16.08MB, +76KB 초과)에서 bigram만 축소하여 16MB 이내 달성.

## 변경 사항 (vs Exp 45)
- bigram_vocab_size: 10240 → 9216 (~80KB 절약)
- 나머지 동일: pruning 13%, encoder_mlp_layers 5, encoder_kv_heads 2

## 예상 결과
- val_bpb: ~1.143 (bigram 축소로 +0.001)
- 크기: ~16.00MB (76KB - 80KB 절약 = 예산 내)

## 실제 결과
(학습 후 기록 예정)
