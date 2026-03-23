# 11L MLP6 Prune12% Bigram9216 (Exp 59)

## 목적
Exp 45 대비 MLP를 전체 6개 encoder layer에 적용하고, pruning을 줄이고 bigram을 축소하는 트레이드오프 실험.
More MLP reduction, less pruning → potentially better quality.

## 변경 사항 (vs Exp 45)
- encoder_mlp_layers: 5 → 6 (전체 encoder layer에 MLP 2.75x 적용)
- pruning quantile: 0.13 → 0.12 (pruning 감소)
- bigram_vocab_size: 10240 → 9216 (~80KB 절약)

## 예상 결과
- val_bpb: ~1.143 (MLP 확대 + pruning 감소로 품질 유지/개선 기대)
- 크기: ~16.00MB (MLP 축소 + bigram 축소로 예산 내)

## 실제 결과
(학습 후 기록 예정)
