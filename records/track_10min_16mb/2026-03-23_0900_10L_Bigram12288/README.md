# 10L + BigramHash 12288

## 목적
BigramHash 버킷을 12288로 확대하여 더 많은 바이그램 패턴 포착. Pruning 7%로 크기 보정.

## 변경 사항 (vs SOTA)
- BigramHash: 10240 → 12288
- Pruning: 5% → 7%

## 예상 결과
- val_bpb: 1.140~1.142 (BigramHash 확대로 -0.001 bpb)
- 크기: ~15.9MB (BigramHash +200KB, pruning으로 상쇄)

## 실제 결과
(학습 후 기록 예정)
