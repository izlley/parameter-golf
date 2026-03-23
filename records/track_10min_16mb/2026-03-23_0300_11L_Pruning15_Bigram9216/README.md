# 11L + Pruning 15% + BigramHash 9216

## 목적
11L(val_bpb 1.1369, 17.1MB)을 16MB에 수용하기 위해 공격적 pruning(15%)과 BigramHash 축소(9216)를 적용.

## 변경 사항 (vs SOTA 10L)
- NUM_LAYERS: 10 → 11
- Pruning: 5% → 15%
- BigramHash: 10240 → 9216
- FP16_KEEP: blocks.8 → blocks.9

## 크기 추정
- 11L pruning 10%: 17.1MB
- Pruning 15% 추가 절약: ~400KB
- BigramHash 9216 절약: ~100KB
- 예상: ~16.6MB (아직 초과 가능)

## 예상 결과
- val_bpb: 1.140~1.145 (pruning 15%가 10% 대비 +0.003~0.005 bpb 악화)
- 크기: ~16.5MB (여전히 초과 가능)

## 실제 결과
(학습 후 기록 예정)
