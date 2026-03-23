# 11L Combined Slim (Encoder GQA 2 + MLP 2.75x)

## 목적
11L 모델에서 encoder 레이어를 구조적으로 슬림화: GQA 2 + MLP 2.75x 결합.
pruning은 10%로 유지하여 품질 손해 최소화.

## 변경 사항 (vs SOTA 10L)
- NUM_LAYERS: 10 → 11
- Encoder(5L) KV heads: 4 → 2
- Encoder(3L) MLP: 3.0x → 2.75x
- Pruning: 5% → 10% (기존 11L과 동일)
- FP16_KEEP: blocks.9.attn.c_k

## 크기 추정
- 11L pruning 10%: 17.1MB
- GQA 2 절약: ~-250KB
- MLP 2.75x 절약: ~-300KB
- 순 절약: ~-550KB → 16.55MB (아직 초과 가능, pruning 조정 필요할 수 있음)

## 실제 결과
(학습 후 기록 예정)
