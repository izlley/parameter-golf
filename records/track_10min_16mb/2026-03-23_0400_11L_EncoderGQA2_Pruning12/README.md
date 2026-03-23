# 11L + Encoder GQA 2 + Pruning 12%

## 목적
11L 모델에서 encoder 레이어(5개)의 KV heads를 4→2로 줄여 크기 절약.
초반 레이어는 local 패턴 위주라 적은 KV heads로도 충분하다는 가설.

## 변경 사항 (vs SOTA 10L)
- NUM_LAYERS: 10 → 11
- Encoder(5L) KV heads: 4 → 2
- Pruning: 5% → 12%
- FP16_KEEP: blocks.9.attn.c_k

## 크기 추정
- 11L pruning 10%: 17.1MB
- Encoder GQA 2 절약: 5 × (512×128×2) × ~0.4 ≈ -250KB
- Pruning 12% 추가: ~-200KB
- 예상: ~16.6MB

## 실제 결과
(학습 후 기록 예정)
