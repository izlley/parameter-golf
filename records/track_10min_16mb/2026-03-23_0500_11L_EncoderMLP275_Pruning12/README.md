# 11L + Encoder MLP 2.75x + Pruning 12%

## 목적
11L 모델에서 초반 3개 레이어의 MLP를 3.0x → 2.75x로 축소하여 크기 절약.
초반 레이어는 표현력 요구가 덜하므로 작은 MLP로도 충분하다는 가설.

## 변경 사항 (vs SOTA 10L)
- NUM_LAYERS: 10 → 11
- Encoder 3L MLP: 3.0x → 2.75x (hidden 1536→1408)
- Pruning: 5% → 12%
- FP16_KEEP: blocks.9.attn.c_k

## 크기 추정
- 11L pruning 10%: 17.1MB
- MLP 축소 절약: 3 × (512×128×2) × ~0.4 ≈ -300KB
- Pruning 12% 추가: ~-200KB
- 예상: ~16.6MB

## 실제 결과
(학습 후 기록 예정)
