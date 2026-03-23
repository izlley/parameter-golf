# 10L + More Regularization

## 목적
더 강한 정규화(WD 0.06)와 긴 warmdown(4000), 더 많은 SWA 체크포인트(start 0.30)로 양자화 내성 강화.

## 변경 사항 (vs SOTA)
- Weight decay: 0.04 → 0.06
- Warmdown iters: 3500 → 4000
- SWA start_frac: 0.35 → 0.30

## 예상 결과
- val_bpb: 1.140~1.143
- 크기: ~15.8MB (변화 없음)

## 실제 결과
(학습 후 기록 예정)
