# 10L + Label Smoothing

## 목적
Label smoothing(0.1)으로 소프트 타겟을 사용하여 양자화 후 일반화 성능 향상.

## 변경 사항 (vs SOTA)
- Label smoothing: 0.0 → 0.1 (F.cross_entropy에 적용)

## 예상 결과
- val_bpb: 1.140~1.143 (소폭 개선 또는 동등)
- 크기: ~15.8MB (변화 없음)

## 실제 결과
(학습 후 기록 예정)
