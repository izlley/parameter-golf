# 10L + Temperature Optimization

## 목적
모델이 over/under-confident할 수 있으므로, eval 시 logits를 temperature T로 나누어 최적 T를 탐색.
T < 1.0은 더 confident한 예측, T > 1.0은 더 soft한 예측.

## 변경 사항 (vs SOTA)
- 학습: 완전 동일
- 평가: T=[0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08] grid search

## 예상 결과
- 최적 T ≈ 0.96~1.02 범위
- val_bpb: 1.141~1.143 (미세 개선 또는 동등)
- 평가 시간: ~9배 (8 temperatures + 기본), but 10분 내 가능

## 실제 결과
(학습 후 기록 예정)
