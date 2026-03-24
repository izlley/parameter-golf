# 10L SWA-Last Blend (Exp 67)

## 목적
SWA 평균 모델과 마지막 체크포인트를 고정 비율로 블렌딩.
Exp 40에서 SWA + last logit 앙상블이 val_bpb 1.1365로 best-ever 기록.
그러나 logit 앙상블은 2개 모델이 필요해 제출 불가 → 가중치 공간에서 블렌딩하면 단일 모델로 유사한 효과.

## 변경 사항 (vs SOTA 10L)
- 학습 종료 후 last state dict 저장
- SWA 평균 계산
- 블렌딩: `final = swa_blend_ratio × SWA + (1 - swa_blend_ratio) × last`
- `swa_blend_ratio=0.6` (SWA 60%, last 40%)
- 블렌딩된 단일 모델을 양자화/압축

## 근거
- SWA는 loss landscape의 넓은 영역을 평균 → 일반화 향상이나 과도한 평활화 위험
- Last checkpoint는 학습 후반의 최적 지점 → 더 날카로운 최적해
- 두 모델의 가중치 공간 보간 → SWA의 안정성 + last의 첨예함 결합
- Exp 40 결과(logit 앙상블 1.1365)가 이 전략의 이론적 근거

## 예상 결과
- val_bpb: ~1.139~1.142
- 크기: ~15.9MB (SOTA와 동일)
- 위험: 낮음 (가중치 보간은 안전한 연산)

## 실제 결과
(학습 후 기록 예정)
