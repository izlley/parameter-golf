# Exp 185: OmniQuant Learnable Weight Clipping

## 목적 (Purpose)
현재 5개 percentile 후보 grid search 방식을 gradient-based 최적화로 교체.
Per-row clipping bound를 STE(Straight-Through Estimator) 기반으로 학습하여 최적 양자화 scale 탐색.

## 베이스
Exp 172 (11L MuonTTT SensitivityEWQ, TTT val_bpb 1.1227, 15.9MB)

## 변경 사항
- **Learnable Weight Clipping (LWC)**: `quantize_int6_per_row`와 `quantize_int5_per_row` 교체
  - Per-row learnable log-scale 파라미터: `s = row_amax * exp(log_s) / clip_range`
  - STE: forward는 round 사용, backward는 round 무시하여 gradient 전달
  - Adam optimizer (lr=0.01), 200 steps per tensor
  - MSE(original - reconstructed) 최소화
  - Warm start: log_s=0 (row_amax/clip_range 초기값, percentile search의 1.0 후보와 동일)
- **근거**: OmniQuant (ICLR 2024 Spotlight) — LWC가 fixed percentile보다 항상 우수
  - Grid search는 5개 후보만 탐색, LWC는 continuous space에서 최적화
  - 특히 outlier 분포가 비균일한 row에서 큰 효과
- **기존 SensitivityEWQ 유지**: layer별 int5/int6 배분은 동일

## 예상 결과
- TTT val_bpb: ~1.1215-1.1225 (양자화 에러 ~10-20% 감소, -0.001~0.002 bpb)
- artifact size: ~15.9MB ✅ (scale 형태 동일, 값만 최적화)
- step_avg: ~95ms (변동 없음)
- 양자화 시간: tensor당 200 Adam steps → 전체 ~10-20초 추가 (acceptable)

## 실제 결과
(실행 후 기록)
