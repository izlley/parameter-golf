# Exp 184: 1-sqrt Cooldown Schedule

## 목적 (Purpose)
현재 linear cooldown을 `1-sqrt(progress)` 형태로 변경하여 base 모델 품질 개선.
Linear 대비 초반에 더 천천히 감소하여 "river valley" loss landscape에서 더 효율적으로 최적점 도달.

## 베이스
Exp 172 (11L MuonTTT SensitivityEWQ, TTT val_bpb 1.1227, 15.9MB)

## 변경 사항
- **LR Schedule 변경**: warmdown 구간에서 `lr * (1 - sqrt(progress))` 적용
  - 기존: `lr * (1 - progress)` (linear)
  - 변경: `lr * (1 - sqrt(progress))` (1-sqrt)
  - progress = (step - warmdown_start) / warmdown_iters
  - wallclock 기반 schedule도 동일하게 변경
- **근거**: "Understanding WSD Learning Rates: A River Valley Loss Landscape Perspective" (2024)
  - `1-sqrt`가 linear, cosine 대비 일관되게 우수
  - 초반에 LR이 느리게 감소 → 더 넓은 optima 탐색
  - 후반에 빠르게 감소 → sharp minimum 회피
- **다른 변경 없음**: SensitivityEWQ 등 모든 기존 기능 유지

## 예상 결과
- TTT val_bpb: ~1.1210-1.1225 (base 모델 -0.001~0.003 개선, EWQ 비용 동일)
- artifact size: ~15.9MB ✅ (변동 없음)
- step_avg: ~95ms (변동 없음)

## 실제 결과
(실행 후 기록)
