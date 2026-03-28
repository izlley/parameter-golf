# Exp 175: MuonTTT + EWQ Extended (int5 layers 0-7)

## 목적 (Purpose)
Exp 165 EWQ(int5 layers 0-5, 16.1MB — 98KB 초과)의 int5 범위를 확대하여 16MB 통과.
Layers 0-7을 int5로 양자화, layers 8-10만 int6 유지.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **EWQ Extended**: int5 적용 범위 확대
  - Exp 165: layers 0-5 (6개) → int5
  - 본 실험: layers 0-7 (8개) → int5, layers 8-10 (3개)만 int6
- **추가 절감 예상**: 2개 레이어 추가 int5 → ~0.5MB 추가 절감
- **QAT**: int5 대상 레이어에 clip_range=15 적용

## 예상 결과
- TTT val_bpb: ~1.1240-1.1260 (Exp 165 +0.007 대비 +0.009-0.010)
- artifact size: ~15.5-15.8MB ✅ (Exp 165 대비 ~0.3-0.5MB 추가 절감)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
