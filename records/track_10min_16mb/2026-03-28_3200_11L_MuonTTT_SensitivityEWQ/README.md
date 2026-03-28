# Exp 172: MuonTTT + Sensitivity-Based EWQ

## 목적 (Purpose)
Exp 165 EWQ(layers 0-5 일괄 int5, +0.007 bpb)의 개선판.
Per-layer sensitivity 분석으로 민감도 낮은 레이어만 int5 적용하여 bpb 손실 최소화.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **Sensitivity Analysis**: 학습 완료 후 calibration forward pass 1회
  - 각 레이어의 weight를 int5로 양자화 → reconstruction error 측정
  - error가 낮은(=민감도 낮은) 레이어부터 int5 적용
  - 목표 크기(16MB)에 도달할 때까지 레이어 추가
- **Adaptive int5 배분**: 고정 0-5 대신, sensitivity 기반 선택
  - MLP weight와 Attention weight 별도 sensitivity 측정
  - 텐서 단위로 int5/int6 결정 가능
- **QAT**: sensitivity로 선정된 레이어만 int5 QAT 적용

## 예상 결과
- TTT val_bpb: ~1.1175-1.1200 (Exp 165의 +0.007 대비 +0.003 수준)
- artifact size: ~15.5-16.0MB (Exp 165와 비슷한 크기 절감)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
