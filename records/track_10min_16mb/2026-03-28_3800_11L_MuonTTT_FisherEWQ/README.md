# Exp 178: MuonTTT + Fisher-Informed EWQ

## 목적 (Purpose)
Reconstruction MSE 대신 **Fisher Information (gradient 기반)** sensitivity를 사용.
MSE는 "weight 공간에서의 오차"만 보지만, Fisher는 "loss에 미치는 실제 영향"을 측정.
gradient가 큰 weight는 양자화 오차에 민감 → int6 유지, gradient가 작은 weight는 int5 허용.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **Fisher Sensitivity**: 학습 완료(EMA 적용) 후, validation 데이터 1 batch로 forward+backward
  - 각 파라미터의 `grad.pow(2)` 수집 (Fisher diagonal 근사)
  - 양자화 오차 `delta = param - dequantize(quantize_int5(param))`
  - Fisher-weighted impact: `(grad^2 * delta^2).sum()` per tensor
- **Tensor Selection**: Fisher impact가 낮은 텐서부터 int5 배정
  - 목표: ~2.0MB 절감까지
- **QAT**: heuristic으로 layer 0-5에 int5 QAT 적용
- **오버헤드**: calibration forward+backward 1회 (~0.5초) — 학습 시간 영향 무시 가능

## 예상 결과
- TTT val_bpb: ~1.1175-1.1195 (이론상 가장 정확한 sensitivity, Exp 172/176 대비 개선)
- artifact size: ~15.8-16.0MB (Exp 165와 유사)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
