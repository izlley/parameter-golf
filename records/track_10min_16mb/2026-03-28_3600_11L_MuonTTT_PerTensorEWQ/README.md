# Exp 176: MuonTTT + Per-Tensor Sensitivity EWQ

## 목적 (Purpose)
Exp 172(Layer-level Sensitivity EWQ)를 **텐서 단위**로 세분화.
같은 레이어 내에서도 MLP weight, Attention QKV, Output proj 등 텐서별 민감도가 다름.
텐서 단위로 int5/int6를 결정하여 같은 크기 절감에서 bpb 손실 최소화.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **Per-Tensor Sensitivity**: 학습 완료 후 각 2D weight 텐서에 대해
  - MSE(int5 quantize → dequantize) vs MSE(int6 quantize → dequantize) 비교
  - sensitivity = (err_int5 - err_int6) * numel → 텐서 크기 가중
- **Greedy Selection**: sensitivity/size_saving 비율이 낮은 텐서부터 int5 배정
  - 목표: ~2.0MB 절감 (Exp 165 수준)까지 int5 텐서 누적
- **QAT**: 선정된 텐서의 CastedLinear에 int5 QAT 적용
  - 단, QAT 단계에서는 sensitivity 결과 없으므로 heuristic(layer 0-5)으로 근사

## 예상 결과
- TTT val_bpb: ~1.1185-1.1210 (Exp 165 +0.007 대비 +0.003-0.005 수준)
- artifact size: ~15.8-16.0MB (Exp 165와 유사한 절감)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
