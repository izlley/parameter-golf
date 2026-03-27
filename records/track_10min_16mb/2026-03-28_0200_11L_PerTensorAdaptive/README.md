# Exp 142: NorMuon + Per-tensor Adaptive Precision

## 목적
레이어 단위 int5/int6 배분 대신, 텐서 단위 sensitivity analysis로 최적 배분. 양자화 오차가 작은 텐서에만 int5 적용하여 bpb 손실 최소화.

## 베이스
Exp 131 (NorMuon+EWQ, val_bpb 1.1296, 15,554,069B)

## 변경 사항
- Post-training: 각 텐서별 int5/int6 양자화 오차 비교
- Sensitivity가 낮은 텐서부터 int5 적용 (상위 40%)
- QAT: layers 0-4 int5 (근사치, 실제 post-training과 약간 다를 수 있음)

## 예상 결과
- val_bpb: ~1.120-1.125 (per-tensor 최적화로 EWQ 대비 개선)
- 크기: ~15.5-16.0MB (16MB 통과 목표)
- step_avg: ~95ms

## 실제 결과
_(예정)_
