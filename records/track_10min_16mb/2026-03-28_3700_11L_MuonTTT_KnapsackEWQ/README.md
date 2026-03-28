# Exp 177: MuonTTT + Knapsack-Optimized EWQ

## 목적 (Purpose)
목표 크기(16MB)를 정확히 맞추는 최적화된 int5 배분.
각 텐서의 (크기 절감량, bpb 비용)을 측정하고, cost/benefit ratio가 낮은 텐서부터
int5를 배정하여 **정확히 목표 크기 이하**가 될 때까지 누적.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **Size Estimation**: 각 텐서를 int5/int6로 양자화 후 zstd 압축 크기 측정
  - 텐서별 `saving = compressed_size(int6) - compressed_size(int5)`
- **Sensitivity Measurement**: per-tensor MSE 차이 (Exp 176과 동일)
- **Knapsack Greedy**: `sensitivity / saving` 비율로 정렬
  - 비율이 낮은(= 크기 대비 bpb 비용이 적은) 텐서부터 int5 배정
  - 누적 saving이 `current_size - 16MB`를 초과하면 중단
- **QAT**: heuristic으로 layer 0-5에 int5 QAT 적용 (sensitivity 기반 최적이 아닐 수 있음)

## 예상 결과
- TTT val_bpb: ~1.1180-1.1200 (최적 배분으로 Exp 165 대비 bpb 개선)
- artifact size: ~15.9-16.0MB (목표 크기에 가장 가깝게 맞춤)
- step_avg: ~95ms (변경 없음, 양자화 시간 약간 증가)

## 실제 결과
(실행 후 기록)
