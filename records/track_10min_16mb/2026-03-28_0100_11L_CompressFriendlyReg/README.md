# Exp 141: NorMuon + Compression-friendly Regularization

## 목적
NorMuon의 uniform weight 분포가 zstd 압축을 방해하는 문제를 해결. L1 regularization으로 weight sparsity를 유도하여 압축률 개선.

## 베이스
Exp 121 (NorMuon, val_bpb 1.1183, 17,158,779B — 크기 초과)

## 변경 사항
- L1 regularization 추가: lambda=0.0001, 큰 weight matrix(ndim==2, >65K elements)에만 적용
- 학습 중 weight가 0 근처로 집중되도록 유도 → zstd 압축률 향상

## 예상 결과
- val_bpb: ~1.119 (L1으로 소폭 악화 가능)
- 크기: ~16.0-16.5MB (sparsity 유도로 압축률 개선)
- step_avg: ~95-96ms (L1 계산 오버헤드 미미)

## 실제 결과
_(예정)_
