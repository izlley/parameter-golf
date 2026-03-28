# Exp 171: MuonTTT + Entropy Regularization

## 목적 (Purpose)
학습 중 weight entropy penalty를 추가하여 weight 분포를 압축 친화적으로 변환.
Weight가 더 적은 distinct 값에 집중되면 zstd 압축률 대폭 향상.
BackSlash (ICML 2025), HEMP 논문 기반.

## 베이스
Exp 162 (11L MuonTTT, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- **Entropy Loss**: 학습 loss에 `lambda * H(weights)` 항 추가
  - H(weights): per-layer weight histogram (64 bins) → softmax → Shannon entropy
  - lambda: 0에서 시작, warmdown (마지막 30%) 동안 ramp-up
  - QAT와 동시 적용하여 양자화 grid에 weight 집중 유도
- **구현**: 매 N step마다 entropy 계산 (매 step은 overhead 과도)
  - entropy_interval: 10 steps
  - lambda_max: 0.001 (tunable)

## 예상 결과
- TTT val_bpb: ~1.1160-1.1180 (entropy loss에 의한 소폭 품질 손실)
- artifact size: ~15.5-16.5MB (zstd 압축률 15-25% 향상 기대)
- step_avg: ~96-97ms (entropy 계산 overhead)

## 실제 결과
(실행 후 기록)
