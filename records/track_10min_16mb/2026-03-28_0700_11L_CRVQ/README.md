# Exp 147: CRVQ (Channel-Relaxed Vector Quantization)

## 목적 (Purpose)
기존 uniform scalar int6 quantization 대신 Vector Quantization으로 압축 방식 교체.
Channel reordering + multi-codebook 구조로 동일 품질에서 weight 크기 40-50% 감소 목표.

## 베이스
Exp 121 (11L NorMuon, val_bpb 1.1183, 17.16MB)

## 변경 사항
- **학습**: 기존과 완전 동일 (NorMuon, 11L, etc.)
- **저장 시 압축 교체**: int6+zstd 대신 CRVQ+zstd 적용
  - Channel reordering: weight matrix의 row를 유사도 기준 정렬하여 codebook 효율 극대화
  - Multi-codebook VQ: 2개 codebook (primary + residual)으로 sub-4-bit 표현
  - Vector dimension: 4 (4개 weight를 하나의 vector로 묶어 quantize)
  - Codebook size: 256 entries per codebook (8-bit index)
- **역양자화**: codebook lookup으로 weight 복원

## 예상 결과
- val_bpb: ~1.1190-1.1200 (학습 동일, 양자화 손실 약간 다를 수 있음)
- artifact size: ~10-12MB (현재 대비 ~40% 감소)
- step_avg: ~95ms (학습 영향 없음)

## 실제 결과
(실행 후 기록)
