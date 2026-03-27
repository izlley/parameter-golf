# Exp 151: Attention Dictionary Learning — Q/K/V 공유 원자 분해

## 목적 (Purpose)
"Share Your Attention" (Feb 2026) 논문 적용.
모든 레이어의 Q/K/V projection 행렬을 K개의 공유 dictionary atom 가중합으로 구성.
Attention 파라미터 ~60% 감소로 artifact size 대폭 축소.

## 베이스
Exp 121 (11L NorMuon, val_bpb 1.1183, 17.16MB)

## 변경 사항
- **Dictionary Atoms**: 6개의 공유 atom matrix (dim x dim) — 모든 레이어가 공유
- **Per-layer Mixing**: 각 레이어의 Q/K/V/O가 atom의 가중합으로 구성
  - W_q[layer] = sum(alpha_q[layer][k] * atom[k]) for k in range(K)
  - alpha는 learnable scalar (11L × 4projections × 6atoms = 264 scalars)
- **파라미터 절감**: 기존 Attention weight = 11L × 4 × (512×512) ≈ 11.5M
  → Dictionary: 6 × (512×512) = 1.57M + mixing scalars 264개 ≈ 1.57M
  → ~86% 감소 (극단적이므로 atom 수 조정 필요할 수 있음)
- MLP는 기존 그대로 유지 (독립 파라미터)

## 예상 결과
- val_bpb: ~1.1250-1.1350 (attention 표현력 손실 있을 수 있음)
- artifact size: ~10-12MB (attention 파라미터 대폭 감소)
- step_avg: ~95-96ms (atom lookup은 빠름)

## 실제 결과
(실행 후 기록)
