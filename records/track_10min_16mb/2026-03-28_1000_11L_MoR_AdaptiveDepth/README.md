# Exp 150: MoR (Mixture-of-Recursions) — Token-Adaptive Depth

## 목적 (Purpose)
NeurIPS 2025 "Mixture-of-Recursions" 논문 적용.
Weight sharing(재귀 레이어) + 토큰 단위 adaptive depth routing.
쉬운 토큰은 1회, 어려운 토큰은 2회 반복하여 compute 효율화.

## 베이스
Exp 121 (11L NorMuon, val_bpb 1.1183, 17.16MB)

## 변경 사항
- **Recursive Block**: Layer 8-9를 하나의 공유 블록으로 교체 (최대 2회 반복)
  - 실효 depth: 10-12L (토큰별 가변), 고유 파라미터: ~10L
- **Router**: lightweight 1-layer router가 각 토큰의 반복 횟수 결정
  - Router input: 현재 hidden state의 norm + layer output residual norm
  - Binary routing: 반복 1회 vs 2회 (sigmoid threshold)
  - 학습 중: Gumbel-sigmoid로 differentiable routing
- **torch.compile 호환**: dynamic depth 대신 항상 2회 실행 + mask로 선택
  - 모든 토큰에 2회 실행하되, 1회로 충분한 토큰은 첫 결과 사용 (mask blend)

## 예상 결과
- val_bpb: ~1.1170-1.1200 (effective depth 증가 효과)
- artifact size: ~15-16MB (1L 절감)
- step_avg: ~97-100ms (2회 반복 오버헤드)

## 실제 결과
(실행 후 기록)
