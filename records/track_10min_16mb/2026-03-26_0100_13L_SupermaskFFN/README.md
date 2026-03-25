# Exp 100: Supermask SharedFFN (SupSup-style)

## Purpose
SharedFFN이 일관되게 성능 저하(-0.028~0.078 bpb)를 보이는 문제를 해결.
단일 wide FFN을 공유하되, 원본 SupSup(RAIVNLab/supsup) 방식의 STE 바이너리 마스크로
레이어별 서로 다른 subnetwork를 명확히 선택.

## Base
Exp 97: SharedFFN + UNet (`2026-03-25_2200_13L_SharedFFN_UNet`)

## Approach
- **SupSup-style 바이너리 마스크**: sigmoid 연속 마스크 대신 STE(Straight-Through Estimator) + top-k sparsity
- `_GetSubnet` autograd.Function: forward에서 top-k% → binary {0,1}, backward에서 gradient straight-through
- 마스크 초기화: `uniform(-1, 1)` (원본 SupSup은 kaiming_uniform, 채널 수준이라 단순 uniform 사용)
- Sparsity=0.5 (각 레이어가 hidden channel의 50%를 선택)
- 적용: `leaky_relu → square → binary_mask → proj`
- W_ffn은 1개만 저장, 바이너리 mask는 레이어별로 다른 subnetwork 선택
- BigramHash 12288, U-Net skip connections, mlp_mult=4.0

## Key Changes vs Exp 97 (SharedFFN)
- `_GetSubnet` STE autograd.Function 추가
- `SupermaskSharedMLP`: sigmoid → STE binary mask, kaiming init → uniform(-1,1)
- `SUPERMASK_SPARSITY` env var 추가 (default 0.5)

## Key Changes vs 이전 sigmoid 버전
- 마스크가 {0,1} 바이너리 → 명확한 subnetwork 분리 (sigmoid의 0.3~0.7 모호 구간 제거)
- STE로 gradient 전달 → 바이너리에도 학습 가능
- `scores.abs()` 기반 top-k → score 부호가 아닌 크기로 중요도 판단

## Expected Results
- SW baseline: 1.1300~1.1400 (SharedFFN 베이스보다 개선 기대)
- Size: ~14.5MB (여유 있음)
